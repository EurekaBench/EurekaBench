import ast
import importlib.util
import json
import os
import sys
import threading
import time
import traceback

import fire
import h5py
import numpy as np
from archive import DreamsSimulator, trailing_int

SNAPSHOT = 90
HUBBLE = 0.6909
MW_MASS_RANGE_MSUN = (7e11, 2.5e12)
SELECT_FIELDS = ["SnapNum", "SubhaloID", "FirstSubhaloInFOFGroupID",
                 "LastProgenitorID", "GroupMassType"]


def load_param_rows(sim):
    path = sim.params_file()
    lines = [l.strip() for l in open(path) if l.strip()]
    header = lines[0].lstrip("#").replace(",", " ").split()
    rows = [dict(zip(header, map(float, l.replace(",", " ").split())))
            for l in lines[1:]]
    return header, rows


def locate_mw_subtree(data):
    sn = data["SnapNum"]
    z0 = np.flatnonzero(sn == SNAPSHOT)
    if len(z0) == 0:
        return None
    gmt = data["GroupMassType"][z0].astype(np.float64)
    tot_msun = gmt.sum(1) * 1e10 / HUBBLE
    contam = gmt[:, 2] / np.maximum(gmt.sum(1), 1e-20)
    win = (tot_msun > MW_MASS_RANGE_MSUN[0]) & (tot_msun < MW_MASS_RANGE_MSUN[1])
    if not win.any():
        return None
    best = np.argmin(contam[win])
    target = data["FirstSubhaloInFOFGroupID"][z0[win][best]]
    root_rows = np.flatnonzero(data["SubhaloID"] == target)
    if len(root_rows) != 1:
        return None
    r = int(root_rows[0])
    lo, hi = int(data["SubhaloID"][r]), int(data["LastProgenitorID"][r])
    rows = np.arange(r, r + hi - lo + 1)
    if rows[-1] >= len(sn) or not np.array_equal(
            data["SubhaloID"][rows], np.arange(lo, hi + 1)):
        rows = np.flatnonzero((data["SubhaloID"] >= lo) & (data["SubhaloID"] <= hi))
    return dict(rows=rows,
                mw_group_mass_msun=np.float64(tot_msun[win][best]),
                contamination=np.float64(contam[win].min()))


MAX_STATS = 8
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
STATS_TIMEOUT = 7200.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
MIN_EVAL_SIMS = 20
ACCESS_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessed_sims.json")

LINK_FIELDS = ["SubhaloID", "DescendantID", "FirstProgenitorID",
               "NextProgenitorID", "MainLeafProgenitorID", "LastProgenitorID",
               "SnapNum"]

REFERENCE_LEVELS = {
    "prior_full_readout": {
        "r2": 0.9608, "rmse": 0.0306,
        "what": "the prior study's message-passing network on trees whose nodes "
                "carry snapshot, DM, stellar, gas mass and SFR; the measured "
                "black-box ceiling under this tree definition"},
    "prior_structure_readout": {
        "r2": 0.8335, "rmse": 0.0632,
        "what": "the same network when every node carries only its snapshot "
                "number"},
    "reference_study_gnn": {
        "r2_structure_only": 0.708, "r2_best": 0.957,
        "what": "the graph neural network of arXiv:2511.05367 on the same suite "
                "(its rmse values are quoted in its own normalized units and are "
                "not comparable to keV^-1)"},
}
REFERENCE = {
    "PH1": ("r2", ">=", 0.8335),
    "PH2": ("rmse", "<=", 0.0632),
    "PH3": ("r2", ">=", 0.9608),
    "PH4": ("rmse", "<=", 0.0306),
}

PA_IDS = ["PA1", "PA2", "PA3", "PA4"]


def clip01(x):
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, float(x))))


def pa_scores(in_suite, refit, reduced):
    r2_full = clip01(in_suite.get("r2"))
    r2_mb = clip01((reduced.get("main_branch") or {}).get("r2"))
    r2_fl = (reduced.get("flattened") or {}).get("r2")
    r2_fl = 0.0 if r2_fl is None or not np.isfinite(r2_fl) else float(r2_fl)
    return {
        "PA1": round(r2_full, 4),
        "PA2": round(clip01(refit.get("r2")), 4),
        "PA3": round(clip01(r2_full - r2_mb), 4),
        "PA4": round(clip01(min(r2_fl, r2_full)), 4),
    }


def progress(label, done, total, t0, note=""):
    elapsed = time.time() - t0
    frac = done / total if total else 1.0
    filled = int(round(30 * frac))
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    print(f"[{label}] |{'#' * filled}{'.' * (30 - filled)}| {done}/{total} "
          f"elapsed {elapsed:5.0f}s eta {eta:5.0f}s {note}", flush=True)


def call_with_timeout(fn, timeout, *args):
    out, error = [], []

    def run():
        try:
            out.append(fn(*args))
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    t0 = time.time()
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} exceeded {timeout:.0f}s")
    if error:
        raise error[0]
    return out[0], time.time() - t0


def is_main_guard(node):
    test = node.test
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__"
                    for c in test.comparators))


class ImportScan(ast.NodeVisitor):
    def __init__(self):
        self.modules = []

    def visit_If(self, node):
        if not is_main_guard(node):
            self.generic_visit(node)

    def visit_Import(self, node):
        self.modules += [a.name.split(".")[0] for a in node.names]

    def visit_ImportFrom(self, node):
        if node.module:
            self.modules.append(node.module.split(".")[0])


def load_mechanism(mechanism_path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("agent_mechanism", mechanism_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blocking, problems = [], []
    for name in ("PROPERTIES", "STATISTICS", "COEFFS",
                 "compute_statistics", "predict_inv_mwdm", "fit_coeffs"):
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
    if hasattr(module, "PROPERTIES") and not all(
            isinstance(p, str) for p in module.PROPERTIES):
        blocking.append("PROPERTIES entries are not all strings")
    if hasattr(module, "STATISTICS") and len(list(module.STATISTICS)) > MAX_STATS:
        problems.append(f"STATISTICS has {len(list(module.STATISTICS))} entries, "
                        f"limit {MAX_STATS}")
    if hasattr(module, "COEFFS") and len(list(module.COEFFS)) > MAX_COEFFS:
        problems.append(f"COEFFS has {len(list(module.COEFFS))} entries, "
                        f"limit {MAX_COEFFS}")
    with open(mechanism_path) as f:
        scan = ImportScan()
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy that the model can reach: {bad}")
    return module, blocking, problems


def read_tree(path, fields):
    with h5py.File(path, "r") as f:
        missing = [n for n in set(fields) | set(SELECT_FIELDS) if n not in f]
        if missing:
            raise KeyError(f"fields {missing} are not in {path}")
        select = {k: f[k][...] for k in SELECT_FIELDS}
        located = locate_mw_subtree(select)
        if located is None:
            raise RuntimeError("no Milky-Way host tree under the task definition")
        rows = located["rows"]
        tree = {}
        for name in fields:
            if name in select:
                tree[name] = select[name][rows]
            else:
                tree[name] = f[name][...][rows]
    return tree


def main_branch_variant(tree):
    ids = tree["SubhaloID"]
    fp = tree["FirstProgenitorID"]
    pos = {int(i): k for k, i in enumerate(ids)}
    keep = []
    cur = 0
    while cur is not None:
        keep.append(cur)
        nxt = int(fp[cur])
        cur = pos.get(nxt) if nxt != -1 else None
    keep = np.asarray(keep, dtype=int)
    return {k: v[keep] for k, v in tree.items()}


def flattened_variant(tree):
    out = dict(tree)
    ids = np.asarray(tree["SubhaloID"])
    n = len(ids)
    desc = np.concatenate([[-1], ids[:-1]])
    fp = np.concatenate([ids[1:], [-1]])
    out["DescendantID"] = desc.astype(ids.dtype)
    out["FirstProgenitorID"] = fp.astype(ids.dtype)
    out["NextProgenitorID"] = np.full(n, -1, dtype=ids.dtype)
    out["MainLeafProgenitorID"] = np.full(n, ids[-1], dtype=ids.dtype)
    out["LastProgenitorID"] = np.full(n, ids[-1], dtype=ids.dtype)
    return out


def stats_vector(mech, tree):
    stats = np.asarray(mech.compute_statistics(tree), dtype=np.float64).reshape(-1)
    if stats.shape != (len(list(mech.STATISTICS)),):
        raise RuntimeError(f"compute_statistics returned shape {stats.shape}, "
                           f"expected ({len(list(mech.STATISTICS))},)")
    return stats


def compute_metrics(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
        return {"r2": None, "rmse": None, "mre": None,
                "error": "predictions are not a finite array of one value per simulation"}
    return {
        "r2": float(1 - np.mean((pred - truth) ** 2) / np.var(truth)),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mre": float(np.mean(np.abs(pred - truth) / truth)),
    }


def per_simulation_check(mech, stats, coeffs, pred, n_probe=32, seed=0, tolerance=1e-9):
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(pred), size=min(n_probe, len(pred)), replace=False)
    worst = 0.0
    for i in probe:
        alone = np.asarray(mech.predict_inv_mwdm(stats[i:i + 1], coeffs),
                           dtype=np.float64).reshape(-1)
        worst = max(worst, abs(float(alone[0]) - float(pred[i])))
    return {"n_probed": int(len(probe)), "max_difference": worst,
            "passed": bool(worst <= tolerance)}


def predict_metrics(mech, stats, truth, coeffs):
    try:
        pred, seconds = call_with_timeout(mech.predict_inv_mwdm, PREDICT_TIMEOUT,
                                          stats, coeffs)
        out = compute_metrics(pred, truth)
        out["predict_seconds"] = round(seconds, 3)
        return out, np.asarray(pred, np.float64).reshape(-1)
    except Exception as exc:
        return {"r2": None, "rmse": None, "mre": None, "error": str(exc)}, None


def reference_verdicts(metrics):
    out = {}
    for cid, (name, op, level) in REFERENCE.items():
        value = None if metrics is None else metrics.get(name)
        if value is None:
            out[cid] = False
        elif op == ">=":
            out[cid] = bool(value >= level)
        else:
            out[cid] = bool(value <= level)
    return out


def main(mechanism, output, data_dir=None, n_sims=256, seed=0, accessed_file=ACCESS_RECORD, predict_timeout=None,
         fit_timeout=None, stats_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT, STATS_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    if stats_timeout is not None:
        STATS_TIMEOUT = float(stats_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s, fit {FIT_TIMEOUT:.0f}s, "
          f"statistics {STATS_TIMEOUT:.0f}s", flush=True)
    mechanism_path = str(mechanism)
    report = {
        "mechanism_path": mechanism_path,
        "protocol": {
            "reference_study": "arXiv:2511.05367",
            "held_out_set": "every usable simulation of the DREAMS WDM Milky-Way "
                            "zoom-in suite whose data the run never downloaded, "
                            "read from accessed_sims.json recorded during the run",
            "tree": "the SubLink subtree rooted at the z=0 central subhalo of the "
                    "least low-resolution-contaminated FOF group with total mass "
                    "between 7e11 and 2.5e12 Msun, exactly the task definition; "
                    "nodes delivered in depth-first order with the linkage fields "
                    f"{LINK_FIELDS} always included",
            "prediction": "per simulation: compute_statistics over the nodes of its "
                          "Milky-Way tree, then predict_inv_mwdm",
            "metrics": "r2 against 1/M_WDM in keV^-1, taken about the mean of the "
                       "true values (rmse and the mean relative error are reported "
                       "alongside but not scored); the true value is uniform over "
                       "the sampled range, so always predicting the prior mean "
                       "gives r2 = 0 and an exact prediction gives r2 = 1",
            "passes": "once with the submitted COEFFS on the full held-out set, "
                      "once with constants refit by the submission's own fit_coeffs "
                      "on a random half of the held-out set (functional form "
                      "frozen) and scored on the disjoint other half",
            "reduced_forms": {
                "main_branch": "each held-out tree reduced to the chain of first "
                               "progenitors of the root; the recovery of a true "
                               "mechanism collapses there (the reference study "
                               "measures r2 = 0.073)",
                "flattened": "each held-out tree with its merger linkage replaced "
                             "by a chain in storage order, nodes and times kept; "
                             "the recovery of a true mechanism survives (the "
                             "reference study measures 0.873 against 0.708)",
                "note": "both reduced forms are evaluated with the submitted "
                        "constants; the linkage fields of the main-branch form "
                        "are the stored values of the kept nodes",
            },
            "scores": {
                "PA1": "r2 with the submitted constants, clipped to 0 and 1",
                "PA2": "r2 of the refit pass on the disjoint half, clipped to 0 and 1",
                "PA3": "r2 on the full trees minus r2 on the main branch alone, "
                       "each floored at 0, clipped to 0 and 1",
                "PA4": "r2 with the merger linkage replaced by a chain, capped at "
                       "the full-tree r2, clipped to 0 and 1",
            },
            "reference_levels": REFERENCE_LEVELS,
            "seed": seed,
        },
        "pa_scores": {}, "reference_criteria": {},
        "reference_note": "comparison against the measured black-box readouts of "
                          "the prior study and the reference study's network; "
                          "reference only, never counted in any score",
    }

    mech = None
    if not os.path.exists(mechanism_path):
        report["error"] = "mechanism.py not found"
    else:
        try:
            mech, blocking, problems = load_mechanism(mechanism_path)
        except Exception:
            mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
        report["interface_problems"] = problems
        report["blocking_problems"] = blocking
        for problem in problems:
            print(f"[eval] contract violation, reported and not scored here: {problem}",
                  flush=True)
        if mech is None or blocking:
            mech = None
            report["error"] = "mechanism.py cannot be evaluated: " + "; ".join(blocking)
    if report.get("error"):
        print(f"[eval] ERROR: {report['error']}", flush=True)

    if mech is not None:
        if str(accessed_file).strip().lower() in ("none", "ignore"):
            accessed = set()
            report["exclusion_waived"] = True
            report["protocol"]["exclusion_note"] = (
                "access-based exclusion waived by the operator: the metrics are "
                "computed over simulations the agent saw during its investigation "
                "and the in-suite pass is not blind; the refit pass still scores "
                "on a half disjoint from the fitting half")
            print("[eval] WARNING: access-based exclusion waived; the metrics are "
                  "not blind for this run")
        elif accessed_file == ACCESS_RECORD and not os.path.isfile(accessed_file):
            accessed = set()
            print("[eval] no access record: the run read no simulation, so none is excluded")
        else:
            if not os.path.isfile(accessed_file):
                raise RuntimeError(
                    f"{accessed_file} not found; the held-out set is the complement "
                    "of the simulations the run accessed, so the evaluation cannot "
                    "proceed without it")
            with open(accessed_file) as f:
                accessed = set(json.load(f).get("boxes", []))
            print(f"[eval] excluding {len(accessed)} simulations the run accessed")

        sim = DreamsSimulator(data_dir=data_dir)
        rng = np.random.default_rng(seed)
        suite, names = sim.sim_names()
        columns, rows = load_param_rows(sim)
        if len(rows) != len(names):
            raise RuntimeError(f"parameter table has {len(rows)} rows for "
                               f"{len(names)} usable simulations")
        inv = {name: row["WDM"] for name, row in zip(names, rows)}
        report["protocol"]["target_note"] = ("the WDM column of the parameter "
                                             "table, 1/M_WDM in keV^-1")

        eligible = [n for n in names if n not in accessed]
        if n_sims and n_sims < len(eligible):
            print(f"[eval] WARNING: sampling {n_sims} of {len(eligible)} eligible "
                  f"simulations instead of the full complement")
            eligible = sorted(rng.choice(eligible, size=n_sims, replace=False),
                              key=trailing_int)
        report["n_excluded_accessed"] = len(accessed)
        report["n_eval_sims_requested"] = len(eligible)

        fields = list(dict.fromkeys(list(mech.PROPERTIES) + LINK_FIELDS))
        variants = ["full", "main_branch", "flattened"]
        stats = {v: [] for v in variants}
        truth, used, n_nodes = [], [], []
        t_stats_max = 0.0
        t0 = time.time()
        for k, name in enumerate(eligible):
            try:
                tree = read_tree(sim.tree_path(name), fields)
                forms = {"full": tree,
                         "main_branch": main_branch_variant(tree),
                         "flattened": flattened_variant(tree)}
                vecs = {}
                for v in variants:
                    vecs[v], seconds = call_with_timeout(stats_vector, STATS_TIMEOUT,
                                                         mech, forms[v])
                    if v == "full":
                        t_stats_max = max(t_stats_max, seconds)
            except Exception as exc:
                print(f"  [skip] {name}: {exc}", flush=True)
                continue
            for v in variants:
                stats[v].append(vecs[v])
            truth.append(inv[name])
            used.append(name)
            n_nodes.append(len(tree["SnapNum"]))
            if (k + 1) % 50 == 0 or (k + 1) == len(eligible):
                progress("stats", k + 1, len(eligible), t0, f"{len(used)} usable")
        if len(used) < MIN_EVAL_SIMS:
            report["error"] = f"only {len(used)} usable held-out simulations"
            print(f"[eval] ERROR: {report['error']}", flush=True)
        else:
            stats = {v: np.vstack(s) for v, s in stats.items()}
            truth = np.asarray(truth, np.float64)
            n_nodes = np.asarray(n_nodes)
            report["n_eval_sims_used"] = int(len(used))
            report["eval_sims"] = list(used)
            report["var_true_inv_mwdm"] = round(float(np.var(truth)), 6)
            report["nodes_per_tree"] = {"min": int(n_nodes.min()),
                                        "median": int(np.median(n_nodes)),
                                        "max": int(n_nodes.max())}

            coeffs = list(mech.COEFFS)
            in_suite, pred = predict_metrics(mech, stats["full"], truth, coeffs)
            in_suite["stats_seconds_max_per_tree"] = round(t_stats_max, 3)
            if pred is not None:
                in_suite["per_simulation"] = per_simulation_check(
                    mech, stats["full"], coeffs, pred, seed=seed)
                if not in_suite["per_simulation"]["passed"]:
                    print("  [note] predict_inv_mwdm does not predict one "
                          "simulation at a time: a simulation's value changes "
                          f"by up to {in_suite['per_simulation']['max_difference']:.3g} "
                          "when it is evaluated on its own; recorded for the "
                          "judge, the metrics are still computed", flush=True)
            report["in_suite"] = in_suite
            print(f"[eval] in-suite ({len(used)} sims, submitted constants): "
                  f"r2={in_suite['r2']} rmse={in_suite['rmse']} mre={in_suite['mre']}",
                  flush=True)

            reduced = {}
            for v in ["main_branch", "flattened"]:
                reduced[v], _ = predict_metrics(mech, stats[v], truth, coeffs)
                print(f"[eval] {v} (submitted constants): r2={reduced[v]['r2']} "
                      f"rmse={reduced[v]['rmse']}", flush=True)
            report["reduced_forms"] = reduced

            half = len(truth) // 2
            fit_sel = np.zeros(len(truth), dtype=bool)
            fit_sel[rng.permutation(len(truth))[:half]] = True
            refit = {"n_fit": int(fit_sel.sum()), "n_test": int((~fit_sel).sum())}
            try:
                coeffs2, seconds = call_with_timeout(
                    mech.fit_coeffs, FIT_TIMEOUT,
                    stats["full"][fit_sel], truth[fit_sel], list(mech.COEFFS))
                coeffs2 = [float(c) for c in coeffs2]
                if len(coeffs2) > MAX_COEFFS:
                    violation = (f"fit_coeffs returned {len(coeffs2)} constants, "
                                 f"limit {MAX_COEFFS}")
                    refit["contract_violation"] = violation
                    problems.append(violation)
                    report["interface_problems"] = problems
                    print(f"[eval] contract violation, recorded and scored anyway: "
                          f"{violation}", flush=True)
                refit["fit_seconds"] = round(seconds, 3)
                refit["coeffs"] = [round(c, 8) for c in coeffs2]
                metrics2, _ = predict_metrics(mech, stats["full"][~fit_sel],
                                              truth[~fit_sel], coeffs2)
                refit.update(metrics2)
            except Exception as exc:
                refit.update({"r2": None, "rmse": None, "mre": None, "error": str(exc)})
            report["refit"] = refit
            print(f"[eval] refit ({refit['n_fit']} fit / {refit['n_test']} test sims): "
                  f"r2={refit['r2']} rmse={refit['rmse']} mre={refit['mre']}",
                  flush=True)

            report["pa_scores"] = pa_scores(in_suite, refit, reduced)
            report["reference_criteria"] = reference_verdicts(in_suite)

    for pid in PA_IDS:
        report["pa_scores"].setdefault(pid, 0.0)
    if not report["reference_criteria"]:
        report["reference_criteria"] = reference_verdicts(None)
    report["predictive_accuracy"] = round(
        float(np.mean([report["pa_scores"][p] for p in PA_IDS])), 4)
    report["predictive_accuracy_total"] = len(PA_IDS)
    report["reference_score"] = sum(1 for v in report["reference_criteria"].values() if v)
    report["reference_total"] = len(REFERENCE)

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 64)
    print(f"[eval] predictive accuracy {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    print(f"[eval] vs the measured readout levels (not scored): "
          f"{report['reference_score']}/{report['reference_total']}")
    if report.get("interface_problems"):
        print(f"[eval] contract violations recorded for the judge: "
              f"{report['interface_problems']}")
    if report.get("error"):
        print(f"[eval] evaluation incomplete: {report['error']}")
    print(f"[eval] results: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
