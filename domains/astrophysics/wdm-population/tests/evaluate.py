import ast
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import traceback

import fire
import h5py
import numpy as np
from archive import DreamsSimulator, trailing_int

PROBLEM = "wdm_population"
SNAPSHOT = 90
MAX_STATS = 8
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
STATS_TIMEOUT = 7200.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
MIN_EVAL_SIMS = 20
ACCESS_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessed_sims.json")

PA_IDS = ["PA1", "PA2", "PA3"]

REFERENCE_LEVELS = {
    "deepset_without_true_omega_m": {
        "r2": 0.951, "rmse": 0.031, "mre": 0.130,
        "what": "DeepSet + normalizing flow on subhalo-level properties plus the "
                "mean and standard deviation of 14 subhalo properties, without "
                "the true Omega_m; the best published black-box level"},
    "best_symbolic_formula": {
        "rmse": 0.066,
        "what": "the best published closed-form symbolic-regression formula, "
                "built from two gas-mass statistics of the simulation"},
    "gas_only_network_without_omega_m": {
        "r2": 0.814, "rmse": 0.060,
        "what": "a network reading the statistics of the single most informative "
                "property (the subhalo gas mass) without the true Omega_m"},
}
REFERENCE = {
    "PH1": ("r2", ">=", 0.951),
    "PH2": ("rmse", "<=", 0.031),
    "PH3": ("mre", "<=", 0.130),
    "PH4": ("rmse", "<=", 0.066),
    "PH5": ("r2", ">=", 0.814),
    "PH6": ("rmse", "<=", 0.060),
}


def floatable(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_param_table(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    header = None
    if lines[0].startswith("#"):
        header, lines = lines[0].lstrip("#").split(), lines[1:]
    elif not all(floatable(t) for t in lines[0].replace(",", " ").split()):
        header, lines = lines[0].replace(",", " ").split(), lines[1:]
    if header is None:
        raise RuntimeError(f"{path} has no header line naming its columns")
    rows = {}
    for row_idx, line in enumerate(lines):
        toks = line.replace(",", " ").split()
        offset = len(toks) - len(header)
        name_id = trailing_int(toks[0]) if offset == 1 and not floatable(toks[0]) else None
        sim_id = name_id if name_id is not None else row_idx
        rows[sim_id] = {c: float(v) for c, v in zip(header, toks[max(offset, 0):])}
    return header, rows


def wdm_targets(columns, rows):
    wdm_cols = [c for c in columns
                if "wdm" in c.lower() or c.lower() in ("mdm", "m_dm")]
    if len(wdm_cols) != 1:
        raise RuntimeError(f"cannot identify the WDM column among {columns}")
    col = wdm_cols[0]
    raw = {i: r[col] for i, r in rows.items()}
    lo, hi = min(raw.values()), max(raw.values())
    if 0.02 <= lo and hi <= 0.7:
        return raw, f"column '{col}' read as 1/M_WDM [keV^-1]"
    if 1.5 <= lo and hi <= 35:
        return {i: 1.0 / v for i, v in raw.items()}, f"column '{col}' read as M_WDM [keV], inverted"
    raise RuntimeError(f"column '{col}' has range [{lo:.4g}, {hi:.4g}], matching "
                       "neither M_WDM in keV nor 1/M_WDM in keV^-1")


def read_fields(paths, group, names):
    parts = {n: [] for n in names}
    for p in paths:
        with h5py.File(p, "r") as f:
            if group not in f:
                continue
            g = f[group]
            missing = [n for n in names if n not in g]
            if missing:
                raise KeyError(f"fields {missing} are not in {p}")
            for n in names:
                parts[n].append(g[n][...])
    if not parts[names[0]]:
        raise RuntimeError(f"group {group!r} holds no data across {paths}")
    return {n: np.concatenate(v) for n, v in parts.items()}


def progress(label, done, total, t0, note=""):
    elapsed = time.time() - t0
    frac = done / total if total else 1.0
    filled = int(round(30 * frac))
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    print(f"[{label}] |{'#' * filled}{'.' * (30 - filled)}| {done}/{total} "
          f"elapsed {elapsed:5.0f}s eta {eta:5.0f}s {note}", flush=True)


def call_with_timeout(fn, timeout, *args):
    ctx = multiprocessing.get_context("fork")
    receive, send = ctx.Pipe(duplex=False)

    def run():
        try:
            send.send((True, fn(*args)))
        except BaseException as exc:
            send.send((False, f"{type(exc).__name__}: {exc}"))
        finally:
            send.close()

    worker = ctx.Process(target=run, daemon=True)
    start = time.time()
    worker.start()
    send.close()
    name = getattr(fn, "__name__", str(fn))
    ready = False
    while time.time() - start < timeout:
        ready = receive.poll(min(HEARTBEAT, timeout - (time.time() - start)))
        if ready:
            break
        print(f"[eval]   ... {name} still running, {time.time() - start:.0f}s of "
              f"{timeout:.0f}s", flush=True)
    if not ready:
        worker.terminate()
        worker.join(5)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        receive.close()
        raise TimeoutError(f"{name} exceeded {timeout:.0f}s")
    try:
        ok, payload = receive.recv()
    except EOFError:
        worker.join(5)
        raise RuntimeError(f"{name} died without returning a result "
                           f"(exit code {worker.exitcode})")
    finally:
        receive.close()
        worker.join(5)
        if worker.is_alive():
            worker.kill()
    if not ok:
        raise RuntimeError(f"{name} raised {payload}")
    return payload, time.time() - start


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


def stats_vector(mech, props):
    stats = np.asarray(mech.compute_statistics(props), dtype=np.float64).reshape(-1)
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
        gap = abs(float(alone[0]) - float(pred[i]))
        worst = max(worst, gap if np.isfinite(gap) else np.inf)
    return {"n_probed": int(len(probe)), "max_difference": worst,
            "passed": bool(worst <= tolerance)}


def predict_each(mech, stats, coeffs):
    return np.array([np.asarray(mech.predict_inv_mwdm(stats[i:i + 1], coeffs),
                                dtype=np.float64).reshape(-1)[0]
                     for i in range(len(stats))])


def predict_scored(mech, stats, coeffs, seed):
    pred, seconds = call_with_timeout(mech.predict_inv_mwdm, PREDICT_TIMEOUT,
                                      stats, coeffs)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    info = {"predict_seconds": round(seconds, 3)}
    if pred.shape != (len(stats),):
        return pred, info
    try:
        check, _ = call_with_timeout(per_simulation_check, PREDICT_TIMEOUT,
                                     mech, stats, coeffs, pred, 32, seed)
    except Exception as exc:
        check = {"passed": False, "error": str(exc)}
    info["per_simulation"] = check
    if not check["passed"]:
        print("  [note] predict_inv_mwdm does not predict one simulation at a time "
              f"({check.get('error') or 'max difference %.3g' % check['max_difference']}); "
              "every simulation is scored from its own call", flush=True)
        pred, seconds = call_with_timeout(predict_each, PREDICT_TIMEOUT,
                                          mech, stats, coeffs)
        info["scored_one_simulation_at_a_time"] = True
        info["predict_each_seconds"] = round(seconds, 3)
    return pred, info


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


def floored(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return round(min(1.0, max(0.0, float(value))), 4)


def pa_scores(in_suite, refit):
    mre = in_suite.get("mre")
    return {"PA1": floored(in_suite.get("r2")),
            "PA2": floored(None if mre is None else 1.0 - mre),
            "PA3": floored(refit.get("r2"))}


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
        "problem": PROBLEM,
        "mechanism_path": mechanism_path,
        "seed": seed,
        "protocol": {
            "reference_study": "arXiv:2510.05037",
            "held_out_set": "every simulation of the DREAMS WDM box set whose "
                            "catalog the run never downloaded, read from "
                            "accessed_sims.json recorded during the run",
            "prediction": "per simulation: compute_statistics over all subhalos of "
                          "the z=0 catalog, then predict_inv_mwdm; when a probed "
                          "simulation predicted on its own differs from the batch "
                          "call, every simulation is scored from its own call",
            "metrics": "r2 and the mean relative error mean(|pred - true| / true) "
                       "against 1/M_WDM in keV^-1; rmse is recorded for the "
                       "reference comparison only",
            "pa_scores": {
                "PA1": "r2 with the submitted COEFFS on the full held-out set, "
                       "floored at 0",
                "PA2": "1 - mean relative error with the submitted COEFFS on the "
                       "full held-out set, floored at 0",
                "PA3": "r2 on the other half after the submission's own fit_coeffs "
                       "refits the constants on a random half of the held-out set "
                       "(functional form frozen), floored at 0"},
            "reference_levels": REFERENCE_LEVELS,
            "seed": seed,
        },
        "pa_scores": {pid: 0.0 for pid in PA_IDS},
        "predictive_accuracy": 0.0,
        "reference_criteria": {},
        "reference_note": "comparison against the levels the reference study "
                          "published (its best black-box without the true Omega_m, "
                          "its best closed-form formula, and its single-property "
                          "network); reference only, never counted in any score",
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
            print(f"[eval] contract violation, recorded and not scored here: {problem}",
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
                "computed over simulations the agent may have seen during its "
                "investigation, so the pass with the submitted constants is not "
                "blind; the refit pass still scores on a half disjoint from the "
                "fitting half")
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
        info = sim.list_simulations()
        columns, rows = parse_param_table(sim.params_file())
        if len(rows) != info["n_simulations"]:
            raise RuntimeError(f"parameter table has {len(rows)} rows for "
                               f"{info['n_simulations']} simulations")
        inv, unit_note = wdm_targets(columns, rows)
        report["protocol"]["target_note"] = unit_note

        eligible = [n for n in info["simulations"]
                    if n not in accessed and trailing_int(n) in inv]
        if n_sims and n_sims < len(eligible):
            print(f"[eval] WARNING: sampling {n_sims} of {len(eligible)} eligible "
                  f"simulations instead of the full complement")
            eligible = sorted(rng.choice(eligible, size=n_sims, replace=False),
                              key=trailing_int)
        report["n_excluded_accessed"] = len(accessed)
        report["n_eval_sims_requested"] = len(eligible)

        properties = list(mech.PROPERTIES)
        stats, truth, used = [], [], []
        t_stats, t_stats_max = 0.0, 0.0
        t0 = time.time()
        for k, name in enumerate(eligible):
            try:
                props = read_fields(sim.catalog_paths(name, SNAPSHOT),
                                    "Subhalo", properties)
                s, seconds = call_with_timeout(stats_vector, STATS_TIMEOUT, mech, props)
            except Exception as exc:
                print(f"  [skip] {name}: {exc}", flush=True)
                continue
            t_stats += seconds
            t_stats_max = max(t_stats_max, seconds)
            stats.append(s)
            truth.append(inv[trailing_int(name)])
            used.append(name)
            if (k + 1) % 50 == 0 or (k + 1) == len(eligible):
                progress("stats", k + 1, len(eligible), t0, f"{len(used)} usable")
        if len(used) < MIN_EVAL_SIMS:
            report["error"] = f"only {len(used)} usable held-out simulations"
            print(f"[eval] ERROR: {report['error']}", flush=True)
        else:
            stats = np.vstack(stats)
            truth = np.asarray(truth, np.float64)
            report["n_eval_sims_used"] = int(len(used))
            report["eval_sims"] = list(used)
            report["var_true_inv_mwdm"] = round(float(np.var(truth)), 6)
            report["statistics_summary"] = [
                {"name": str(name),
                 "min": round(float(np.min(stats[:, i])), 6),
                 "median": round(float(np.median(stats[:, i])), 6),
                 "max": round(float(np.max(stats[:, i])), 6),
                 "mean": round(float(np.mean(stats[:, i])), 6),
                 "std": round(float(np.std(stats[:, i])), 6)}
                for i, name in enumerate(list(mech.STATISTICS)[:stats.shape[1]])]

            try:
                pred, detail = predict_scored(mech, stats, list(mech.COEFFS), seed)
                in_suite = compute_metrics(pred, truth)
                in_suite.update(detail)
                in_suite["stats_seconds_total"] = round(t_stats, 1)
                in_suite["stats_seconds_max_per_catalog"] = round(t_stats_max, 3)
            except Exception as exc:
                in_suite = {"r2": None, "rmse": None, "mre": None, "error": str(exc)}
            report["in_suite"] = in_suite
            print(f"[eval] in-suite ({len(used)} sims, submitted constants): "
                  f"r2={in_suite['r2']} rmse={in_suite['rmse']} mre={in_suite['mre']}",
                  flush=True)

            half = len(truth) // 2
            fit_sel = np.zeros(len(truth), dtype=bool)
            fit_sel[rng.permutation(len(truth))[:half]] = True
            refit = {"n_fit": int(fit_sel.sum()), "n_test": int((~fit_sel).sum())}
            try:
                coeffs2, seconds = call_with_timeout(
                    mech.fit_coeffs, FIT_TIMEOUT,
                    stats[fit_sel], truth[fit_sel], list(mech.COEFFS))
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
                pred2, detail = predict_scored(mech, stats[~fit_sel], coeffs2, seed)
                refit.update(compute_metrics(pred2, truth[~fit_sel]))
                refit.update(detail)
            except Exception as exc:
                refit.update({"r2": None, "rmse": None, "mre": None, "error": str(exc)})
            report["refit"] = refit
            print(f"[eval] refit ({refit['n_fit']} fit / {refit['n_test']} test sims): "
                  f"r2={refit['r2']} rmse={refit['rmse']} mre={refit['mre']}",
                  flush=True)

            report["pa_scores"] = pa_scores(in_suite, refit)
            report["reference_criteria"] = reference_verdicts(in_suite)

    if not report["reference_criteria"]:
        report["reference_criteria"] = reference_verdicts(None)
    report["predictive_accuracy"] = round(
        sum(report["pa_scores"][p] for p in PA_IDS) / len(PA_IDS), 4)
    report["predictive_accuracy_total"] = len(PA_IDS)
    report["reference_score"] = sum(1 for v in report["reference_criteria"].values() if v)
    report["reference_total"] = len(REFERENCE)

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 64)
    print(f"[eval] predictive accuracy: {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    print(f"[eval] vs the reference study's published levels (context, never scored): "
          f"{report['reference_score']}/{report['reference_total']}")
    if report.get("interface_problems"):
        print(f"[eval] contract violations recorded: {report['interface_problems']}")
    if report.get("error"):
        print(f"[eval] evaluation incomplete: {report['error']}")
    print(f"[eval] results: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
