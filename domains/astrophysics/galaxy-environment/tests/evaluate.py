import ast
import importlib.util
import json
import math
import os
import sys
import threading
import time
import traceback

import fire
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from archive import BasicCamelsBlackBox

PROBLEM = "galaxy_environment"
PAPER = ("Sims et al., CAMELS Environments: The Impact of Local Neighbours on "
         "Galaxy Evolution across the SIMBA, IllustrisTNG, ASTRID, and "
         "Swift-EAGLE Simulations, arXiv:2601.06290")

SUITES = [("SIMBA", "SIMBA"), ("IllustrisTNG", "IllustrisTNG"),
          ("ASTRID", "Astrid"), ("Swift-EAGLE", "Swift-EAGLE")]
GEN = "L25n256"
N_SIMS = 27
SNAPSHOT = 90
MSTAR_MIN_MSUN = 1e8
N_NEIGH = 10
BIN_DEX = 0.25
LOG_M_MIN = 10.0
LOG_M_MAX = 14.0
PCT_UNDER = 25
PCT_OVER = 75
CI_PERCENT = 90
P_SIG = 0.05
H_FALLBACK = 0.6711
FRACTIONS = ["f_b", "f_cgm"]
MODELS = [name for name, _ in SUITES]

MAX_COEFFS = 16
PREDICT_TIMEOUT = 7200.0
ALLOWED_IMPORTS = ("numpy", "scipy")
PHYSICAL_LITERALS = (
    4.30091e-06,
    6.674e-11, 6.674e-08,
    1.989e33, 1.989e30,
    3.0857e24, 3.0857e22, 3.0857e21,
    2.775e11,
    1.6726e-24, 1.6726e-27,
    1.3807e-16, 1.3807e-23,
    2.998e10, 2.998e08,
    0.59, 0.61,
    0.76, 0.24,
    0.251,
)
TASK_LITERALS = (0.25, 0.5, 0.75, 25.0, 75.0, 100.0)
ALLOWED_FLOAT_LITERALS = ((0.0, 1.0, 2.0, 0.3, 0.049, 0.8, 0.6711)
                          + PHYSICAL_LITERALS + TASK_LITERALS)
LITERAL_RTOL = 1e-4
MAX_INT_LITERAL = 16
MASS_AXIS = (10.0, 14.0)

GROUP_FIELDS = ["GroupMass", "GroupMassType", "GroupFirstSub"]
SUB_FIELDS = ["SubhaloMassType", "SubhaloPos", "SubhaloGrNr",
              "SubhaloMassInHalfRadType"]

PA_IDS = ("PA1", "PA2", "PA3", "PA4")
PA_NAMES = {"PA1": "direction", "PA2": "size", "PA3": "mass_dependence",
            "PA4": "disagreement"}
HALF_WIDTH_FLOOR = 1e-6


def progress(msg):
    print(msg, flush=True)


def fmt_seconds(seconds):
    minutes, sec = divmod(int(max(seconds, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{sec:02d}s"


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


def fitted_literals(tree):
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "COEFFS" for t in node.targets):
            continue
        for n in ast.walk(node):
            if not isinstance(n, ast.Constant):
                continue
            v = n.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if isinstance(v, int) and 0 <= v <= MAX_INT_LITERAL:
                continue
            if float(v).is_integer():
                continue
            if v != 0 and float(math.log10(abs(v))).is_integer():
                continue
            if MASS_AXIS[0] <= v <= MASS_AXIS[1]:
                continue
            if any(abs(v - a) <= LITERAL_RTOL * abs(a) + 1e-12
                   for a in ALLOWED_FLOAT_LITERALS):
                continue
            out.append(float(v))
    return sorted(set(out))


def load_mechanism(path):
    with open(path) as f:
        source = f.read()
    scan = ImportScan()
    scan.visit(ast.parse(source))
    problems = [f"imports {m}, only {' and '.join(ALLOWED_IMPORTS)} are allowed"
                for m in dict.fromkeys(scan.modules) if m not in ALLOWED_IMPORTS]
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("agent_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blocking = []
    for name in ("COEFFS", "predict_offset"):
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
    if hasattr(module, "COEFFS"):
        try:
            n = len(list(module.COEFFS))
        except TypeError:
            blocking.append("COEFFS is not a sequence")
        else:
            if n > MAX_COEFFS:
                problems.append(f"COEFFS holds {n} constants, the limit is {MAX_COEFFS}")
    extra = fitted_literals(ast.parse(source))
    if extra:
        problems.append(
            f"{len(extra)} fitted constants are written into the body of mechanism.py "
            f"instead of COEFFS, so the constant budget the instruction sets is not "
            f"the number of constants the mechanism uses: {extra}")
    return module, blocking, problems


def fetch_hdf5(bb, relpath):
    for attempt in (0, 1):
        path = bb.fetch(relpath)
        try:
            with h5py.File(path, "r"):
                return path
        except OSError:
            os.remove(path)
            if attempt:
                raise
    return path


def read_catalog(path):
    with h5py.File(path, "r") as f:
        z = float(f["Header"].attrs["Redshift"])
        if abs(z) > 0.02:
            raise RuntimeError(f"{path}: snapshot {SNAPSHOT} is at z={z}, not z=0")
        box = float(np.atleast_1d(f["Header"].attrs["BoxSize"]).flat[0])
        h = float(f["Header"].attrs.get("HubbleParam", H_FALLBACK))
        group = {name: f["Group"][name][:] for name in GROUP_FIELDS}
        sub = {name: f["Subhalo"][name][:] for name in SUB_FIELDS}
    return group, sub, box, h


def delta_n(central_pos, galaxy_pos, box):
    tree = cKDTree(np.mod(galaxy_pos, box), boxsize=box)
    dist, _ = tree.query(np.mod(central_pos, box), k=N_NEIGH + 1)
    self_hit = dist[:, 0] < 1e-8
    radius = np.where(self_hit, dist[:, N_NEIGH], dist[:, N_NEIGH - 1])
    rho_n = N_NEIGH / (4.0 / 3.0 * np.pi * radius ** 3)
    return rho_n / (len(galaxy_pos) / box ** 3)


def halos_of_sim(group, sub, box, h):
    m_halo = group["GroupMass"] * 1e10 / h
    gas = group["GroupMassType"][:, 0]
    stars = group["GroupMassType"][:, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_b = np.where(group["GroupMass"] > 0, (gas + stars) / group["GroupMass"], np.nan)
    inner_gas = np.zeros(len(m_halo))
    grnr = sub["SubhaloGrNr"].astype(np.int64)
    valid = (grnr >= 0) & (grnr < len(m_halo))
    np.add.at(inner_gas, grnr[valid], sub["SubhaloMassInHalfRadType"][valid, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        f_cgm = np.where(group["GroupMass"] > 0,
                         np.maximum(gas - inner_gas, 0.0) / group["GroupMass"], np.nan)
    mstar_msun = sub["SubhaloMassType"][:, 4] * 1e10 / h
    galaxy_pos = sub["SubhaloPos"][mstar_msun > MSTAR_MIN_MSUN]
    first = group["GroupFirstSub"].astype(np.int64)
    has_central = (first >= 0) & (first < len(mstar_msun))
    delta = np.full(len(m_halo), np.nan)
    delta[has_central] = delta_n(sub["SubhaloPos"][first[has_central]], galaxy_pos, box)
    keep = has_central & np.isfinite(f_b) & np.isfinite(f_cgm) & (m_halo > 0)
    return {"m_halo": m_halo[keep], "f_b": f_b[keep], "f_cgm": f_cgm[keep],
            "delta": delta[keep]}


def gather_suite(bb, path_name):
    parts = {"m_halo": [], "f_b": [], "f_cgm": [], "delta": []}
    for i in range(N_SIMS):
        path = fetch_hdf5(bb, f"FOF_Subfind/{path_name}/{GEN}/CV/CV_{i}/"
                              f"groups_{SNAPSHOT:03d}.hdf5")
        sim = halos_of_sim(*read_catalog(path))
        for key in parts:
            parts[key].append(sim[key])
        if (i + 1) % 9 == 0:
            progress(f"    {path_name} {i + 1}/{N_SIMS} catalogs")
    return {key: np.concatenate(arrs) for key, arrs in parts.items()}


def bin_rows(data, field, rng, n_boot):
    edges = np.arange(LOG_M_MIN, LOG_M_MAX + 1e-9, BIN_DEX)
    lo_pct, hi_pct = (100 - CI_PERCENT) / 2, 100 - (100 - CI_PERCENT) / 2
    logm = np.log10(data["m_halo"])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (logm >= lo) & (logm < hi) & np.isfinite(data["delta"])
        row = {"logm_lo": float(lo), "logm_hi": float(hi), "n": int(sel.sum())}
        val, delta = data[field][sel], data["delta"][sel]
        if row["n"] == 0:
            rows.append({**row, "valid": False})
            continue
        p_under, p_over = np.percentile(delta, [PCT_UNDER, PCT_OVER])
        under, over = val[delta <= p_under], val[delta >= p_over]
        median_all = float(np.median(val))
        if len(under) == 0 or len(over) == 0 or median_all <= 0:
            rows.append({**row, "valid": False})
            continue
        boot = ((np.median(rng.choice(over, size=(n_boot, len(over))), axis=1)
                 - np.median(rng.choice(under, size=(n_boot, len(under))), axis=1))
                / np.median(rng.choice(val, size=(n_boot, len(val))), axis=1))
        rows.append({**row, "valid": True,
                     "median_all": median_all,
                     "median_under": float(np.median(under)),
                     "median_over": float(np.median(over)),
                     "offset": float((np.median(over) - np.median(under)) / median_all),
                     "offset_ci": [float(np.percentile(boot, lo_pct)),
                                   float(np.percentile(boot, hi_pct))],
                     "p_mwu": float(mannwhitneyu(under, over,
                                                 alternative="two-sided").pvalue)})
    return rows


def measure(bb, n_boot, seed):
    rng = np.random.default_rng(seed)
    progress(f"[eval] {len(SUITES)} suites x {N_SIMS} catalogs; an estimate of the "
             f"total is printed after the first suite")
    stats = {}
    t0 = time.time()
    for i, (name, path_name) in enumerate(SUITES):
        progress(f"[eval] measuring {name}")
        data = gather_suite(bb, path_name)
        stats[name] = {field: bin_rows(data, field, rng, n_boot) for field in FRACTIONS}
        elapsed = time.time() - t0
        remaining = elapsed * (len(SUITES) - i - 1) / (i + 1)
        progress(f"  {name}: {len(data['m_halo'])} haloes | elapsed "
                 f"{fmt_seconds(elapsed)} | estimated remaining {fmt_seconds(remaining)}")
    return stats


def panel_truth(rows):
    valid = [r for r in rows if r.get("valid")]
    return {
        "logm": np.array([0.5 * (r["logm_lo"] + r["logm_hi"]) for r in valid]),
        "offset": np.array([r["offset"] for r in valid]),
        "ci_lo": np.array([r["offset_ci"][0] for r in valid]),
        "ci_hi": np.array([r["offset_ci"][1] for r in valid]),
        "significant": np.array([r["p_mwu"] < P_SIG for r in valid], dtype=bool),
        "n": [r["n"] for r in valid],
    }


def predict_panel(mech, model, fraction, logm):
    out, seconds = call_with_timeout(mech.predict_offset, PREDICT_TIMEOUT,
                                     model, logm, list(mech.COEFFS))
    pred = np.asarray(out[fraction], dtype=np.float64).reshape(-1)
    if pred.shape != logm.shape or not np.all(np.isfinite(pred)):
        raise ValueError(f"predict_offset returned {pred.shape} for {fraction}, not one "
                         f"finite value per mass bin ({logm.shape})")
    return pred, seconds


def per_bin_check(mech, model, logm, coeffs, pred, fraction, n_probe=6, tolerance=1e-9):
    probe = np.unique(np.linspace(0, len(logm) - 1, min(n_probe, len(logm))).astype(int))
    worst = 0.0
    for i in probe:
        alone = np.asarray(mech.predict_offset(model, logm[i:i + 1], coeffs)[fraction],
                           dtype=np.float64).reshape(-1)
        worst = max(worst, abs(float(alone[0]) - float(pred[i])))
    return {"n_probed": int(len(probe)), "max_difference": float(worst),
            "passed": bool(worst <= tolerance)}


def determinism_check(mech, model, logm, coeffs, pred, fraction):
    again = np.asarray(mech.predict_offset(model, logm, coeffs)[fraction],
                       dtype=np.float64).reshape(-1)
    worst = float(np.max(np.abs(again - pred))) if again.shape == pred.shape else float("inf")
    return {"max_difference": worst, "passed": bool(worst == 0.0)}


def clip01(value):
    return float(min(1.0, max(0.0, float(value))))


def direction_score(panels):
    hits = total = 0
    for p in panels:
        sig = p["truth"]["significant"]
        pred, meas = p["pred"][sig], p["truth"]["offset"][sig]
        total += int(sig.sum())
        hits += int(np.sum((np.sign(pred) == np.sign(meas)) & (pred != 0)))
    return {"score": (hits / total) if total else 0.0,
            "n_significant_bins": total, "n_sign_matches": hits}


def size_score(panels):
    error = null = 0.0
    for p in panels:
        t = p["truth"]
        half = np.maximum(0.5 * (t["ci_hi"] - t["ci_lo"]), HALF_WIDTH_FLOOR)
        error += float(np.sum(np.abs(p["pred"] - t["offset"]) / half))
        null += float(np.sum(np.abs(t["offset"]) / half))
    return {"score": clip01(1.0 - error / null) if null > 0 else 0.0,
            "error_sum": error, "null_error_sum": null}


def mass_dependence_score(panels):
    per = {}
    for p in panels:
        pred, meas = p["pred"], p["truth"]["offset"]
        if len(pred) < 2 or np.std(pred) == 0 or np.std(meas) == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(pred, meas)[0, 1])
        per[f"{p['model']} {p['fraction']}"] = clip01(r if np.isfinite(r) else 0.0)
    return {"score": float(np.mean(list(per.values()))) if per else 0.0,
            "per_panel": per}


def disagreement_score(panels):
    by_key = {(p["model"], p["fraction"]): p for p in panels}
    error = null = 0.0
    n_pairs = 0
    for fraction in FRACTIONS:
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                pa, pb = by_key[(a, fraction)], by_key[(b, fraction)]
                ka = {round(float(v), 3): j for j, v in enumerate(pa["truth"]["logm"])}
                kb = {round(float(v), 3): j for j, v in enumerate(pb["truth"]["logm"])}
                shared = sorted(set(ka) & set(kb))
                if not shared:
                    continue
                ia = np.array([ka[k] for k in shared])
                ib = np.array([kb[k] for k in shared])
                diff_meas = pa["truth"]["offset"][ia] - pb["truth"]["offset"][ib]
                diff_pred = pa["pred"][ia] - pb["pred"][ib]
                error += float(np.sum(np.abs(diff_pred - diff_meas)))
                null += float(np.sum(np.abs(diff_meas)))
                n_pairs += 1
    return {"score": clip01(1.0 - error / null) if null > 0 else 0.0,
            "error_sum": error, "null_error_sum": null, "n_pairs": n_pairs}


def all_scores(panels):
    return {"direction": direction_score(panels),
            "size": size_score(panels),
            "mass_dependence": mass_dependence_score(panels),
            "disagreement": disagreement_score(panels)}


def main(mechanism, output, data_dir=None, n_boot=1000, seed=0, predict_timeout=None):
    global PREDICT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s", flush=True)
    t_start = time.time()
    mechanism_path = str(mechanism)
    report = {
        "problem": PROBLEM,
        "mechanism_path": mechanism_path,
        "protocol": {
            "reference_study": PAPER,
            "settings": "the eight panels of Fig. 4 and Fig. 5: f_B (Eq. 4) and "
                        "f_CGM (Eq. 5) in the four CV sets at z=0",
            "data": f"the {N_SIMS} CV simulations of each model, pooled (Sec. 2.1)",
            "environment": f"delta_{N_NEIGH} of Eq. 1 and Eq. 2, galaxies with "
                           f"M_star > {MSTAR_MIN_MSUN:.0e} Msun (Sec. 2.2)",
            "binning": f"{BIN_DEX} dex mass bins from log10 M = {LOG_M_MIN:g} to "
                       f"{LOG_M_MAX:g}, overdense above the {PCT_OVER}th and "
                       f"underdense below the {PCT_UNDER}th percentile of "
                       f"delta_{N_NEIGH} within the bin (Sec. 2.4)",
            "offset": "the offset the two figures compare, (median of the overdense "
                      "haloes minus median of the underdense haloes) over the "
                      "all-halo median of the bin",
            "instruments": f"the {CI_PERCENT}% confidence interval of the bootstrapped "
                           f"offset and the Mann-Whitney U test at p < {P_SIG} "
                           f"(Sec. 2.4)",
            "scores": {
                "PA1": "direction: the share of the significant bins in which the "
                       "predicted offset has the measured sign; a predicted zero has "
                       "no sign",
                "PA2": "size: 1 minus the summed error of the predicted offsets in "
                       "units of each bin's confidence half-width, over the same sum "
                       "for a prediction of zero offset, floored at 0",
                "PA3": "mass dependence: the Pearson correlation across the mass "
                       "bins between predicted and measured offsets, floored at 0, "
                       "averaged over the eight model and fraction pairs",
                "PA4": "disagreement: over the six model pairs and both fractions, "
                       "1 minus the summed absolute error of the predicted "
                       "differences between two models' offsets, over the same sum "
                       "for a prediction of no difference, floored at 0",
            },
            "submission_contract": f"COEFFS at most {MAX_COEFFS} constants, only "
                                   f"{' and '.join(ALLOWED_IMPORTS)}, predict_offset "
                                   f"returns within {PREDICT_TIMEOUT:.0f}s; these are "
                                   f"the limits the instruction gave the agent, not "
                                   f"settings of the paper",
            "n_boot": n_boot, "seed": seed,
        },
        "panels": [], "scores": {},
        "pa_scores": {pid: 0.0 for pid in PA_IDS}, "predictive_accuracy": 0.0,
        "problems": [], "seconds": None,
    }

    mech = None
    if not os.path.exists(mechanism_path):
        report["error"] = "mechanism.py not found"
        progress(f"[eval] ERROR: {mechanism_path} not found")
    else:
        try:
            mech, blocking, problems = load_mechanism(mechanism_path)
        except Exception:
            mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
        report["interface_problems"] = problems
        report["blocking_problems"] = blocking
        for problem in problems:
            progress(f"[eval] contract violation, recorded and not scored here: {problem}")
        if mech is None or blocking:
            mech = None
            report["error"] = "mechanism.py cannot be evaluated: " + "; ".join(blocking)
            progress(f"[eval] ERROR: {report['error']}")

    stats = measure(BasicCamelsBlackBox(data_dir=data_dir), n_boot, seed)

    panels = []
    for model in MODELS:
        for fraction in FRACTIONS:
            truth = panel_truth(stats[model][fraction])
            if len(truth["logm"]) == 0:
                raise RuntimeError(f"{model} {fraction}: no usable mass bin")
            entry = {"model": model, "fraction": fraction,
                     "n_bins": int(len(truth["logm"])),
                     "log10_M_halo": [round(float(v), 3) for v in truth["logm"]],
                     "n_haloes": truth["n"],
                     "measured_offset": [round(float(v), 5) for v in truth["offset"]],
                     "measured_offset_ci": [[round(float(a), 5), round(float(b), 5)]
                                            for a, b in zip(truth["ci_lo"], truth["ci_hi"])],
                     "significant": [bool(v) for v in truth["significant"]]}
            pred = None
            if mech is not None:
                try:
                    pred, seconds = predict_panel(mech, model, fraction, truth["logm"])
                    entry["predicted_offset"] = [round(float(v), 5) for v in pred]
                    entry["predict_seconds"] = round(seconds, 3)
                    entry["per_bin"] = per_bin_check(mech, model, truth["logm"],
                                                     list(mech.COEFFS), pred, fraction)
                    entry["deterministic"] = determinism_check(
                        mech, model, truth["logm"], list(mech.COEFFS), pred, fraction)
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    report["problems"].append(f"{model} {fraction}: {entry['error']}")
                    pred = None
            report["panels"].append(entry)
            panels.append({"model": model, "fraction": fraction, "truth": truth,
                           "pred": pred})
            head = f"  {model:14s} {fraction:6s} bins={entry['n_bins']:2d}"
            if pred is None:
                progress(f"{head} | {entry.get('error') or report.get('error')}")
            else:
                sig = truth["significant"]
                agree = int(np.sum((np.sign(pred[sig]) == np.sign(truth["offset"][sig]))
                                   & (pred[sig] != 0)))
                progress(f"{head} | signs {agree}/{int(sig.sum())} | rmse "
                         f"{float(np.sqrt(np.mean((pred - truth['offset']) ** 2))):.4f}")

    if mech is not None and all(p["pred"] is not None for p in panels):
        report["scores"] = all_scores(panels)
        report["pa_scores"] = {pid: round(report["scores"][PA_NAMES[pid]]["score"], 4)
                               for pid in PA_IDS}
    report["predictive_accuracy"] = round(float(np.mean(list(
        report["pa_scores"].values()))), 4)
    report["predictive_accuracy_total"] = len(PA_IDS)
    report["seconds"] = round(time.time() - t_start, 1)

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("=" * 64)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report["problems"]:
        print(f"[eval] problem: {problem}")
    for pid in PA_IDS:
        print(f"  {pid} {PA_NAMES[pid]:<16} {report['pa_scores'][pid]:.4f}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']}")
    print(f"[eval] results: {out_path}  ({fmt_seconds(time.time() - t_start)})")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
