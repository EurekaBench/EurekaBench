import ast
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import fire
import h5py
import numpy as np
from archive import BasicCamelsBlackBox
import reference_formula

GEN = "L25n256"
GALAXY_MIN_STARS = 20
GALAXY_NEEDS_STELLAR_RADIUS = True
PER_SIM = 20
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0

SNAPSHOT_FOR_Z = {0.0: 90, 1.0: 60, 2.0: 44, 3.0: 32}

SETTINGS = [
    {"suite": "IllustrisTNG", "z": 0.0, "refit": False, "pa": "PA1"},
    {"suite": "Astrid", "z": 0.0, "refit": True, "pa": "PA2"},
    {"suite": "SIMBA", "z": 0.0, "refit": True, "pa": "PA3"},
    {"suite": "Swift-EAGLE", "z": 0.0, "refit": True, "pa": "PA4"},
    {"suite": "IllustrisTNG", "z": 1.0, "refit": True, "pa": "PA5"},
    {"suite": "IllustrisTNG", "z": 2.0, "refit": True, "pa": "PA6"},
    {"suite": "IllustrisTNG", "z": 3.0, "refit": True, "pa": "PA7"},
    {"suite": "Astrid", "z": 1.0, "refit": True, "pa": "PA8"},
    {"suite": "Astrid", "z": 2.0, "refit": True, "pa": "PA9"},
    {"suite": "Astrid", "z": 3.0, "refit": True, "pa": "PA10"},
]

REFERENCE_SCORES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reference_scores.json")


def load_human_reference():
    if not os.path.exists(REFERENCE_SCORES):
        print(f"[eval] {REFERENCE_SCORES} not found; the human reference is not reported",
              flush=True)
        return {}
    with open(REFERENCE_SCORES) as f:
        scores = json.load(f)
    human = {}
    for entry in scores["settings"]:
        protocol = entry.get("paper_protocol") or {}
        if protocol.get("r2") is not None:
            human[f"{entry['suite']} z={entry['z']:g}"] = {
                "r2": float(protocol["r2"]), "accuracy": float(protocol["accuracy"])}
    return human


def normalized_score(r2):
    if r2 is None or not np.isfinite(r2):
        return 0.0
    return float(min(1.0, max(0.0, r2)))

LARGER_SCATTER = ("Fig. 2 shows systematically larger scatter than IllustrisTNG "
                  "and Astrid; the paper quotes no number for this panel")
HIGH_Z_RANGE = {"r2": [0.74, 0.76], "accuracy": [0.056, 0.060],
                "note": "quoted in Section III.2 as the range over the z=1, 2, 3 "
                        "panels of IllustrisTNG and Astrid together"}
PAPER_REPORTED = {
    "IllustrisTNG z=0": {"r2": 0.77, "accuracy": 0.056},
    "Astrid z=0": {"r2": 0.73, "accuracy": 0.060},
    "SIMBA z=0": LARGER_SCATTER,
    "Swift-EAGLE z=0": LARGER_SCATTER,
    "IllustrisTNG z=1": HIGH_Z_RANGE, "IllustrisTNG z=2": HIGH_Z_RANGE,
    "IllustrisTNG z=3": HIGH_Z_RANGE, "Astrid z=1": HIGH_Z_RANGE,
    "Astrid z=2": HIGH_Z_RANGE, "Astrid z=3": HIGH_Z_RANGE,
}

OMEGA_M_COLUMNS = ["Omega_m", "Omega0", "Om"]
CALIBRATION_EDGES = np.linspace(0.1, 0.5, 9)


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
    for name in ("PROPERTIES", "COEFFS", "predict_omega_m", "fit_coeffs"):
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
    if hasattr(module, "PROPERTIES") and not all(isinstance(p, str) for p in module.PROPERTIES):
        blocking.append("PROPERTIES entries are not all strings")
    if hasattr(module, "COEFFS") and len(list(module.COEFFS)) > MAX_COEFFS:
        problems.append(f"COEFFS has {len(list(module.COEFFS))} entries, limit {MAX_COEFFS}")
    with open(mechanism_path) as f:
        scan = ImportScan()
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy that the formula can reach: {bad}")
    return module, blocking, problems


def load_omega_m_table(bb, suite):
    path = bb.fetch(f"Parameters/{suite}/CosmoAstroSeed_{suite}_{GEN}_LH.txt")
    with open(path) as f:
        header = f.readline().strip().lstrip("#").split()
        column = next(i for i, name in enumerate(header) if name in OMEGA_M_COLUMNS)
        return {parts[0]: float(parts[column])
                for parts in (line.split() for line in f) if parts}


def select_simulations(table, n_sims):
    names = sorted(table, key=lambda s: (table[s], int(s.split("_")[1])))
    if n_sims >= len(names):
        return names
    picks = ((np.arange(n_sims) + 0.5) / n_sims * len(names)).astype(int)
    return [names[i] for i in sorted(set(picks.tolist()))]


def read_catalog(bb, relpath, properties, per_sim, rng, keep_catalogs):
    cached = os.path.exists(os.path.join(bb.data_dir, relpath))
    path = bb.fetch(relpath)
    try:
        with h5py.File(path, "r") as f:
            redshift = float(f["Header"].attrs["Redshift"])
            sub = f["Subhalo"]
            missing = [p for p in list(properties) + ["SubhaloLenType"] if p not in sub]
            if missing:
                raise KeyError(f"fields {missing} are not in this catalogue")
            keep = sub["SubhaloLenType"][:, 4] > GALAXY_MIN_STARS
            if GALAXY_NEEDS_STELLAR_RADIUS:
                keep &= sub["SubhaloHalfmassRadType"][:, 4] > 0
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                return None, None, redshift
            chosen = rng.permutation(idx)
            eval_idx = np.sort(chosen[:per_sim])
            fit_idx = np.sort(chosen[per_sim:2 * per_sim])
            fields = {p: sub[p][:] for p in properties}
        take = lambda sel: {p: v[sel] for p, v in fields.items()}
        return take(eval_idx), (take(fit_idx) if len(fit_idx) else None), redshift
    finally:
        if not keep_catalogs and not cached and os.path.exists(path):
            os.remove(path)


def prefetch(bb, suite, snapshot, sims, workers):
    todo = [f"FOF_Subfind/{suite}/{GEN}/LH/{sim}/groups_{snapshot:03d}.hdf5" for sim in sims]
    todo = [r for r in todo if not os.path.exists(os.path.join(bb.data_dir, r))]
    if not todo:
        return

    def pull(relpath):
        try:
            bb.fetch(relpath)
        except Exception as exc:
            print(f"  [prefetch failed] {relpath}: {exc}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, finished in enumerate(pool.map(pull, todo), 1):
            if done % 50 == 0 or done == len(todo):
                progress("prefetch", done, len(todo), t0,
                         f"{suite} snap {snapshot}, {workers} workers")


def gather(bb, suite, target_z, sims, properties, table, per_sim, seed, keep_catalogs):
    snapshot = SNAPSHOT_FOR_Z[target_z]
    rng = np.random.default_rng(seed)
    eval_parts, fit_parts = [], []
    eval_omega, fit_omega = [], []
    redshifts, skipped = [], 0
    t0 = time.time()
    label = f"{suite} z={target_z:g}"
    for i, sim in enumerate(sims):
        relpath = f"FOF_Subfind/{suite}/{GEN}/LH/{sim}/groups_{snapshot:03d}.hdf5"
        try:
            ev, ft, redshift = read_catalog(bb, relpath, properties, per_sim, rng,
                                            keep_catalogs)
        except Exception as exc:
            print(f"  [skip] {suite}/{sim} snap {snapshot}: {exc}", flush=True)
            skipped += 1
            continue
        redshifts.append(redshift)
        if ev is None:
            skipped += 1
            continue
        eval_parts.append(ev)
        eval_omega.append(np.full(len(next(iter(ev.values()))), table[sim]))
        if ft is not None:
            fit_parts.append(ft)
            fit_omega.append(np.full(len(next(iter(ft.values()))), table[sim]))
        if (i + 1) % 25 == 0 or (i + 1) == len(sims):
            progress(label, i + 1, len(sims), t0,
                     f"{sum(len(o) for o in eval_omega)} eval galaxies")
    if not eval_parts:
        raise RuntimeError(f"no usable simulations for {suite} z={target_z}")
    stack = lambda parts: {p: np.concatenate([q[p] for q in parts]) for p in properties}
    return {
        "eval_props": stack(eval_parts), "eval_omega": np.concatenate(eval_omega),
        "fit_props": stack(fit_parts) if fit_parts else None,
        "fit_omega": np.concatenate(fit_omega) if fit_omega else None,
        "snapshot": snapshot, "actual_z": float(np.median(redshifts)),
        "n_sims_used": len(eval_parts), "n_sims_skipped": skipped,
    }


def compute_metrics(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
        return {"r2": None, "accuracy": None, "mae": None,
                "error": "predictions are not a finite array of one value per galaxy"}
    return {
        "r2": float(1 - np.mean((pred - truth) ** 2) / np.var(truth)),
        "accuracy": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mae": float(np.mean(np.abs(pred - truth))),
    }


def calibration(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    curve = []
    for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        sel = (truth >= lo) & (truth < hi)
        if sel.sum() < 10:
            continue
        curve.append({"true_omega_m": round(float(0.5 * (lo + hi)), 3),
                      "n": int(sel.sum()),
                      "median_predicted": round(float(np.median(pred[sel])), 4),
                      "p16": round(float(np.percentile(pred[sel], 16)), 4),
                      "p84": round(float(np.percentile(pred[sel], 84)), 4)})
    return curve


def per_galaxy_check(mech, props, coeffs, pred, n_probe=32, seed=0, tolerance=1e-9):
    rng = np.random.default_rng(seed)
    n = len(pred)
    probe = rng.choice(n, size=min(n_probe, n), replace=False)
    worst = 0.0
    for i in probe:
        alone = {name: np.asarray(value)[i:i + 1] for name, value in props.items()}
        one = np.asarray(mech.predict_omega_m(alone, coeffs), dtype=np.float64).reshape(-1)
        worst = max(worst, abs(float(one[0]) - float(pred[i])))
    return {"n_probed": int(len(probe)), "max_difference": worst,
            "passed": bool(worst <= tolerance)}


def evaluate_agent(mech, setting, data):
    result = {"refit": setting["refit"]}
    coeffs = list(mech.COEFFS)
    if setting["refit"]:
        if data["fit_props"] is None:
            raise RuntimeError("no galaxies available to recalibrate the constants")
        coeffs, seconds = call_with_timeout(mech.fit_coeffs, FIT_TIMEOUT, data["fit_props"],
                                            data["fit_omega"], list(mech.COEFFS))
        coeffs = [float(c) for c in coeffs]
        if len(coeffs) > MAX_COEFFS:
            result["contract_violation"] = (f"fit_coeffs returned {len(coeffs)} "
                                            f"constants, limit {MAX_COEFFS}")
            print(f"[eval] contract violation, recorded and scored anyway: "
                  f"{result['contract_violation']}", flush=True)
        result["fit_seconds"] = round(seconds, 3)
        result["n_fit_galaxies"] = int(len(data["fit_omega"]))
    result["coeffs"] = [round(c, 8) for c in coeffs]
    pred, seconds = call_with_timeout(mech.predict_omega_m, PREDICT_TIMEOUT,
                                      data["eval_props"], coeffs)
    result["predict_seconds"] = round(seconds, 3)
    result["per_galaxy"] = per_galaxy_check(mech, data["eval_props"], coeffs, pred)
    if not result["per_galaxy"]["passed"]:
        print(f"  [note] predict_omega_m does not predict one galaxy at a time: a "
              f"galaxy's value changes by up to "
              f"{result['per_galaxy']['max_difference']:.3g} when it is evaluated on "
              f"its own; recorded for the judge, the metrics below are still computed",
              flush=True)
    result.update(compute_metrics(pred, data["eval_omega"]))
    result["calibration"] = calibration(pred, data["eval_omega"])
    return result


def evaluate_reference(setting, data):
    pred = reference_formula.predict_omega_m(data["eval_props"], setting["suite"],
                                             data["actual_z"])
    k, c0, a0 = reference_formula.COEFFS[setting["suite"]]
    result = {"published_coeffs_k_c0_a0": [k, c0, a0], "refit": False}
    result.update(compute_metrics(pred, data["eval_omega"]))
    result["calibration"] = calibration(pred, data["eval_omega"])
    return result


def main(mechanism, output, data_dir=None, n_sims=1000, per_sim=PER_SIM, seed=0,
         keep_catalogs=True, workers=8, predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s, fit {FIT_TIMEOUT:.0f}s", flush=True)
    mechanism_path = str(mechanism)
    all_pa = [s["pa"] for s in SETTINGS]
    human = load_human_reference()
    report = {
        "mechanism_path": mechanism_path,
        "protocol": {
            "reference_study": reference_formula.PAPER,
            "settings": "the four CAMELS LH suites at z=0 and IllustrisTNG and Astrid "
                        "at z=1, 2, 3, as in Fig. 2 and Fig. 3 of the reference study",
            "galaxy_definition": f"subhalo with SubhaloLenType[4] > {GALAXY_MIN_STARS} and "
                                 f"SubhaloHalfmassRadType[4] > 0",
            "galaxies_per_simulation": per_sim,
            "galaxies_excluded": "subhalos whose stellar half mass radius is zero, for "
                                 "which the compactness ratio of the reference equation "
                                 "is undefined; this is one row in about three thousand "
                                 "at z=3 and none at z=0",
            "n_simulations_requested": n_sims,
            "simulation_selection": "every simulation of the LH set, as in the "
                                    "reference study",
            "seed": seed,
            "constants": "submitted COEFFS on IllustrisTNG z=0; recalibrated by the "
                         "submission's own fit_coeffs on a disjoint galaxy sample of "
                         "the same simulations for every other setting, mirroring the "
                         "reference study's per-suite recalibration",
            "metrics": "every criterion scores the coefficient of determination (Eq. 3) "
                       "between the predicted and the true Omega_m, floored at 0 and "
                       "capped at 1, with no threshold; accuracy, the root-mean-square "
                       "deviation (Eq. 4), and mae are reported alongside",
            "reference": "the reference study's Eq. 5 and Eq. 6 with its published "
                         "Table 3 constants, evaluated on exactly the same galaxies; "
                         "reported for comparison and never counted in any score",
            "human_reference": "the reference study's own equation on the same galaxies "
                               "under this protocol, refit where the submission is refit "
                               "(paper_protocol in reference_scores.json), scored the same "
                               "way; reported for comparison and never counted in any "
                               "score",
        },
        "paper_reported": PAPER_REPORTED,
        "settings": [], "pa_scores": {}, "human_reference_pa_scores": {},
    }

    if not os.path.exists(mechanism_path):
        report["error"] = "mechanism.py not found"
        print(f"[eval] ERROR: {mechanism_path} not found", flush=True)
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
            report["error"] = ("mechanism.py cannot be evaluated: "
                               + "; ".join(blocking))
            print(f"[eval] ERROR: {report['error']}", flush=True)

    if report.get("error") is None:
        bb = BasicCamelsBlackBox(data_dir=data_dir)
        properties = list(dict.fromkeys(list(mech.PROPERTIES)
                                        + reference_formula.PROPERTIES))
        for setting in SETTINGS:
            label = f"{setting['suite']} z={setting['z']:g}"
            print(f"[eval] {label} (refit={setting['refit']})", flush=True)
            entry = {"suite": setting["suite"], "z": setting["z"], "pa": setting["pa"],
                     "human_reference": human.get(label)}
            try:
                table = load_omega_m_table(bb, setting["suite"])
                sims = select_simulations(table, n_sims)
                if workers > 1 and keep_catalogs:
                    prefetch(bb, setting["suite"], SNAPSHOT_FOR_Z[setting["z"]], sims,
                             workers)
                data = gather(bb, setting["suite"], setting["z"], sims, properties,
                              table, per_sim, seed, keep_catalogs)
            except Exception as exc:
                print(f"  [error] {label}: {exc}", flush=True)
                entry["error"] = str(exc)
                entry["score"] = 0.0
                report["settings"].append(entry)
                report["pa_scores"][setting["pa"]] = entry["score"]
                continue
            entry.update({
                "snapshot": data["snapshot"], "actual_z": round(data["actual_z"], 4),
                "n_sims_used": data["n_sims_used"], "n_sims_skipped": data["n_sims_skipped"],
                "n_eval_galaxies": int(len(data["eval_omega"])),
                "var_true_omega_m": round(float(np.var(data["eval_omega"])), 6),
            })
            entry["paper_reported"] = PAPER_REPORTED.get(label)
            entry["reference"] = evaluate_reference(setting, data)
            try:
                entry["agent"] = evaluate_agent(mech, setting, data)
            except Exception as exc:
                print(f"  [error] {label}: {exc}", flush=True)
                entry["agent"] = {"error": str(exc), "r2": None, "accuracy": None}
            entry["score"] = round(normalized_score(entry["agent"].get("r2")), 4)
            report["settings"].append(entry)
            report["pa_scores"][setting["pa"]] = entry["score"]
            a, r, h = entry["agent"], entry["reference"], entry["human_reference"]
            if a.get("r2") is not None:
                print(f"  agent    : r2={a['r2']:+.4f} accuracy={a['accuracy']:.4f} "
                      f"mae={a['mae']:.4f}  ({setting['pa']} score {entry['score']:.4f})",
                      flush=True)
            else:
                print(f"  agent    : no finite prediction ({setting['pa']} score 0)",
                      flush=True)
            print(f"  reference: r2={r['r2']:+.4f} accuracy={r['accuracy']:.4f} "
                  f"mae={r['mae']:.4f}  ({entry['n_eval_galaxies']} galaxies, "
                  f"snap {entry['snapshot']}, z={entry['actual_z']})", flush=True)
            if h is not None:
                print(f"  human ref: r2={h['r2']:+.4f} accuracy={h['accuracy']:.4f}  "
                      f"(for comparison, not counted)", flush=True)

    for pid in all_pa:
        report["pa_scores"].setdefault(pid, 0.0)
    report["pa_scores"] = {pid: report["pa_scores"][pid] for pid in all_pa}
    for s in SETTINGS:
        label = f"{s['suite']} z={s['z']:g}"
        if label in human:
            report["human_reference_pa_scores"][s["pa"]] = round(
                normalized_score(human[label]["r2"]), 4)
    report["predictive_accuracy"] = round(
        float(np.mean([report["pa_scores"][pid] for pid in all_pa])), 4)
    report["predictive_accuracy_total"] = len(all_pa)

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 64)
    print(f"[eval] predictive accuracy, the coefficient of determination of each criterion "
          f"floored at 0 and capped at 1: {json.dumps(report['pa_scores'])}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']:.4f}"
          f"  (mean of {report['predictive_accuracy_total']} criteria, each 0-1)")
    if report.get("interface_problems"):
        print(f"[eval] contract violations recorded for the judge: "
              f"{report['interface_problems']}")
    if report.get("error"):
        print(f"[eval] no setting was evaluated: {report['error']}")
    print(f"[eval] results: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
