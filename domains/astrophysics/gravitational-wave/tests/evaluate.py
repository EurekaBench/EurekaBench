import ast
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import h5py
import numpy as np
from archive import BasicCamelsBlackBox

PACKAGE_DIR = Path(__file__).resolve().parent
GEN = "L25n256"
SNAPSHOTS = list(range(44, 91, 2))
MIN_SNAPSHOTS = 20
GRAV_G = 6.6743e-11
LIGHT_C = 2.99792458e8
MSUN_KG = 1.98892e30
MPC_M = 3.0857e22
F_REF = 1.0 / (365.25 * 86400.0)
Q_CENTER = 0.33
Q_SIGMA_DEX = 0.5
PTA_MEASUREMENT = {
    "amplitude": 2.4e-15, "err_plus": 0.7e-15, "err_minus": 0.6e-15,
    "reference_frequency": "1/yr", "source": "NANOGrav 15-year data set",
}
trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def progress(label, done, total, t0, note=""):
    elapsed = time.time() - t0
    frac = done / total if total else 1.0
    filled = int(round(30 * frac))
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    print(f"[{label}] |{'#' * filled}{'.' * (30 - filled)}| {done}/{total} "
          f"elapsed {elapsed:5.0f}s eta {eta:5.0f}s {note}", flush=True)


def mass_ratio_factor():
    q = np.linspace(1e-4, 1.0, 20000)
    p = np.exp(-0.5 * ((np.log10(q) - np.log10(Q_CENTER)) / Q_SIGMA_DEX) ** 2) / q
    p /= trapezoid(p, q)
    return float(trapezoid(p * (q ** 3 / (1 + q)) ** (1.0 / 3.0), q))


def catalog_relpath(suite, set_type, sim, snap):
    return f"FOF_Subfind/{suite}/{GEN}/{set_type}/{sim}/groups_{snap:03d}.hdf5"


def snapshot_term(path):
    with h5py.File(path, "r") as f:
        header = f["Header"].attrs
        redshift = float(header["Redshift"])
        hubble = float(header["HubbleParam"])
        box_mpc = float(header["BoxSize"]) / 1e3 / hubble
        bh = f["Subhalo"]["SubhaloBHMass"][:] if "Subhalo" in f else np.zeros(0)
    masses = bh[bh > 0].astype(np.float64) * 1e10 / hubble
    return redshift, float(np.sum(masses ** (5.0 / 3.0))) / box_mpc ** 3


def measure_raw_amplitude(bb, suite, set_type, sim, qfac):
    redshifts, terms = [], []
    for snap in SNAPSHOTS:
        try:
            z, s = snapshot_term(bb.fetch(catalog_relpath(suite, set_type, sim, snap)))
        except Exception as exc:
            print(f"  [skip snap] {suite}/{set_type}/{sim} snap {snap}: {exc}", flush=True)
            continue
        redshifts.append(z)
        terms.append(s)
    if len(terms) < MIN_SNAPSHOTS:
        raise RuntimeError(f"only {len(terms)} usable snapshots")
    z = np.array(redshifts)
    s = np.array(terms)
    order = np.argsort(z)
    integral = trapezoid(s[order] * (1 + z[order]) ** (-1.0 / 3.0), z[order])
    dens_si = qfac * integral * MSUN_KG ** (5.0 / 3.0) / MPC_M ** 3
    pref = 4 * GRAV_G ** (5.0 / 3.0) / (3 * np.pi ** (1.0 / 3.0) * LIGHT_C ** 2) \
        * F_REF ** (-4.0 / 3.0)
    return float(np.sqrt(pref * dens_si))


def prefetch(bb, jobs, workers):
    if workers <= 1:
        return

    def pull(relpath):
        try:
            bb.fetch(relpath)
        except Exception as exc:
            print(f"  [prefetch failed] {relpath}: {exc}", flush=True)

    done, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(pull, jobs):
            done += 1
            if done % 200 == 0 or done == len(jobs):
                progress("prefetch", done, len(jobs), t0, "catalogs")


MAX_STATISTICS = 8
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
STATS_TIMEOUT = 7200.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0

FIDUCIAL = {"omega_m": 0.3, "omega_b": 0.049, "sigma_8": 0.8,
            "hubble": 0.6711, "ns": 0.9624}
PARAM_ALIASES = {
    "omega_m": ("Omega0", "Omega_m", "Om"),
    "sigma_8": ("sigma8", "sigma_8"),
    "omega_b": ("OmegaBaryon", "Omega_b"),
    "hubble": ("HubbleParam", "h"),
    "ns": ("n_s", "ns"),
}
PARAM_NAMES = list(FIDUCIAL)

SETTINGS = [
    {"suite": "SIMBA", "set": "1P", "refit": False, "pa": "PA1"},
    {"suite": "SIMBA", "set": "CV", "refit": False, "pa": "PA2"},
    {"suite": "SIMBA", "set": "1P", "refit": True, "fit_on": ("SIMBA", "CV"),
     "pa": "PA3"},
]
PA_IDS = ["PA1", "PA2", "PA3"]

PAPER_REPORTED = {
    "reference_study": "arXiv:2602.15938",
    "pta_measurement": PTA_MEASUREMENT,
    "fiducial_deficit": "the fiducial IllustrisTNG, MillenniumTNG and Simba models "
                        "under-predict the NANOGrav 15-yr amplitude by about a "
                        "factor of two (A_GWB 1.1e-15 to 1.5e-15 across 35-500 "
                        "Mpc/h boxes)",
    "feedback_variations": "feedback variants change the amplitude by up to a "
                           "factor of 2 for the Simba 50 Mpc/h variant boxes and "
                           "up to a factor of 10 for CAMELS extreme feedback "
                           "variations",
    "cosmic_variance": "at 25 Mpc/h cosmic variance gives a 2-sigma deviation of "
                       "about 0.1 dex in the predicted amplitude",
}


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
    for name in ("PROPERTIES", "STATISTICS", "COEFFS", "compute_statistics",
                 "predict_agwb", "fit_coeffs"):
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
    if hasattr(module, "PROPERTIES") and not all(isinstance(p, str) for p in module.PROPERTIES):
        blocking.append("PROPERTIES entries are not all strings")
    if hasattr(module, "STATISTICS") and len(list(module.STATISTICS)) > MAX_STATISTICS:
        problems.append(f"STATISTICS has {len(list(module.STATISTICS))} entries, "
                        f"limit {MAX_STATISTICS}")
    if hasattr(module, "COEFFS") and len(list(module.COEFFS)) > MAX_COEFFS:
        problems.append(f"COEFFS has {len(list(module.COEFFS))} entries, limit {MAX_COEFFS}")
    with open(mechanism_path) as f:
        scan = ImportScan()
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy that the law can reach: {bad}")
    return module, blocking, problems


def load_params_table(bb, suite, set_type):
    path = bb.fetch(f"Parameters/{suite}/CosmoAstroSeed_{suite}_{GEN}_{set_type}.txt")
    with open(path) as f:
        columns = f.readline().strip().lstrip("#").split()
        rows = {}
        for line in f:
            parts = line.split()
            if parts:
                rows[parts[0]] = dict(zip(columns[1:], parts[1:]))
    return rows


def cosmo_params(row):
    out = {}
    for name in PARAM_NAMES:
        for alias in PARAM_ALIASES[name]:
            if alias in row:
                out[name] = float(row[alias])
                break
        else:
            out[name] = FIDUCIAL[name]
    return out


def sim_statistics(bb, suite, set_type, sim, properties, mech, cosmo, timing):
    redshifts, vectors = [], []
    for snap in SNAPSHOTS:
        path = bb.fetch(catalog_relpath(suite, set_type, sim, snap))
        with h5py.File(path, "r") as f:
            redshift = float(f["Header"].attrs["Redshift"])
            sub = f["Subhalo"] if "Subhalo" in f else None
            missing = [p for p in properties
                       if sub is None or p not in sub]
            if missing:
                raise KeyError(f"fields {missing} are not in this catalogue")
            props = {p: sub[p][:] for p in properties}
        params = dict(cosmo)
        params["redshift"] = redshift
        vec, seconds = call_with_timeout(mech.compute_statistics, STATS_TIMEOUT,
                                         props, params)
        timing["max_stats_seconds"] = max(timing["max_stats_seconds"], seconds)
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if len(vec) != len(mech.STATISTICS):
            raise RuntimeError(f"compute_statistics returned {len(vec)} values for "
                               f"{len(mech.STATISTICS)} STATISTICS")
        redshifts.append(redshift)
        vectors.append(vec)
    return np.array(redshifts), np.array(vectors)


def gather(bb, suite, set_type, mech, k_amp, qfac, timing, retries=2):
    table = load_params_table(bb, suite, set_type)
    names = list(table)
    properties = list(dict.fromkeys(mech.PROPERTIES))
    collected, failures, redshift_grid = {}, {}, None
    label = f"{suite} {set_type}"
    for attempt in range(retries + 1):
        pending = [s for s in names if s not in collected]
        if not pending:
            break
        if attempt > 0:
            print(f"  [retry {attempt}] {label}: {len(pending)} simulations "
                  f"failed, trying again", flush=True)
        t0 = time.time()
        for i, sim in enumerate(pending):
            cosmo = cosmo_params(table[sim])
            try:
                zs, vecs = sim_statistics(bb, suite, set_type, sim, properties,
                                          mech, cosmo, timing)
                raw = measure_raw_amplitude(bb, suite, set_type, sim, qfac)
                if redshift_grid is None:
                    redshift_grid = zs
                elif not np.allclose(zs, redshift_grid, atol=2e-2):
                    raise RuntimeError("snapshot redshifts deviate from the grid")
            except Exception as exc:
                failures[sim] = f"{type(exc).__name__}: {exc}"
                print(f"  [skip] {suite}/{set_type}/{sim}: {failures[sim]}",
                      flush=True)
                continue
            failures.pop(sim, None)
            collected[sim] = (vecs, raw * k_amp, cosmo)
            if (i + 1) % 20 == 0 or (i + 1) == len(pending):
                progress(label, i + 1, len(pending), t0,
                         f"{len(collected)} usable")
    kept = [s for s in names if s in collected]
    if not kept:
        raise RuntimeError(f"no usable simulations for {suite} {set_type}")
    return {
        "stats": np.array([collected[s][0] for s in kept]),
        "redshifts": redshift_grid,
        "params": {name: np.array([collected[s][2][name] for s in kept])
                   for name in PARAM_NAMES},
        "truth": np.array([collected[s][1] for s in kept]),
        "sims": kept,
        "n_skipped": len(names) - len(kept),
        "skipped": [{"sim": s, "reason": failures[s]} for s in names
                    if s in failures],
    }


def compute_metrics(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)) or np.any(pred <= 0):
        return {"r2_log10": None, "rmse_dex": None,
                "error": "predictions are not a finite positive array of one value "
                         "per simulation"}
    lp, lt = np.log10(pred), np.log10(truth)
    return {
        "r2_log10": float(1 - np.mean((lp - lt) ** 2) / np.var(lt)),
        "rmse_dex": float(np.sqrt(np.mean((lp - lt) ** 2))),
        "mae_dex": float(np.mean(np.abs(lp - lt))),
    }


def per_sim_check(mech, data, coeffs, pred, n_probe=8, seed=0, rtol=1e-9):
    rng = np.random.default_rng(seed)
    n = len(pred)
    probe = rng.choice(n, size=min(n_probe, n), replace=False)
    worst = 0.0
    for i in probe:
        alone_params = {name: values[i:i + 1] for name, values in data["params"].items()}
        one = np.asarray(mech.predict_agwb(data["stats"][i:i + 1], data["redshifts"],
                                           alone_params, coeffs),
                         dtype=np.float64).reshape(-1)
        worst = max(worst, abs(float(one[0]) - float(pred[i])) / max(abs(float(pred[i])), 1e-300))
    return {"n_probed": int(len(probe)), "max_relative_difference": worst,
            "passed": bool(worst <= rtol)}


def truth_context(data):
    truth = data["truth"]
    out = {
        "n_sims": int(len(truth)),
        "truth_median": float(np.median(truth)),
        "truth_min": float(truth.min()),
        "truth_max": float(truth.max()),
        "truth_spread_factor": float(truth.max() / truth.min()),
        "deficit_factor_vs_pta": float(PTA_MEASUREMENT["amplitude"] / np.median(truth)),
    }
    lt = np.log10(truth)
    out["truth_2sigma_dex"] = float(np.percentile(lt, 84) - np.percentile(lt, 16))
    return out


def evaluate_agent(mech, setting, data, fit_data):
    result = {"refit": setting["refit"]}
    coeffs = list(mech.COEFFS)
    if setting["refit"]:
        coeffs, seconds = call_with_timeout(
            mech.fit_coeffs, FIT_TIMEOUT, fit_data["stats"], fit_data["redshifts"],
            fit_data["params"], fit_data["truth"], list(mech.COEFFS))
        coeffs = [float(c) for c in coeffs]
        if len(coeffs) > MAX_COEFFS:
            result["contract_violation"] = (f"fit_coeffs returned {len(coeffs)} "
                                            f"constants, limit {MAX_COEFFS}")
            print(f"[eval] contract violation, recorded and scored anyway: "
                  f"{result['contract_violation']}", flush=True)
        result["fit_seconds"] = round(seconds, 3)
        result["n_fit_simulations"] = int(len(fit_data["truth"]))
    result["coeffs"] = [round(float(c), 10) for c in coeffs]
    pred, seconds = call_with_timeout(mech.predict_agwb, PREDICT_TIMEOUT,
                                      data["stats"], data["redshifts"],
                                      data["params"], coeffs)
    result["predict_seconds"] = round(seconds, 3)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape == data["truth"].shape and np.all(np.isfinite(pred)):
        result["per_simulation"] = per_sim_check(mech, data, coeffs, pred)
        if not result["per_simulation"]["passed"]:
            print(f"  [note] predict_agwb does not predict one simulation at a "
                  f"time: a simulation's value changes by up to a relative "
                  f"{result['per_simulation']['max_relative_difference']:.3g} when "
                  f"evaluated on its own; recorded for the judge, the metrics "
                  f"below are still computed", flush=True)
    result.update(compute_metrics(pred, data["truth"]))
    return result


def pa_score(metrics):
    r2 = metrics.get("r2_log10")
    if r2 is None or not np.isfinite(r2):
        return 0.0
    return round(max(0.0, min(1.0, float(r2))), 4)


def main(mechanism, output, data_dir=None, workers=12,
         predict_timeout=None, fit_timeout=None, stats_timeout=None):
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
    all_pa = list(PA_IDS)
    calibration_path = PACKAGE_DIR / "instrument_calibration.json"
    report = {
        "mechanism_path": mechanism_path,
        "protocol": {
            "reference_study": PAPER_REPORTED["reference_study"],
            "settings": "the 1P and CV sets of CAMELS-SIMBA, every simulation of "
                        "each set, the black hole populations of the snapshots "
                        "between z=0 and z=2, measured by the instrument of the "
                        "previous study",
            "held_out": "the agent explored IllustrisTNG only; SIMBA was never "
                        "accessible to it",
            "constants": "submitted COEFFS for PA1 and PA2; for PA3 the "
                         "submission's own fit_coeffs refits them on the SIMBA CV "
                         "simulations with the functional form fixed",
            "metrics": "each criterion is the coefficient of determination between "
                       "predicted and measured log10 amplitude over its setting, "
                       "floored at 0; rmse and mae are reported alongside but not "
                       "scored",
        },
        "paper_reported": PAPER_REPORTED,
        "settings": [], "pa_scores": {pid: 0.0 for pid in PA_IDS},
    }

    mech = None
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

    if report.get("error") is None and not calibration_path.exists():
        report["error"] = f"{calibration_path} not found"
        print(f"[eval] ERROR: {report['error']}", flush=True)

    if report.get("error") is None:
        with open(calibration_path) as f:
            k_amp = float(json.load(f)["k_amp"])
        qfac = mass_ratio_factor()
        bb = BasicCamelsBlackBox(data_dir=data_dir)
        timing = {"max_stats_seconds": 0.0}

        needed = sorted({(s["suite"], s["set"]) for s in SETTINGS}
                        | {s["fit_on"] for s in SETTINGS if s.get("fit_on")})
        jobs = []
        for suite, set_type in needed:
            for sim in load_params_table(bb, suite, set_type):
                for snap in SNAPSHOTS:
                    jobs.append(catalog_relpath(suite, set_type, sim, snap))
        jobs = list(dict.fromkeys(jobs))
        print(f"[eval] ensuring {len(jobs)} catalogs are cached "
              f"({workers} workers)", flush=True)
        prefetch(bb, jobs, workers)

        data_cache = {}
        try:
            for suite, set_type in needed:
                print(f"[eval] gathering {suite} {set_type}", flush=True)
                data_cache[(suite, set_type)] = gather(bb, suite, set_type, mech,
                                                       k_amp, qfac, timing)
        except Exception as exc:
            report["error"] = f"data gathering failed: {exc}"
            print(f"[eval] ERROR: {report['error']}", flush=True)

        if timing["max_stats_seconds"] > 10.0:
            report.setdefault("interface_problems", []).append(
                f"compute_statistics took up to {timing['max_stats_seconds']:.1f}s "
                f"per catalog, limit 10s")

        if report.get("error") is None:
            for setting in SETTINGS:
                label = (f"{setting['suite']} {setting['set']}"
                         + (" (refit)" if setting["refit"] else ""))
                print(f"[eval] {label}", flush=True)
                data = data_cache[(setting["suite"], setting["set"])]
                entry = {"suite": setting["suite"], "set": setting["set"],
                         "refit": setting["refit"]}
                entry.update(truth_context(data))
                entry["n_sims_skipped"] = data["n_skipped"]
                if data["skipped"]:
                    entry["skipped_sims"] = data["skipped"]
                fit_data = data_cache.get(setting.get("fit_on"))
                try:
                    entry["agent"] = evaluate_agent(mech, setting, data, fit_data)
                except Exception as exc:
                    print(f"  [error] {label}: {exc}", flush=True)
                    entry["agent"] = {"error": str(exc), "r2_log10": None,
                                      "rmse_dex": None}
                entry["pa"] = setting["pa"]
                entry["score"] = pa_score(entry["agent"])
                report["pa_scores"][setting["pa"]] = entry["score"]
                report["settings"].append(entry)
                a = entry["agent"]
                if a.get("r2_log10") is not None:
                    print(f"  agent: r2(log10 A)={a['r2_log10']:+.4f} "
                          f"rmse={a['rmse_dex']:.4f} dex mae={a['mae_dex']:.4f} dex "
                          f"-> {setting['pa']} {entry['score']:.4f}", flush=True)
                print(f"  truth: median={entry['truth_median']:.3e} "
                      f"spread x{entry['truth_spread_factor']:.1f} "
                      f"deficit x{entry['deficit_factor_vs_pta']:.2f} vs PTA "
                      f"({entry['n_sims']} sims)", flush=True)

    report["predictive_accuracy"] = round(
        sum(report["pa_scores"][p] for p in all_pa) / len(all_pa), 4)
    report["predictive_accuracy_total"] = len(all_pa)

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 64)
    print(f"[eval] predictive accuracy: {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    if report.get("interface_problems"):
        print(f"[eval] contract violations recorded for the judge: "
              f"{report['interface_problems']}")
    if report.get("error"):
        print(f"[eval] no setting was evaluated: {report['error']}")
    print(f"[eval] results: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
