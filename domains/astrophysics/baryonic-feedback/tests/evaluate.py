import ast
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import h5py
import numpy as np
from scipy.spatial import cKDTree
from archive import DreamsSimulator, trailing_int

HOST_DEFINITION = ("the most massive Friends-of-Friends group of the zoom-in whose dark "
                   "matter holds fewer than 2 percent low-resolution boundary particles "
                   "(GroupLenType[:, 2] against GroupLenType[:, 1]); boxes whose most "
                   "massive clean group falls outside (3e11, 5e12) Msun in M200c are "
                   "skipped")
CENTRAL_DEFINITION = ("the most massive gravitationally bound subhalo of the host group "
                      "(GroupFirstSub); its stellar mass is SubhaloMassType[4] and its "
                      "black hole mass is SubhaloMassType[5], both from the z=0 Subfind "
                      "catalog and converted to Msun with no factors of h")


def measure_central(cat_paths, host):
    with h5py.File(cat_paths[0], "r") as f:
        hubble = float(f["Header"].attrs["HubbleParam"])
        first = int(f["Group"]["GroupFirstSub"][host["index"]])
        if first < 0:
            return None
        sub = f["Subhalo"]
        mass_type = sub["SubhaloMassType"][first].astype(np.float64)
        len_type = sub["SubhaloLenType"][first].astype(np.int64)
        pos = sub["SubhaloPos"][first].astype(np.float64)
        half_rad = float(sub["SubhaloHalfmassRadType"][first][4])
    to_msun = 1e10 / hubble
    return {"subhalo": first,
            "pos": pos,
            "stellar_half_rad_ckpc_h": half_rad,
            "n_star_particles": int(len_type[4]),
            "mstar_msun": float(mass_type[4] * to_msun),
            "mbh_msun": float(mass_type[5] * to_msun)}


SNAPSHOT = 90
MAX_LOWRES_FRACTION = 0.02
MIN_HOST_DM_PARTICLES = 100_000
HOST_MASS_WINDOW_MSUN = (3e11, 5e12)


def progress(label, done, total, t0, note=""):
    elapsed = time.time() - t0
    frac = done / total if total else 1.0
    filled = int(round(30 * frac))
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    print(f"[{label}] |{'#' * filled}{'.' * (30 - filled)}| {done}/{total} "
          f"elapsed {elapsed:5.0f}s eta {eta:5.0f}s {note}", flush=True)


def read_param_table(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    if not lines[0].startswith("#"):
        raise RuntimeError(f"{path} has no header line naming its columns")
    columns = lines[0].lstrip("#").split()
    rows = {}
    for box_id, line in enumerate(lines[1:]):
        values = line.split()
        if len(values) != len(columns):
            raise RuntimeError(f"{path} row {box_id}: {len(values)} values for "
                               f"{len(columns)} columns")
        rows[box_id] = {c: float(v) for c, v in zip(columns, values)}
    return columns, rows


def find_host(cat_paths):
    if len(cat_paths) != 1:
        raise RuntimeError(f"expected a single-file catalog, got {cat_paths}")
    with h5py.File(cat_paths[0], "r") as f:
        header = dict(f["Header"].attrs)
        group = f["Group"]
        len_type = group["GroupLenType"][:].astype(np.int64)
        m200 = group["Group_M_Crit200"][:].astype(np.float64)
        r200 = group["Group_R_Crit200"][:].astype(np.float64)
        pos = group["GroupPos"][:].astype(np.float64)
    hubble = float(header["HubbleParam"])
    hi, lo = len_type[:, 1], len_type[:, 2]
    clean = (hi >= MIN_HOST_DM_PARTICLES) & (lo < MAX_LOWRES_FRACTION * (hi + lo))
    if not clean.any():
        return None
    idx = int(np.flatnonzero(clean)[np.argmax(m200[clean])])
    m200_msun = m200[idx] * 1e10 / hubble
    if not (HOST_MASS_WINDOW_MSUN[0] <= m200_msun <= HOST_MASS_WINDOW_MSUN[1]):
        return None
    return {"index": idx,
            "pos": pos[idx],
            "m200_msun": float(m200_msun),
            "r200_ckpc_h": float(r200[idx]),
            "r200_kpc": float(r200[idx] / hubble),
            "lowres_fraction": float(lo[idx] / max(hi[idx] + lo[idx], 1)),
            "type_start": len_type[:idx].sum(axis=0).tolist(),
            "type_count": len_type[idx].tolist(),
            "box_size": float(header["BoxSize"]),
            "hubble": hubble,
            "omega_m": float(header["Omega0"]),
            "redshift": float(header["Redshift"])}


MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
MIN_EVAL_SIMS = 20
MAX_HOST_MATCH_DEX = 0.15
ACCESS_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessed_sims.json")

CEILING_K_GRID = (4, 8, 16, 32, 64, 128, 256)
CEILING_FOLDS = 5
CEILING_RIDGE = 1e-6
REFERENCE_BINS = 12
TARGETS = ("log10_mstar", "log10_mbh")
PA_MEASURES = (("PA1", "submitted", "log10_mstar"),
               ("PA2", "submitted", "log10_mbh"),
               ("PA3", "extrapolation", "log10_mstar"),
               ("PA4", "extrapolation", "log10_mbh"))
PA_IDS = [pid for pid, _, _ in PA_MEASURES]

PARAM_KEYS = ("omega_m", "sigma_8", "a_sn1", "a_sn2", "a_agn")
PARAM_COLUMNS = {"omega_m": "Om", "sigma_8": "s8", "a_sn1": "SN1",
                 "a_sn2": "SN2", "a_agn": "BHFF"}


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
    for name in ("COEFFS", "predict_baryons", "fit_coeffs"):
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
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


def measure_box(bb, name):
    hydro = bb.catalog_paths(name, SNAPSHOT, "hydro")
    host = find_host(hydro)
    if host is None:
        raise RuntimeError("no clean Milky-Way-mass host in the hydrodynamic catalog")
    central = measure_central(hydro, host)
    if central is None:
        raise RuntimeError("the host group holds no bound subhalo")
    twin_host = find_host(bb.catalog_paths(name, SNAPSHOT, "nbody"))
    if twin_host is None:
        raise RuntimeError("no clean Milky-Way-mass host in the gravity-only twin")
    offset = abs(np.log10(twin_host["m200_msun"] / host["m200_msun"]))
    if offset > MAX_HOST_MATCH_DEX:
        raise RuntimeError(f"hydro and twin hosts differ by {offset:.3f} dex in M200c, "
                           f"above the {MAX_HOST_MATCH_DEX} dex matching limit")
    return {"m200_nbody": twin_host["m200_msun"],
            "r200_nbody": twin_host["r200_kpc"],
            "m200_hydro": host["m200_msun"],
            "mstar_msun": central["mstar_msun"],
            "mbh_msun": central["mbh_msun"]}


def measure_box_with_retry(bb, name, attempts=2):
    for attempt in range(attempts):
        try:
            return measure_box(bb, name)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0)


def build_props(rows, boxes, data):
    props = {"m200_nbody": np.array([data[b]["m200_nbody"] for b in boxes]),
             "r200_nbody": np.array([data[b]["r200_nbody"] for b in boxes])}
    for key in PARAM_KEYS:
        column = PARAM_COLUMNS[key]
        props[key] = np.array([rows[trailing_int(b)][column] for b in boxes])
    return props


def take(props, sel):
    return {k: v[sel] for k, v in props.items()}


def design_matrix(props, reduced=False):
    columns = [np.log10(props["m200_nbody"]), np.log10(props["a_sn1"]),
               np.log10(props["a_sn2"]), np.log10(props["a_agn"])]
    if not reduced:
        columns += [np.log10(props["r200_nbody"]), props["omega_m"], props["sigma_8"]]
    return np.column_stack(columns)


def standardise(a, b):
    mu, sd = a.mean(0), a.std(0) + 1e-12
    return (a - mu) / sd, (b - mu) / sd


def knn_predict(a, y_fit, b, k):
    a, b = standardise(a, b)
    tree = cKDTree(a)
    _, idx = tree.query(b, k=int(min(k, len(a))))
    idx = np.atleast_2d(idx.T).T
    return y_fit[idx].mean(axis=1)


def quadratic_features(a):
    cols = [np.ones(len(a))] + [a[:, i] for i in range(a.shape[1])]
    for i in range(a.shape[1]):
        for j in range(i, a.shape[1]):
            cols.append(a[:, i] * a[:, j])
    return np.column_stack(cols)


def ridge_predict(a, y_fit, b, alpha=CEILING_RIDGE):
    a, b = standardise(a, b)
    xa, xb = quadratic_features(a), quadratic_features(b)
    gram = xa.T @ xa + alpha * len(xa) * np.eye(xa.shape[1])
    return xb @ np.linalg.solve(gram, xa.T @ y_fit)


def ceiling_family():
    members = [(f"knn{k}_{name}", reduced,
                (lambda k_: (lambda a, y, b: knn_predict(a, y, b, k_)))(k))
               for k in CEILING_K_GRID for name, reduced in (("full", False), ("reduced", True))]
    members += [("quadratic_full", False, ridge_predict),
                ("quadratic_reduced", True, ridge_predict)]
    return members


def cv_rmse(fn, a, y, folds, seed):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    pred = np.empty(len(y), dtype=np.float64)
    for f in range(folds):
        sel, rest = fold == f, fold != f
        pred[sel] = fn(a[rest], y[rest], a[sel])
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def ceiling_predict(props_fit, y_fit, props_score, seed=0):
    best, best_rmse, best_name = None, np.inf, None
    for name, reduced, fn in ceiling_family():
        a = design_matrix(props_fit, reduced)
        if len(a) < 2 * CEILING_FOLDS:
            continue
        try:
            score = cv_rmse(fn, a, y_fit, CEILING_FOLDS, seed)
        except Exception:
            continue
        if score < best_rmse:
            best, best_rmse, best_name = (reduced, fn), score, name
    if best is None:
        return np.full(len(next(iter(props_score.values()))), float(np.mean(y_fit))), "mean", np.nan
    reduced, fn = best
    pred = fn(design_matrix(props_fit, reduced), y_fit, design_matrix(props_score, reduced))
    return pred, best_name, best_rmse


def irreducible_scatter(props, y):
    a, _ = standardise(design_matrix(props, reduced=True), design_matrix(props, reduced=True))
    tree = cKDTree(a)
    _, idx = tree.query(a, k=2)
    return float(np.sqrt(np.mean((y - y[idx[:, 1]]) ** 2) / 2.0))


def reference_predict(props_fit, y_fit, props_score):
    x_fit, x_score = np.log10(props_fit["m200_nbody"]), np.log10(props_score["m200_nbody"])
    edges = np.quantile(x_fit, np.linspace(0.0, 1.0, REFERENCE_BINS + 1))
    edges = np.unique(edges)
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x_fit >= lo) & (x_fit <= hi)
        if sel.sum() >= 5:
            centres.append(0.5 * (lo + hi))
            medians.append(np.median(y_fit[sel]))
    if len(centres) < 2:
        return np.full(len(x_score), float(np.median(y_fit)))
    return np.interp(x_score, np.array(centres), np.array(medians))


def metrics(pred, truth):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
        return {"r2": None, "rmse": None, "n": int(truth.size),
                "error": "predictions are not a finite value per system"}
    return {"r2": float(1 - np.mean((pred - truth) ** 2) / np.var(truth)),
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "n": int(truth.size)}


def target_columns(data, boxes):
    mstar = np.array([data[b]["mstar_msun"] for b in boxes])
    mbh = np.array([data[b]["mbh_msun"] for b in boxes])
    return {"log10_mstar": (np.log10(np.maximum(mstar, 1e-30)), mstar > 0),
            "log10_mbh": (np.log10(np.maximum(mbh, 1e-30)), mbh > 0)}


def cv_predict(predictor, props, y, folds, seed):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    out = np.empty(len(y), dtype=np.float64)
    for f in range(folds):
        score_sel, fit_sel = fold == f, fold != f
        out[score_sel] = predictor(take(props, fit_sel), y[fit_sel],
                                   take(props, score_sel))
    return out


def cv_constants_predict(mech, props, truths, score_sel, folds, seed):
    idx = np.flatnonzero(score_sel)
    fold = np.random.default_rng(seed).permutation(len(idx)) % folds
    both = truths["log10_mbh"][1]
    pred = np.empty((len(idx), 2), dtype=np.float64)
    fold_coeffs, seconds = [], 0.0
    for f in range(folds):
        fit_idx = idx[(fold != f) & both[idx]]
        y_fit = np.column_stack([truths[t][0][fit_idx] for t in TARGETS])
        coeffs, spent = call_with_timeout(mech.fit_coeffs, FIT_TIMEOUT, take(props, fit_idx),
                                          y_fit, list(mech.COEFFS))
        coeffs = [float(c) for c in coeffs]
        if len(coeffs) > MAX_COEFFS:
            print(f"[eval] contract violation, recorded and scored anyway: fit_coeffs returned "
                  f"{len(coeffs)} constants, limit {MAX_COEFFS}", flush=True)
        out, _ = call_with_timeout(mech.predict_baryons, PREDICT_TIMEOUT,
                                   take(props, idx[fold == f]), coeffs)
        out = np.asarray(out, dtype=np.float64)
        if out.shape != (int(np.count_nonzero(fold == f)), 2):
            raise RuntimeError(f"predict_baryons returned {out.shape} on fold {f}")
        pred[fold == f] = out
        fold_coeffs.append([round(c, 8) for c in coeffs])
        seconds += spent
    return pred, fold_coeffs, seconds


def score_pass(mech, coeffs, props, truths, fit_sel, score_sel, label, folds, seed,
               cv_constants=False):
    entry = {"label": label,
             "n_fit": int(np.count_nonzero(fit_sel)) if fit_sel is not None else 0,
             "n_score": int(np.count_nonzero(score_sel)), "coeffs": None,
             "comparators": ("fitted on this pass's fitting set" if fit_sel is not None
                             else f"{folds}-fold cross-validation over the scored set"),
             "agent": {}, "reference": {}, "ceiling": {}}
    if fit_sel is not None and coeffs is None:
        fit_sel = fit_sel & truths["log10_mbh"][1]
        entry["n_fit"] = int(np.count_nonzero(fit_sel))
        if not entry["n_fit"]:
            raise RuntimeError("no system in the fitting set has both masses defined")
        fit_props = take(props, fit_sel)
        y_fit = np.column_stack([truths[t][0][fit_sel] for t in TARGETS])
        coeffs, seconds = call_with_timeout(mech.fit_coeffs, FIT_TIMEOUT, fit_props,
                                            y_fit, list(mech.COEFFS))
        coeffs = [float(c) for c in coeffs]
        if len(coeffs) > MAX_COEFFS:
            entry["contract_violation"] = (f"fit_coeffs returned {len(coeffs)} constants, "
                                           f"limit {MAX_COEFFS}")
            print(f"[eval] contract violation, recorded and scored anyway: "
                  f"{entry['contract_violation']}", flush=True)
        entry["fit_seconds"] = round(seconds, 3)
    score_props = take(props, score_sel)
    if cv_constants:
        pred, entry["coeffs"], seconds = cv_constants_predict(mech, props, truths, score_sel,
                                                              folds, seed)
        entry["fit_seconds"] = round(seconds, 3)
        entry["constants"] = (f"refit by fit_coeffs on {folds - 1} of {folds} folds of "
                              "the scored set and used on the remaining fold")
    else:
        entry["coeffs"] = [round(float(c), 8) for c in coeffs]
        pred, seconds = call_with_timeout(mech.predict_baryons, PREDICT_TIMEOUT,
                                          score_props, coeffs)
        entry["predict_seconds"] = round(seconds, 3)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != (int(np.count_nonzero(score_sel)), 2):
        raise RuntimeError(f"predict_baryons returned {pred.shape}, expected "
                           f"({int(np.count_nonzero(score_sel))}, 2)")
    for column, target in enumerate(TARGETS):
        y, defined = truths[target]
        keep = defined[score_sel]
        truth = y[score_sel][keep]
        entry["agent"][target] = metrics(pred[:, column][keep], truth)
        props_score = take(score_props, keep)
        if fit_sel is None:
            reference = cv_predict(reference_predict, props_score, truth, folds, seed)
            ceiling = cv_predict(lambda pf, yf, ps: ceiling_predict(pf, yf, ps, seed)[0],
                                 props_score, truth, folds, seed)
            chosen, cv_rmse_fit = "cross-validated over the scored set", None
            scatter = irreducible_scatter(props_score, truth)
        else:
            fit_keep = defined[fit_sel]
            y_fit = y[fit_sel][fit_keep]
            props_fit = take(take(props, fit_sel), fit_keep)
            reference = reference_predict(props_fit, y_fit, props_score)
            ceiling, chosen, cv_rmse_fit = ceiling_predict(props_fit, y_fit, props_score, seed)
            scatter = irreducible_scatter(props_fit, y_fit)
        entry["reference"][target] = metrics(reference, truth)
        entry["ceiling"][target] = metrics(ceiling, truth)
        entry["ceiling"][target]["family_member"] = chosen
        entry["ceiling"][target]["cv_rmse_on_fitting_set"] = (
            None if cv_rmse_fit is None else round(float(cv_rmse_fit), 4))
        entry["ceiling"][target]["nearest_neighbour_scatter"] = round(scatter, 4)
        agent_rmse = entry["agent"][target].get("rmse")
        ceiling_rmse = entry["ceiling"][target].get("rmse")
        entry["ceiling"][target]["beaten_by_agent"] = bool(
            agent_rmse is not None and ceiling_rmse is not None and agent_rmse < ceiling_rmse)
    return entry


def clip01(value):
    return float(min(1.0, max(0.0, value)))


def pass_scores(entry):
    out = {}
    for target in TARGETS:
        agent = (entry["agent"].get(target) or {}).get("rmse")
        reference = (entry["reference"].get(target) or {}).get("rmse")
        if agent is None or reference is None or not reference > 0:
            out[target] = 0.0
        else:
            out[target] = round(clip01(1.0 - agent / reference), 4)
    return out


def main(mechanism, output, data_dir=None, n_sims=256, seed=0, workers=8, folds=5,
         accessed_file=ACCESS_RECORD, cross_validate_constants=False, predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s, fit {FIT_TIMEOUT:.0f}s", flush=True)
    started = time.time()
    timing = {}
    mechanism_path = str(mechanism)
    report = {
        "mechanism_path": mechanism_path,
        "protocol": {
            "held_out_set": "every simulation of the DREAMS CDM Milky-Way zoom-in "
                            "suite whose data the run never downloaded, read from "
                            "accessed_sims.json recorded during the run",
            "host_definition": HOST_DEFINITION,
            "central_galaxy_definition": CENTRAL_DEFINITION,
            "targets": "log10 stellar mass and log10 black hole mass of the central "
                       "galaxy in Msun; the black hole target is scored only on the "
                       "systems whose central galaxy holds one",
            "model_inputs": "the mass and radius inside R200c of the host's "
                            "gravity-only twin and the five varied parameters",
            "host_matching": f"the hydrodynamic and twin hosts are found independently "
                             f"by the same rule and have to agree in M200c to within "
                             f"{MAX_HOST_MATCH_DEX} dex",
            "passes": ["submitted constants on the whole held-out set",
                       "constants refit by the submission's own fit_coeffs on the "
                       "held-out simulations whose supernova wind energy amplitude "
                       "lies below its median, scored on those above it"],
            "reference": "a predictor given the halo mass of the gravity-only twin and "
                         "nothing about the feedback, fitted and scored on exactly the "
                         "same systems as the submission in every pass",
            "ceiling": "the best a function of the model's own inputs can do, "
                       "estimated by selecting among nearest-neighbour smoothers over "
                       f"{len(CEILING_K_GRID)} neighbour counts and quadratic fits, on "
                       "the full input set and on the set with the two inert "
                       "cosmological inputs and the redundant radius dropped, by "
                       f"{CEILING_FOLDS}-fold cross-validation on the fitting set; "
                       "what it leaves behind is the scatter between systems that "
                       "share those inputs, which no model reading only those inputs "
                       "can remove. Each entry also carries the family member chosen, "
                       "its cross-validated error, a model-free nearest-neighbour "
                       "estimate of that scatter, and whether the submission beat it",
            "comparators_in_pass_1": f"{folds}-fold cross-validation over the scored "
                                     "set, so that no system contributes to its own "
                                     "reference or ceiling prediction",
            "metrics": "r2 and rmse of log10 mass per target; a criterion scores 1 "
                       "minus the agent's rmse over the reference's rmse on the same "
                       "systems, floored at 0, so an exact prediction scores 1 and a "
                       "prediction no better than the halo mass alone scores 0",
            "criteria": {pid: f"{pass_name} pass, {target}"
                         for pid, pass_name, target in PA_MEASURES},
            "seed": seed,
        },
        "pa_scores": {},
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
            print("[eval] WARNING: access-based exclusion waived; the metrics are not "
                  "blind for this run", flush=True)
        elif accessed_file == ACCESS_RECORD and not os.path.isfile(accessed_file):
            accessed = set()
            print("[eval] no access record: the run read no simulation, so none is excluded")
        else:
            if not os.path.isfile(accessed_file):
                raise RuntimeError(
                    f"{accessed_file} not found; the held-out set is the complement of "
                    "the simulations the run accessed, so the evaluation cannot proceed")
            with open(accessed_file) as f:
                accessed = set(json.load(f).get("boxes", []))
            print(f"[eval] excluding {len(accessed)} simulations the run accessed")

        bb = DreamsSimulator(data_dir=data_dir)
        rng = np.random.default_rng(seed)
        info = bb.list_simulations()
        columns, rows = read_param_table(bb.params_file())
        if len(rows) != info["n_simulations"]:
            raise RuntimeError(f"parameter table has {len(rows)} rows for "
                               f"{info['n_simulations']} simulations")
        missing = [c for c in PARAM_COLUMNS.values() if c not in columns]
        if missing:
            raise RuntimeError(f"parameter table lacks the columns {missing}")

        eligible = [n for n in info["simulations"]
                    if n not in accessed and trailing_int(n) in rows]
        if n_sims and n_sims < len(eligible):
            print(f"[eval] WARNING: sampling {n_sims} of {len(eligible)} eligible "
                  "simulations instead of the full complement")
            eligible = sorted(rng.choice(eligible, size=n_sims, replace=False),
                              key=trailing_int)
        report["n_excluded_accessed"] = len(accessed)
        report["n_eval_sims_requested"] = len(eligible)

        t_measure = time.time()
        data, skipped = {}, []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(measure_box_with_retry, bb, name): name
                       for name in eligible}
            for done, future in enumerate(as_completed(futures), start=1):
                name = futures[future]
                try:
                    data[name] = future.result()
                except Exception as exc:
                    print(f"  [skip] {name}: {exc}", flush=True)
                    skipped.append(name)
                if done % 25 == 0 or done == len(futures):
                    progress("measure", done, len(futures), t_measure,
                             f"{len(data)} usable")
        timing["measure_seconds"] = round(time.time() - t_measure, 1)
        report["n_measure_skipped"] = len(skipped)

        if len(data) < MIN_EVAL_SIMS:
            report["error"] = f"only {len(data)} usable held-out simulations"
            print(f"[eval] ERROR: {report['error']}", flush=True)
        else:
            boxes = sorted(data, key=trailing_int)
            props = build_props(rows, boxes, data)
            truths = target_columns(data, boxes)
            report["n_eval_sims_used"] = len(boxes)
            report["eval_sims"] = list(boxes)
            report["n_without_black_hole"] = int((~truths["log10_mbh"][1]).sum())

            everything = np.ones(len(boxes), dtype=bool)
            below = props["a_sn1"] < np.median(props["a_sn1"])
            report["constants_cross_validated"] = bool(cross_validate_constants)
            first_label = (f"constants refit by fit_coeffs in {folds}-fold cross-validation"
                           if cross_validate_constants else "submitted constants")
            splits = [("submitted", first_label, None, everything),
                      ("extrapolation",
                       "refit below the median wind energy, scored above it",
                       below, ~below)]

            report["passes"] = []
            scored = {}
            for name, label, fit_sel, score_sel in splits:
                t_pass = time.time()
                print(f"[eval] pass: {label}", flush=True)
                try:
                    entry = score_pass(mech, list(mech.COEFFS) if fit_sel is None else None,
                                       props, truths, fit_sel, score_sel, label,
                                       folds, seed,
                                       cv_constants=bool(cross_validate_constants)
                                       and fit_sel is None)
                except Exception as exc:
                    print(f"  [error] {label}: {exc}", flush=True)
                    entry = {"label": label, "error": str(exc),
                             "agent": {t: {"rmse": None} for t in TARGETS},
                             "reference": {t: {"rmse": None} for t in TARGETS},
                             "ceiling": {t: {"rmse": None} for t in TARGETS}}
                entry["pass"] = name
                entry["seconds"] = round(time.time() - t_pass, 1)
                entry["scores"] = pass_scores(entry)
                scored[name] = entry["scores"]
                report["passes"].append(entry)
                for role in ("agent", "reference", "ceiling"):
                    line = "  ".join(
                        f"{t}: rmse={entry[role][t].get('rmse')} r2={entry[role][t].get('r2')}"
                        for t in TARGETS)
                    print(f"  {role:9s} {line}", flush=True)
                print(f"  {'scores':9s} " + "  ".join(
                    f"{t}: {entry['scores'][t]}" for t in TARGETS), flush=True)

            for pid, pass_name, target in PA_MEASURES:
                report["pa_scores"][pid] = scored.get(pass_name, {}).get(target, 0.0)

    for pid in PA_IDS:
        report["pa_scores"].setdefault(pid, 0.0)
    report["predictive_accuracy"] = round(
        float(np.mean([report["pa_scores"][pid] for pid in PA_IDS])), 4)
    report["predictive_accuracy_total"] = len(PA_IDS)
    timing["total_seconds"] = round(time.time() - started, 1)
    timing["started_unix"] = round(started, 1)
    timing["finished_unix"] = round(time.time(), 1)
    report["timing"] = timing

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 64)
    print(f"[eval] predictive accuracy: {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    print(f"[eval] wall clock: {timing['total_seconds']}s "
          f"(measuring {timing.get('measure_seconds')}s)")
    if report.get("error"):
        print(f"[eval] nothing was scored: {report['error']}")
    print(f"[eval] results: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
