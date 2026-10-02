import ast
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import fire
import numpy as np

PROBLEM = "arc_tokamak"

ARC = dict(
    R_major=4.62,
    a_minor=1.18,
    B_0=11.4,
    elongation=1.8,
    Ip=12.0e6,
    Z_eff=1.5,
    impurity="Ne",
    n_e_ped=2.1e20,
    P_ped=4.5e5,
    rho_ped_top=0.93,
    T_sep=1.0,
    P_aux=20.0e6,
    electron_heat_fraction=0.45,
    aux_location=0.0,
    aux_width=0.1,
)
ARMS = {
    "constant": {"model_name": "constant", "chi_i": 1.0, "chi_e": 1.0,
                 "D_e": 1.0, "V_e": -0.33},
    "constant_chi2": {"model_name": "constant", "chi_i": 2.0, "chi_e": 2.0,
                      "D_e": 1.0, "V_e": -0.33},
    "CGM": {"model_name": "CGM"},
    "CGM_stiff4": {"model_name": "CGM", "chi_stiff": 4.0},
    "CGM_alpha3": {"model_name": "CGM", "alpha": 3.0},
    "bohm-gyrobohm": {"model_name": "bohm-gyrobohm"},
    "bgb_bohm_x2": {"model_name": "bohm-gyrobohm", "chi_i_bohm_multiplier": 2.0,
                    "chi_e_bohm_multiplier": 2.0},
    "qlknn": {"model_name": "qlknn"},
    "qlknn_no_ETG": {"model_name": "qlknn", "include_ETG": False},
    "qlknn_no_TEM": {"model_name": "qlknn", "include_TEM": False},
    "qlknn_no_ITG": {"model_name": "qlknn", "include_ITG": False},
    "qlknn_DVeff": {"model_name": "qlknn", "DV_effective": True},
    "qlknn_coll025": {"model_name": "qlknn", "collisionality_multiplier": 0.25},
    "tglfnn-ukaea": {"model_name": "tglfnn-ukaea"},
    "qualikiz": {"model_name": "qualikiz", "n_processes": 8, "n_max_runs": 2},
}
PRIMARY = ["qlknn", "tglfnn-ukaea", "qualikiz", "CGM", "bohm-gyrobohm", "constant"]


def transport_limits():
    return {"chi_min": 0.05, "chi_max": 3.0, "D_e_min": 0.05}


PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
T_FINAL = 12.0
N_RHO = 25
PROTOCOL_VERSION = 4
TOO_SLOW = {"qualikiz"}
TEST_ARMS = [a for a in PRIMARY if a not in TOO_SLOW]
VARIANT_ARMS = [a for a in ARMS if a not in PRIMARY and a not in TOO_SLOW]
BASELINE_AXES = {
    "pped_multiplier": (0.85, 1.15),
    "P_aux": (12.0e6, 30.0e6),
    "n_e_ped": (1.8e20, 2.4e20),
    "Z_eff": (1.3, 1.8),
}
HEATING_PAIR = (14.0e6, 26.0e6)
NO_ALPHA_P_AUX = (80.0e6, 160.0e6)
PEDESTAL_PAIR = (0.85, 1.15)
FAMILIES = {"cross_model": 12, "heating_pair": 5, "pedestal_pair": 5,
            "variant": 9, "no_alpha": 5}
PA_IDS = [f"PA{i}" for i in range(1, 11)]
GRADIENT_RHO = 0.6
SENSITIVITY = (("LTi", 0, 0), ("LTe", 1, 1), ("Lne", 2, 2))
COEFFICIENTS = ("chi_i", "chi_e", "D_e", "V_e")
SENSITIVITY_STEP = 0.10
SENSITIVITY_BLOCK = 2
SENSITIVITY_OFFSETS = (0, 1)
SIGN_FLOOR = 0.01
COEFFICIENT_FLOOR = transport_limits()["chi_min"]
SENSITIVITY_T_FINAL = 1e-3
SENSITIVITY_SMOOTHING = 0.0
BH = dict(C1=1.17302e-9, C2=1.51361e-2, C3=7.51886e-2, C4=4.60643e-3,
          C5=1.35000e-2, C6=-1.06750e-4, C7=1.36600e-5, BG=34.3827, MRC2=1124656.0)
E_FUSION_J = 17.6e6 * 1.602176634e-19


def fingerprint(seed):
    payload = json.dumps({"seed": seed, "t_final": T_FINAL, "n_rho": N_RHO,
                          "families": FAMILIES, "axes": BASELINE_AXES,
                          "heating_pair": HEATING_PAIR, "pedestal_pair": PEDESTAL_PAIR,
                          "no_alpha_p_aux": NO_ALPHA_P_AUX, "arc": ARC,
                          "arms": sorted(TEST_ARMS + VARIANT_ARMS),
                          "sensitivity": {"step": SENSITIVITY_STEP,
                                          "block": SENSITIVITY_BLOCK,
                                          "offsets": SENSITIVITY_OFFSETS,
                                          "sign_floor": SIGN_FLOOR,
                                          "t_final": SENSITIVITY_T_FINAL,
                                          "smoothing_width": SENSITIVITY_SMOOTHING},
                          "version": PROTOCOL_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def reactivity(t_kev):
    t = np.clip(np.asarray(t_kev, dtype=float), 0.2, 100.0)
    num = t * (BH["C2"] + t * (BH["C4"] + t * BH["C6"]))
    den = 1.0 + t * (BH["C3"] + t * (BH["C5"] + t * BH["C7"]))
    theta = t / (1.0 - num / den)
    xi = (BH["BG"] ** 2 / (4.0 * theta)) ** (1.0 / 3.0)
    sv = BH["C1"] * theta * np.sqrt(xi / (BH["MRC2"] * t ** 3)) * np.exp(-3.0 * xi)
    return sv * 1e-6


def fusion_power(n_i, t_i_kev, volume):
    n_d_n_t = (np.asarray(n_i, dtype=float) ** 2) / 4.0
    density = n_d_n_t * reactivity(t_i_kev) * E_FUSION_J
    dv = np.gradient(np.asarray(volume, dtype=float))
    return float(np.sum(density * dv))


def square_wave(n_cells, offset):
    blocks = (np.arange(n_cells) - int(offset)) // SENSITIVITY_BLOCK
    return np.where(blocks % 2 == 0, 1.0, -1.0)


def stepped_faces(n_faces, offset):
    wave = square_wave(n_faces - 1, offset)
    hit = np.zeros(n_faces, dtype=bool)
    hit[1:-1] = wave[1:] != wave[:-1]
    return hit


def stepped_profiles(profiles, channel, offset, rho):
    out = np.array(profiles, dtype=float)
    n_cells = out.shape[2] - 2
    wave = square_wave(n_cells, offset)
    gradient = face_inverse_scale_length(out[:, channel, :], rho)
    shift = 0.5 * SENSITIVITY_STEP * gradient * np.diff(np.asarray(rho, dtype=float))
    cell_shift = np.zeros((out.shape[0], n_cells))
    for k in np.flatnonzero(stepped_faces(n_cells + 1, offset)):
        cell_shift[:, k - 1] = shift[:, k]
        cell_shift[:, k] = shift[:, k]
    out[:, channel, 1:-1] *= 1.0 + cell_shift * wave
    return out


def face_inverse_scale_length(values, rho):
    values = np.asarray(values, dtype=float)
    d_rho = np.diff(np.asarray(rho, dtype=float))
    grad = np.diff(values, axis=-1) / d_rho
    mid = 0.5 * (values[..., 1:] + values[..., :-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(-grad / mid, nan=0.0, posinf=0.0, neginf=0.0)


def response_sign(base, stepped, base_gradient, stepped_gradient):
    delta = stepped - base
    moved = np.abs(delta) > SIGN_FLOOR * np.maximum(np.abs(base), COEFFICIENT_FLOOR)
    return np.where(moved, np.sign(delta * (stepped_gradient - base_gradient)), 0.0)


def sensitivity_signs(read, profiles, rho):
    profiles = np.asarray(profiles, dtype=float)
    base, base_used = read(profiles)
    base, base_used = np.asarray(base, dtype=float), np.asarray(base_used, dtype=float)
    cells = (Ellipsis, slice(1, -1))
    deviation = float(np.max(np.abs(base_used - profiles)[cells]
                             / np.abs(profiles)[cells]))
    n_faces = base.shape[2]
    out = {}
    for name, channel, coefficient in SENSITIVITY:
        signs = np.zeros((base.shape[0], n_faces))
        base_gradient = face_inverse_scale_length(base_used[:, channel, :], rho)
        for offset in SENSITIVITY_OFFSETS:
            given = stepped_profiles(profiles, channel, offset, rho)
            stepped, used = read(given)
            stepped, used = np.asarray(stepped, dtype=float), np.asarray(used, dtype=float)
            deviation = max(deviation, float(np.max(np.abs(used - given)[cells]
                                                    / np.abs(given)[cells])))
            gradient = face_inverse_scale_length(used[:, channel, :], rho)
            hit = stepped_faces(n_faces, offset)
            signs[:, hit] = response_sign(base[:, coefficient, :],
                                          stepped[:, coefficient, :],
                                          base_gradient, gradient)[:, hit]
        out[name] = signs
    return out, deviation


def pooled_r2(pred, truth):
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 2:
        return None
    residual = np.sum((pred[ok] - truth[ok]) ** 2)
    total = np.sum((truth[ok] - truth[ok].mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else None


def profile_r2(pred, truth):
    scores = []
    for channel in range(pred.shape[1]):
        p, t = pred[:, channel, :], truth[:, channel, :]
        scale = float(np.std(t)) or 1.0
        scores.append(pooled_r2(p / scale, t / scale))
    valid = [s for s in scores if s is not None]
    return float(np.mean(valid)) if valid else None


def cross_model_groups(specs):
    groups = {}
    for spec in specs:
        if spec["family"] == "cross_model":
            groups.setdefault(spec["baseline_id"], []).append(spec["index"])
    return {k: v for k, v in groups.items() if len(v) > 1}


def intervention_pairs(specs, family, axis):
    pairs = {}
    for spec in specs:
        if spec["family"] == family:
            pairs.setdefault(spec["pair_key"], []).append(
                (spec["baseline"][axis], spec["index"]))
    return {k: sorted(v) for k, v in pairs.items() if len(v) == 2}


def paired_response_r2(power, truth_power, pairs):
    if not pairs:
        return None
    low = [v[0][1] for v in pairs.values()]
    high = [v[1][1] for v in pairs.values()]
    return pooled_r2([power[h] - power[l] for l, h in zip(low, high)],
                     [truth_power[h] - truth_power[l] for l, h in zip(low, high)])


def family_rows(specs, family):
    return [s["index"] for s in specs if s["family"] == family]


def predicted_power(pred, truth):
    out = []
    for i in range(pred.shape[0]):
        n_i = pred[i, 2, :] * float(truth["n_i_over_n_e"][i])
        out.append(fusion_power(n_i, pred[i, 0, :], truth["volume"][i]))
    return np.asarray(out, dtype=float)


def measure(pred, truth_stack, truth, specs):
    rho = np.asarray(truth["rho"][0], dtype=float)
    peaking, gradient = [], []
    for i in range(pred.shape[0]):
        n_e = pred[i, 2, :]
        d_volume = np.gradient(np.asarray(truth["volume"][i], dtype=float))
        n_e_volume_avg = float(np.sum(n_e * d_volume) / np.sum(d_volume))
        peaking.append(float(np.interp(0.2, rho, n_e) / n_e_volume_avg))
        t_i = pred[i, 0, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            a_over_lti = np.nan_to_num(-np.gradient(t_i, rho) / t_i, nan=0.0,
                                       posinf=0.0, neginf=0.0)
        gradient.append(float(np.interp(GRADIENT_RHO, rho, a_over_lti)))
    truth_gradient = np.array([float(np.interp(GRADIENT_RHO, rho, row))
                               for row in truth["a_over_LTi"]])
    power = predicted_power(pred, truth)
    truth_power = truth["P_fusion"]

    groups = cross_model_groups(specs)
    members = [i for group in groups.values() for i in group]
    alpha_rows = family_rows(specs, "no_alpha")
    block = {
        "cross_model_r2": pooled_r2(power[members], truth_power[members])
        if members else None,
        "peaking_r2": pooled_r2(peaking, truth["n_e_peaking"]),
        "a_over_LTi_r2": pooled_r2(np.array(gradient), truth_gradient),
        "heating_response_r2": paired_response_r2(
            power, truth_power, intervention_pairs(specs, "heating_pair", "P_aux")),
        "pedestal_response_r2": paired_response_r2(
            power, truth_power,
            intervention_pairs(specs, "pedestal_pair", "pped_multiplier")),
        "no_alpha_r2": pooled_r2(power[alpha_rows], truth_power[alpha_rows])
        if alpha_rows else None,
        "profile_r2": profile_r2(pred, truth_stack),
        "P_fusion_r2": pooled_r2(power, truth_power),
        "n_model_pairs": sum(len(v) * (len(v) - 1) // 2 for v in groups.values()),
    }
    return block


def sign_scores(signs, truth):
    faces = np.asarray(truth["sign_faces"], dtype=bool)
    block = {"n_sign_faces": int(faces.sum())}
    for name, _, _ in SENSITIVITY:
        want = np.asarray(truth[f"sign_{name}"], dtype=float)
        block[f"sign_{name}_model_responds"] = (float(np.mean(want[faces] != 0.0))
                                                if faces.any() else None)
        if signs is None or name not in signs or not faces.any():
            block[f"sign_{name}_agreement"] = None
            continue
        pred = np.asarray(signs[name], dtype=float)
        block[f"sign_{name}_agreement"] = float(np.mean(pred[faces] == want[faces]))
    return block


PA_MEASURES = (("PA1", "cross_model_r2"), ("PA2", "peaking_r2"),
               ("PA3", "a_over_LTi_r2"), ("PA4", "heating_response_r2"),
               ("PA5", "pedestal_response_r2"), ("PA6", "no_alpha_r2"),
               ("PA7", "profile_r2"), ("PA8", "sign_LTi_agreement"),
               ("PA9", "sign_LTe_agreement"), ("PA10", "sign_Lne_agreement"))


def normalized(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def pa_scores(block):
    return {pid: round(normalized((block or {}).get(key)), 4) for pid, key in PA_MEASURES}


def fit_split(specs):
    fit = np.array([s["family"] == "cross_model" and s["baseline_id"] % 2 == 0
                    for s in specs], dtype=bool)
    return fit, ~fit


def subset(props, truth, specs, sel):
    rows = np.flatnonzero(np.asarray(sel))
    picked = []
    for position, row in enumerate(rows):
        picked.append(dict(specs[row], index=position, setting=specs[row]["index"]))
    return ([props[row] for row in rows],
            {key: np.asarray(value)[rows] for key, value in truth.items()},
            picked)


def load_test_set(out_dir=None):
    out = Path(out_dir) if out_dir else TEST_SET_DIR
    with open(out / TEST_SET_MANIFEST.name) as handle:
        manifest = json.load(handle)
    archive = np.load(out / TEST_SET_FILE.name)
    n = manifest["n_settings"]
    props = [{"reported": {}, "transport": {}} for _ in range(n)]
    truth = {}
    for key in archive.files:
        parts = key.split("|")
        if parts[0] != "props":
            truth[parts[1]] = archive[key]
        elif parts[2] == "config":
            props[int(parts[1])]["config"] = json.loads(str(archive[key]))
        else:
            props[int(parts[1])][parts[2]][parts[3]] = archive[key]
    return manifest, props, truth

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_profiles", "fit_coeffs")
TRANSPORT_MEMBER = "transport_coefficients"
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
STATED_PREDICT_S = 120.0
STATED_TRANSPORT_S = 120.0
STATED_FIT_S = 600.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
CHANNELS = ("T_i", "T_e", "n_e")


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
    worker.start()
    send.close()
    name = getattr(fn, "__name__", str(fn))
    start, ready = time.time(), False
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
    return payload


def timed(label, settings, fn, timeout, *args):
    print(f"[eval] {label}: {settings} settings, budget {timeout:.0f}s ...", flush=True)
    start = time.time()
    try:
        out = call_with_timeout(fn, timeout, *args)
    except Exception as exc:
        print(f"[eval] {label}: {type(exc).__name__} after {time.time() - start:.1f}s",
              flush=True)
        raise
    print(f"[eval] {label}: done in {time.time() - start:.1f}s", flush=True)
    return out


class ImportScan(ast.NodeVisitor):
    def __init__(self):
        self.modules = []

    def visit_If(self, node):
        test = node.test
        guard = (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                 and test.left.id == "__name__")
        if not guard:
            self.generic_visit(node)

    def visit_Import(self, node):
        self.modules += [a.name.split(".")[0] for a in node.names]

    def visit_ImportFrom(self, node):
        if node.module:
            self.modules.append(node.module.split(".")[0])


def load_mechanism(path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("agent_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blocking, problems = [], []
    for name in REQUIRED_MEMBERS:
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
    if not hasattr(module, TRANSPORT_MEMBER):
        problems.append(f"missing {TRANSPORT_MEMBER}; PA8 to PA10 score 0")
    if hasattr(module, "COEFFS"):
        try:
            n_coeffs = len(list(module.COEFFS))
        except TypeError:
            n_coeffs = None
            problems.append("COEFFS is not a sequence")
        if n_coeffs is not None and n_coeffs > MAX_COEFFS:
            problems.append(f"COEFFS has {n_coeffs} entries, limit {MAX_COEFFS}")
    scan = ImportScan()
    with open(path) as f:
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy: {bad}")
    return module, blocking, problems


def read_test_set(seed):
    if not (protocol.TEST_SET_FILE.is_file() and protocol.TEST_SET_MANIFEST.is_file()):
        raise RuntimeError(f"no test set at {protocol.TEST_SET_FILE}")
    manifest, props, truth = protocol.load_test_set()
    want = protocol.fingerprint(seed)
    if manifest.get("fingerprint") != want:
        raise RuntimeError(
            f"the test set on disk was built for configuration "
            f"{manifest.get('fingerprint')}, this scoring expects {want}. Rebuild it "
            f"and rescore every run scored against the old one.")
    return manifest, props, truth, manifest["specs"]


def truth_stack(truth):
    return np.stack([np.asarray(truth[key], dtype=np.float64) for key in CHANNELS],
                    axis=1)


def predict(mech, props, coeffs, label="predict_profiles"):
    start = time.time()
    raw = timed(label, len(props), mech.predict_profiles, PREDICT_TIMEOUT, props,
                list(coeffs))
    pred = np.asarray(raw, dtype=np.float64)
    want = (len(props), len(CHANNELS), len(props[0]["reported"]["rho_norm"]))
    if pred.shape != want:
        raise RuntimeError(f"predict_profiles returned shape {pred.shape}, the task "
                           f"states {want}")
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("predict_profiles returned a non-finite value")
    return pred, time.time() - start


def predict_transport(mech, props, coeffs, profiles, label="transport_coefficients"):
    start = time.time()
    raw = timed(label, len(props), mech.transport_coefficients, PREDICT_TIMEOUT, props,
                list(coeffs), np.asarray(profiles, dtype=np.float64))
    out = np.asarray(raw, dtype=np.float64)
    want = (len(props), len(protocol.COEFFICIENTS),
            len(props[0]["reported"]["rho_face_norm"]))
    if out.shape != want:
        raise RuntimeError(f"transport_coefficients returned shape {out.shape}, the "
                           f"task states {want}")
    if not np.all(np.isfinite(out)):
        raise RuntimeError("transport_coefficients returned a non-finite value")
    return out, time.time() - start


def transport_signs(mech, coeffs, props, profiles):
    seconds = []

    def read(given):
        out, took = predict_transport(mech, props, coeffs, given)
        seconds.append(took)
        return out, given

    signs, _ = protocol.sensitivity_signs(read, profiles,
                                          props[0]["reported"]["rho_norm"])
    return signs, max(seconds)


def score_submitted(mech, coeffs, props, truth, specs):
    fit_sel, score_sel = protocol.fit_split(specs)
    fit_props, fit_truth, fit_specs = protocol.subset(props, truth, specs, fit_sel)
    score_props, score_truth, score_specs = protocol.subset(props, truth, specs,
                                                            score_sel)
    start = time.time()
    fitted = [float(c) for c in timed(
        "fit_coeffs", len(fit_props), mech.fit_coeffs, FIT_TIMEOUT,
        fit_props, truth_stack(fit_truth), list(coeffs))]
    fit_seconds = time.time() - start
    pred, seconds = predict(mech, score_props, fitted)
    block = protocol.measure(pred, truth_stack(score_truth), score_truth, score_specs)
    signs = None
    try:
        signs, transport_seconds = transport_signs(mech, fitted, score_props,
                                                   truth_stack(score_truth))
        block["transport_seconds_per_call"] = round(transport_seconds, 2)
        block["transport_within_stated_limit"] = bool(
            transport_seconds <= STATED_TRANSPORT_S)
    except Exception as exc:
        block["transport_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[eval] transport_coefficients could not be read "
              f"({block['transport_error']}); the sign criteria score 0")
    block.update(protocol.sign_scores(signs, score_truth))
    block["fit_seconds"] = round(fit_seconds, 2)
    block["fit_within_stated_limit"] = bool(fit_seconds <= STATED_FIT_S)
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_stated_limit"] = bool(seconds <= STATED_PREDICT_S)
    block["n_coeffs"] = len(coeffs)
    block["refit_coeffs"] = [round(c, 8) for c in fitted]
    block["n_fit_settings"] = len(fit_props)
    block["n_scored_settings"] = len(score_props)
    return block


def independence_check(mech, coeffs, props, truth, specs, row=0):
    fit_sel, score_sel = protocol.fit_split(specs)
    score_props, _, _ = protocol.subset(props, truth, specs, score_sel)
    whole, _ = predict(mech, score_props, coeffs, "predict_profiles (batch)")
    alone, _ = predict(mech, score_props[row:row + 1], coeffs,
                       "predict_profiles (one setting)")
    return bool(np.allclose(whole[row], alone[0]))


def write_report(output, report, started):
    scores = report.get("pa_scores") or {}
    report["predictive_accuracy"] = (round(sum(scores.values()) / len(scores), 4)
                                     if scores else 0.0)
    report["predictive_accuracy_total"] = len(protocol.PA_IDS)
    report["wall_clock_seconds"] = round(time.time() - started, 1)
    path = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("=" * 72)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")
    sub = report.get("submitted") or {}

    def cell(value):
        return f"{value:>13.4f}" if isinstance(value, (int, float)) else f"{'-':>13}"

    print(f"  {'':34}{'measured':>13}{'score':>13}")
    for pid, key in protocol.PA_MEASURES:
        print(f"  {pid + ' ' + key:34}{cell(sub.get(key))}{cell(scores.get(pid))}")
    print(f"  {'predictive accuracy':34}{'':>13}{cell(report['predictive_accuracy'])}")
    print()
    print(f"[eval] predictive accuracy: {json.dumps(scores)} -> "
          f"{report['predictive_accuracy']}")
    print(f"[eval] results: {path}")
    print("=" * 72)


def main(mechanism, output, seed=0, predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s, fit {FIT_TIMEOUT:.0f}s", flush=True)
    started = time.time()
    mech_path = str(mechanism)
    report = {"problem": PROBLEM, "seed": seed,
              "pa_scores": {pid: 0.0 for pid in protocol.PA_IDS}, "problems": []}
    if not os.path.isfile(mech_path):
        report["error"] = f"{mech_path} does not exist"
        write_report(output, report, started)
        return

    manifest, props, truth, specs = read_test_set(seed)
    report["test_set"] = {k: v for k, v in manifest.items() if k != "specs"}

    try:
        mech, blocking, problems = load_mechanism(mech_path)
    except Exception:
        mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
    report["problems"] = problems
    if blocking:
        report["error"] = "; ".join(blocking)
        write_report(output, report, started)
        return

    try:
        coeffs = [float(c) for c in mech.COEFFS]
    except (TypeError, ValueError) as exc:
        report["error"] = f"COEFFS is not a sequence of numbers: {exc}"
        write_report(output, report, started)
        return
    try:
        report["submitted"] = score_submitted(mech, coeffs, props, truth, specs)
    except Exception as exc:
        report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["setting_independent"] = independence_check(mech, coeffs, props, truth,
                                                           specs)
    except Exception as exc:
        report["setting_independent"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.pa_scores(report["submitted"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
