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

import equilibrium

PROBLEM = "nt_core_only"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
T_FINAL = 1.2
N_RHO = 25
NUMERICS = {"max_dt": 0.1, "chi_timestep_prefactor": 200, "dt_reduction_factor": 3}
PROTOCOL_VERSION = 2
MACHINE = dict(
    R_major=1.67, a_minor=0.6, B_0=2.0, Ip=0.9e6, elongation=1.7,
    impurity="C", Z_eff=2.0, n_line=4.0e19,
    T_bc=0.4, rho_bc=0.9, T_sep=0.1,
    P_aux=8.0e6, electron_heat_fraction=0.5, aux_location=0.0, aux_width=0.3,
)
TRANSPORT = {"model_name": "tglfnn-ukaea", "chi_min": 0.05, "chi_max": 30.0,
             "D_e_min": 0.05, "apply_inner_patch": True, "rho_inner": 0.1,
             "chi_i_inner": 1.0, "chi_e_inner": 1.0, "D_e_inner": 0.2,
             "V_e_inner": 0.0}
SCAN_DELTAS = [round(float(d), 1) for d in np.linspace(-0.5, 0.5, 11)]
PAIR_DELTAS = (0.2, 0.3, 0.4, 0.5)
VARIANT_DELTA = 0.4
ELONGATION_VARIANTS = (1.4, 2.0)
SIZE_VARIANTS = (1.5, 2.0)
OPERATING_AXES = {
    "P_aux": (4.0e6, 12.0e6),
    "n_line": (2.5e19, 6.0e19),
    "Z_eff": (1.5, 2.5),
    "electron_heat_fraction": (0.3, 0.7),
}
DENSITY_SCAN = (2.5e19, 3.5e19, 4.5e19, 6.0e19)
HEATING_PAIR = (4.0e6, 12.0e6)
FAMILIES = {"mirrored_pair": 16, "triangularity_scan": 3, "density_scan": 6,
            "heating_pair": 8, "elongation_variant": 6, "size_variant": 4}
PA_IDS = [f"PA{i}" for i in range(1, 8)]
RATIO_RHO_MIN = 0.5
CHI_RHO_MIN = 0.2
E_KEV = 1.602176634e-16
CHANNELS = ("T_i", "T_e", "chi_i", "chi_e")


def fingerprint(seed):
    payload = json.dumps({"seed": seed, "t_final": T_FINAL, "n_rho": N_RHO,
                          "families": FAMILIES, "axes": OPERATING_AXES,
                          "density_scan": DENSITY_SCAN, "heating_pair": HEATING_PAIR,
                          "scan_deltas": SCAN_DELTAS, "pair_deltas": PAIR_DELTAS,
                          "variant_delta": VARIANT_DELTA,
                          "elongation_variants": ELONGATION_VARIANTS,
                          "size_variants": SIZE_VARIANTS, "machine": MACHINE,
                          "transport": TRANSPORT, "numerics": NUMERICS,
                          "a_profile": equilibrium.A_PROFILE,
                          "version": PROTOCOL_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def confinement_time(props, t_i, t_e):
    rho = np.asarray(props["rho"], dtype=float)
    inside = rho <= float(props["rho_bc"]) + 1e-9
    d_volume = np.gradient(np.asarray(props["volume"], dtype=float))
    n_e = np.asarray(props["n_e"], dtype=float)
    n_i = n_e * float(props["n_i_over_n_e"])
    pressure = 1.5 * E_KEV * (n_e * np.asarray(t_e, dtype=float)
                              + n_i * np.asarray(t_i, dtype=float))
    heating = np.asarray(props["p_ext_i"], dtype=float) + np.asarray(props["p_ext_e"],
                                                                     dtype=float)
    stored = float(np.sum((pressure * d_volume)[inside]))
    delivered = float(np.sum((heating * d_volume)[inside]))
    return stored / max(delivered, 1e-30)


def pooled_r2(pred, truth):
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 2:
        return None
    residual = np.sum((pred[ok] - truth[ok]) ** 2)
    total = np.sum((truth[ok] - truth[ok].mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else None


def profile_r2(pred, truth, inside, chi_window):
    scores = []
    for channel in range(pred.shape[1]):
        window = inside if channel < 2 else chi_window
        p, t = pred[:, channel, :][:, window], truth[:, channel, :][:, window]
        scale = float(np.std(t)) or 1.0
        scores.append(pooled_r2(p / scale, t / scale))
    valid = [s for s in scores if s is not None]
    return float(np.mean(valid)) if valid else None


def log_tau(values):
    return np.log(np.maximum(np.asarray(values, dtype=float), 1e-12))


def predicted_tau(pred, props):
    return np.asarray([confinement_time(setting, pred[i, 0, :], pred[i, 1, :])
                       for i, setting in enumerate(props)], dtype=float)


def mirrored_pairs(specs):
    groups = {}
    for spec in specs:
        if spec.get("pair_key") and spec["family"] != "heating_pair":
            groups.setdefault(spec["pair_key"], []).append(spec)
    pairs = {}
    for key, members in groups.items():
        if len(members) != 2:
            continue
        reversed_member = min(members, key=lambda s: s["shape"]["delta"])
        d_member = max(members, key=lambda s: s["shape"]["delta"])
        if reversed_member["shape"]["delta"] < 0 < d_member["shape"]["delta"]:
            pairs[key] = (reversed_member["index"], d_member["index"])
    return pairs


def pair_differences(log_values, pairs):
    return np.array([log_values[nt] - log_values[pt] for nt, pt in pairs.values()])


def triangularity_scan_changes(specs, log_values):
    scans = {}
    for spec in specs:
        if spec["family"] == "triangularity_scan":
            scans.setdefault(spec["scan_key"], []).append(
                (spec["shape"]["delta"], spec["index"]))
    out = []
    for members in scans.values():
        zero = [i for d, i in members if abs(d) < 1e-9]
        if not zero:
            continue
        out += [log_values[i] - log_values[zero[0]] for d, i in members if abs(d) >= 1e-9]
    return np.array(out)


def ratio_r2(pred, truth_stack, props, pairs):
    if not pairs:
        return None
    rho = np.asarray(props[0]["rho"], dtype=float)
    window = (rho >= RATIO_RHO_MIN - 1e-9) & (rho <= float(props[0]["rho_bc"]) + 1e-9)
    scores = []
    for channel in (2, 3):
        p_ratio, t_ratio = [], []
        for nt, pt in pairs.values():
            p_ratio.append(np.log(np.maximum(pred[nt, channel, window], 1e-6)
                                  / np.maximum(pred[pt, channel, window], 1e-6)))
            t_ratio.append(np.log(np.maximum(truth_stack[nt, channel, window], 1e-6)
                                  / np.maximum(truth_stack[pt, channel, window], 1e-6)))
        scores.append(pooled_r2(np.concatenate(p_ratio), np.concatenate(t_ratio)))
    valid = [s for s in scores if s is not None]
    return float(np.mean(valid)) if valid else None


def measure(pred, truth_stack, truth, props, specs):
    rho = np.asarray(props[0]["rho"], dtype=float)
    inside = rho <= float(props[0]["rho_bc"]) + 1e-9
    chi_window = inside & (rho >= CHI_RHO_MIN - 1e-9)
    log_pred = log_tau(predicted_tau(pred, props))
    log_truth = log_tau([confinement_time(setting, truth_stack[i, 0, :],
                                          truth_stack[i, 1, :])
                         for i, setting in enumerate(props)])
    pairs = mirrored_pairs(specs)
    variants = [s["index"] for s in specs
                if s["family"] in ("elongation_variant", "size_variant")]
    heated = [s["index"] for s in specs if s["family"] == "heating_pair"]
    return {
        "tau_r2": pooled_r2(log_pred, log_truth),
        "pair_r2": pooled_r2(pair_differences(log_pred, pairs),
                             pair_differences(log_truth, pairs)) if pairs else None,
        "ratio_r2": ratio_r2(pred, truth_stack, props, pairs),
        "triangularity_scan_r2": pooled_r2(triangularity_scan_changes(specs, log_pred),
                                           triangularity_scan_changes(specs, log_truth)),
        "heating_r2": pooled_r2(log_pred[heated], log_truth[heated]) if heated else None,
        "variant_r2": pooled_r2(log_pred[variants], log_truth[variants])
        if variants else None,
        "profile_r2": profile_r2(pred, truth_stack, inside, chi_window),
        "n_pairs": len(pairs),
    }


PA_MEASURES = (("PA1", "tau_r2"), ("PA2", "pair_r2"), ("PA3", "ratio_r2"),
               ("PA4", "triangularity_scan_r2"), ("PA5", "heating_r2"),
               ("PA6", "variant_r2"), ("PA7", "profile_r2"))


def normalized(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def pa_scores(block):
    return {pid: round(normalized((block or {}).get(key)), 4) for pid, key in PA_MEASURES}


def fit_split(specs):
    fit = np.array([(s["family"] == "mirrored_pair" and s["op"] % 2 == 0)
                    or (s["family"] == "triangularity_scan" and s["op"] == 0)
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
    props = [{} for _ in range(n)]
    truth = {}
    for key in archive.files:
        parts = key.split("|")
        if parts[0] == "props":
            props[int(parts[1])][parts[2]] = archive[key]
        else:
            truth[parts[1]] = archive[key]
    return manifest, props, truth

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_profiles", "fit_coeffs")
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
STATED_PREDICT_S = 120.0
STATED_FIT_S = 600.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
CHANNELS = protocol.CHANNELS


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
    want = (len(props), len(CHANNELS), len(props[0]["rho"]))
    if pred.shape != want:
        raise RuntimeError(f"predict_profiles returned shape {pred.shape}, the task "
                           f"states {want}")
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("predict_profiles returned a non-finite value")
    return pred, time.time() - start


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
    block = protocol.measure(pred, truth_stack(score_truth), score_truth, score_props,
                             score_specs)
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

    print(f"  {'':34}{'R^2':>13}{'score':>13}")
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
        declared = [str(name) for name in mech.PROPERTIES]
    except TypeError:
        declared = []
        problems.append("PROPERTIES is not a sequence of names")
    unknown = sorted(set(declared) - set(props[0]))
    if unknown:
        problems.append(f"PROPERTIES names entries props does not carry: {unknown}")
    report["properties"] = {"declared": declared, "unknown": unknown,
                            "available": sorted(props[0])}

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
