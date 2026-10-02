import ast
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import fire
import numpy as np

PROBLEM = "navigation_goals"

SENSORY = ["AWCL", "AWCR"]
FIRST_LAYER = ["AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR"]
INTEGRATOR = ["RIAL", "RIAR"]
HUB = ["RIML", "RIMR", "AVEL", "AVER", "RIBL", "RIBR"]
COMMAND = ["AVAL", "AVAR", "AVBL", "AVBR"]
HEAD_MOTOR = ["SMDDL", "SMDDR", "SMDVL", "SMDVR", "RIVL", "RIVR"]
PANEL = SENSORY + FIRST_LAYER + INTEGRATOR + HUB + COMMAND + HEAD_MOTOR
REORIENTATION_PLUS = ["AVAL", "AVAR"]
REORIENTATION_MINUS = ["AVBL", "AVBR"]
STEERING_PLUS = ["RIAL"]
STEERING_MINUS = ["RIAR"]
READOUT_TAU_MS = 200.0
PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
CALIBRATION_FILE = TEST_SET_DIR / "calibration.json"
SIGNALS = ("reorientation", "steering")
PA_IDS = ("PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7", "PA8")
N_SETTINGS = 100
N_UNSEEN = 33
DURATION_MS = 9000.0
DT_MS = 0.05
SAVE_EVERY_MS = 5.0
PARAMETER_SET = "C1"
MIN_PA = 2.5
MAX_PA = 6.0
FLAT_NAME = "flat_drive"
SLOW_NAME = "slow_drive"
SLOW_TAU_MS = 3000.0
SLOW_COMPONENT_TAU_MS = 2000.0
PERSISTENCE_FACTOR = 2.0
BELOW_ZERO_FRACTION = 0.30


def fingerprint(seed):
    payload = json.dumps({
        "n_settings": N_SETTINGS, "n_unseen": N_UNSEEN, "seed": seed,
        "duration_ms": DURATION_MS, "dt_ms": DT_MS, "save_every_ms": SAVE_EVERY_MS,
        "parameter_set": PARAMETER_SET, "protocol_version": 6,
        "readout_tau_ms": READOUT_TAU_MS, "panel": PANEL,
        "min_pa": MIN_PA, "max_pa": MAX_PA,
        "steering_cells": [STEERING_PLUS, STEERING_MINUS],
        "reorientation_cells": [REORIENTATION_PLUS, REORIENTATION_MINUS],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def usable(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    return pred.shape == truth.shape and bool(np.isfinite(pred).all())


def pooled_r2(pred, truth):
    if not usable(pred, truth):
        return None
    denom = float(np.var(truth))
    if denom <= 0:
        return None
    return float(1.0 - np.mean((np.asarray(pred, dtype=np.float64) - truth) ** 2)
                 / denom)


def per_setting_r2(pred, truth):
    if not usable(pred, truth):
        return None
    out = []
    for i in range(truth.shape[0]):
        denom = float(np.var(truth[i]))
        if denom > 0:
            out.append(1.0 - float(np.mean((pred[i] - truth[i]) ** 2)) / denom)
    return out or None


def per_setting_median_r2(pred, truth):
    values = per_setting_r2(pred, truth)
    return float(np.median(values)) if values else None


def below_zero_fraction(pred, truth):
    values = per_setting_r2(pred, truth)
    if not values:
        return None
    return float(np.mean(np.asarray(values) < 0.0))


def onepole(x, tau_ms, dt):
    alpha = float(np.exp(-dt / max(tau_ms, dt)))
    out = np.empty_like(np.asarray(x, dtype=np.float64))
    run = np.zeros(out.shape[:-1])
    x = np.asarray(x, dtype=np.float64)
    for k in range(x.shape[-1]):
        run = alpha * run + (1.0 - alpha) * x[..., k]
        out[..., k] = run
    return out


def step_ms(t):
    t = np.asarray(t, dtype=np.float64).ravel()
    return float(np.median(np.diff(t))) if t.size > 1 else 1.0


def slow_component_r2(pred, truth, t):
    if not usable(pred, truth):
        return None
    dt = step_ms(t)
    return pooled_r2(onepole(pred, SLOW_COMPONENT_TAU_MS, dt),
                     onepole(truth, SLOW_COMPONENT_TAU_MS, dt))


def autocorr_halflife(x, t):
    x = np.asarray(x, dtype=np.float64)
    dt = step_ms(t)
    out = []
    for row in x:
        row = row - row.mean()
        denom = float(np.dot(row, row))
        if denom <= 0:
            continue
        n = row.size
        span = max(int(n // 2), 2)
        corr = np.correlate(row, row, mode="full")[n - 1: n - 1 + span] / denom
        below = np.flatnonzero(corr < 0.5)
        out.append(float(below[0]) * dt if below.size else float(span) * dt)
    return float(np.median(out)) if out else None


def withhold(props, ingredient):
    reduced = dict(props)
    stim = np.asarray(props["stim"], dtype=np.float64)
    if ingredient == "flat":
        reduced["stim"] = np.broadcast_to(stim.mean(axis=2, keepdims=True),
                                          stim.shape).copy()
    else:
        reduced["stim"] = onepole(stim, SLOW_TAU_MS, step_ms(props["t"]))
    return reduced


def predict_both(mech, props, coeffs):
    return {
        "reorientation": np.asarray(mech.predict_reorientation(props, list(coeffs)),
                                    dtype=np.float64),
        "steering": np.asarray(mech.predict_steering(props, list(coeffs)),
                               dtype=np.float64),
    }


def make_baselines(mech, coeffs, props, targets, unseen):
    out = {}
    for name, ingredient in ((FLAT_NAME, "flat"), (SLOW_NAME, "slow")):
        reduced = withhold(props, ingredient)
        fitted = [float(c) for c in mech.fit_coeffs(reduced, targets, list(coeffs))]
        pred = predict_both(mech, reduced, fitted)
        out[name] = block(pred, targets, props["t"], unseen)
        out[f"coeffs_{name}"] = fitted
    return out


def block(pred, targets, t, unseen):
    both_pred = np.concatenate([pred[s] for s in SIGNALS], axis=0)
    both_true = np.concatenate([targets[s] for s in SIGNALS], axis=0)
    both_unseen = np.concatenate([unseen for _ in SIGNALS], axis=0)
    out = {
        "pooled_both": pooled_r2(both_pred, both_true),
        "unseen_family_both": pooled_r2(both_pred[both_unseen],
                                        both_true[both_unseen]),
        "pooled": {s: pooled_r2(pred[s], targets[s]) for s in SIGNALS},
        "per_setting_median": {s: per_setting_median_r2(pred[s], targets[s])
                               for s in SIGNALS},
        "below_zero_fraction": {s: below_zero_fraction(pred[s], targets[s])
                                for s in SIGNALS},
        "unseen_family": {s: pooled_r2(pred[s][unseen], targets[s][unseen])
                          for s in SIGNALS},
        "seen_family": {s: pooled_r2(pred[s][~unseen], targets[s][~unseen])
                        for s in SIGNALS},
        "slow_reorientation": slow_component_r2(pred["reorientation"],
                                                targets["reorientation"], t),
        "persistence": {
            "predicted_halflife_ms": {
                "steps": autocorr_halflife(pred["reorientation"][~unseen], t),
                "rotating": autocorr_halflife(pred["reorientation"][unseen], t)},
            "recorded_halflife_ms": {
                "steps": autocorr_halflife(targets["reorientation"][~unseen], t),
                "rotating": autocorr_halflife(targets["reorientation"][unseen], t)},
        },
    }
    return out


def clip01(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def dig(block, path):
    out = block
    for key in path:
        out = (out or {}).get(key) if isinstance(out, dict) else None
    return out


def reduced_run_score(measured, baselines, path):
    value = dig(measured, path)
    if value is None or not np.isfinite(value):
        return 0.0
    floor = 0.0
    for name in (FLAT_NAME, SLOW_NAME):
        other = dig(baselines.get(name), path)
        if other is None or not np.isfinite(other):
            return 0.0
        floor = max(floor, float(other))
    if floor >= 1.0:
        return 0.0
    return clip01((float(value) - floor) / (1.0 - floor))


def per_setting_score(measured):
    median = measured.get("per_setting_median") or {}
    below = measured.get("below_zero_fraction") or {}
    out = []
    for signal in SIGNALS:
        share = below.get(signal)
        if share is None or not np.isfinite(share):
            return 0.0
        out.append(min(clip01(median.get(signal)),
                       clip01(1.0 - float(share) / BELOW_ZERO_FRACTION)))
    return min(out) if out else 0.0


def persistence_score(measured):
    persistence = measured.get("persistence") or {}
    got = persistence.get("predicted_halflife_ms") or {}
    want = persistence.get("recorded_halflife_ms") or {}
    out = []
    for family in ("steps", "rotating"):
        a, b = got.get(family), want.get(family)
        if a is None or b is None or a <= 0 or b <= 0:
            return 0.0
        out.append(clip01(1.0 - abs(np.log(float(a) / float(b)))
                          / np.log(PERSISTENCE_FACTOR)))
    return min(out) if out else 0.0


def pa_scores(measured, baselines, refit=None):
    measured = measured or {}
    baselines = baselines or {}
    out = {
        "PA1": reduced_run_score(measured, baselines, ("pooled_both",)),
        "PA2": reduced_run_score(measured, baselines, ("pooled", "reorientation")),
        "PA3": reduced_run_score(measured, baselines, ("pooled", "steering")),
        "PA4": per_setting_score(measured),
        "PA5": persistence_score(measured),
        "PA6": reduced_run_score(measured, baselines, ("unseen_family_both",)),
        "PA7": reduced_run_score(measured, baselines, ("slow_reorientation",)),
    }
    if refit is None:
        out["PA8"] = 0.0
    else:
        refit_measured, refit_baselines = refit
        out["PA8"] = reduced_run_score(refit_measured, refit_baselines or {},
                                       ("pooled_both",))
    return {pa: float(out[pa]) for pa in PA_IDS}


def subset(props, sel):
    out = dict(props)
    out["stim"] = np.asarray(props["stim"])[sel]
    return out


def load_test_set():
    data = np.load(TEST_SET_FILE)
    props = {"t": np.asarray(data["t"], dtype=np.float64),
             "stim": np.asarray(data["stim"], dtype=np.float64),
             "stim_cells": list(SENSORY)}
    targets = {s: np.asarray(data[s], dtype=np.float64) for s in SIGNALS}
    unseen = np.asarray(data["unseen"], dtype=bool)
    return props, targets, unseen


def load_calibration():
    if not CALIBRATION_FILE.is_file():
        return None
    with open(CALIBRATION_FILE) as f:
        report = json.load(f)
    return {"human_reference": report.get("published_mechanism"),
            "reduced_runs": report.get("reduced_runs"),
            "pa_scores": pa_scores(
                report.get("published_mechanism"), report.get("reduced_runs"),
                ((report.get("published_mechanism_refit_half"),
                  report.get("reduced_runs_refit_half"))
                 if report.get("published_mechanism_refit_half") else None)),
            "optimisation_ceiling": report.get("optimisation_ceiling"),
            "source": str(CALIBRATION_FILE)}

protocol = sys.modules[__name__]


REQUIRED_MEMBERS = ("INPUTS", "COEFFS", "predict_reorientation", "predict_steering",
                    "fit_coeffs")
ALLOWED_IMPORTS = ("numpy", "scipy")
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0


def call_with_timeout(fn, timeout, *args):
    out, error = [], []

    def run():
        try:
            out.append(fn(*args))
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} exceeded {timeout:.0f}s")
    if error:
        raise error[0]
    return out[0]


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
    scan = ImportScan()
    with open(path) as f:
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy: {bad}")
    return module, blocking, problems


def read_test_set(seed):
    with open(protocol.TEST_SET_MANIFEST) as f:
        manifest = json.load(f)
    want = protocol.fingerprint(seed)
    if manifest.get("fingerprint") != want:
        raise RuntimeError(f"the test set was built for configuration "
                           f"{manifest.get('fingerprint')}, this scoring expects {want}")
    props, targets, unseen = protocol.load_test_set()
    return props, targets, unseen, manifest


def predict(mech, props, coeffs):
    start = time.time()
    pred = call_with_timeout(protocol.predict_both, PREDICT_TIMEOUT, mech, props,
                             list(coeffs))
    return pred, time.time() - start


def score_submitted(mech, coeffs, props, targets, unseen):
    baselines = protocol.make_baselines(mech, coeffs, props, targets, unseen)
    pred, seconds = predict(mech, props, coeffs)
    block = protocol.block(pred, targets, props["t"], unseen)
    block["reduced_runs"] = {name: baselines[name]
                             for name in (protocol.FLAT_NAME, protocol.SLOW_NAME)}
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_limit"] = bool(seconds <= PREDICT_TIMEOUT)
    block["n_coeffs"] = len(coeffs)
    return block, baselines


def score_refit(mech, coeffs, props, targets, unseen):
    n = targets["reorientation"].shape[0]
    half = np.zeros(n, dtype=bool)
    half[::2] = True
    start = time.time()
    refit = [float(c) for c in call_with_timeout(
        mech.fit_coeffs, FIT_TIMEOUT, protocol.subset(props, half),
        {s: targets[s][half] for s in protocol.SIGNALS}, list(coeffs))]
    seconds = time.time() - start
    other = protocol.subset(props, ~half)
    other_targets = {s: targets[s][~half] for s in protocol.SIGNALS}
    baselines = protocol.make_baselines(mech, refit, other, other_targets,
                                        unseen[~half])
    pred, _ = predict(mech, other, refit)
    block = protocol.block(pred, other_targets, props["t"], unseen[~half])
    block["reduced_runs"] = {name: baselines[name]
                             for name in (protocol.FLAT_NAME, protocol.SLOW_NAME)}
    block["fit_seconds"] = round(seconds, 2)
    block["fit_within_limit"] = bool(seconds <= FIT_TIMEOUT)
    block["refit_coeffs"] = [round(c, 8) for c in refit]
    return block, baselines


def reference_context():
    return protocol.load_calibration()


def write_report(output, report, started):
    scores = report["pa_scores"]
    report["predictive_accuracy"] = round(
        float(np.mean([scores[pid] for pid in protocol.PA_IDS])), 4)
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
    context = report.get("reference") or {}
    ref = context.get("human_reference") or {}
    ceiling = context.get("optimisation_ceiling") or {}
    ref_scores = context.get("pa_scores") or {}
    sub = report.get("submitted") or {}
    refit = report.get("refit") or {}

    def at(block, key, signal):
        got = (block or {}).get(key)
        if isinstance(got, dict):
            return got.get(signal)
        return got

    rows = []
    for signal in protocol.SIGNALS:
        for key, label in (("pooled", "PA1-3 pooled R2"),
                           ("per_setting_median", "PA4 per-setting median"),
                           ("unseen_family", "PA6 second family")):
            rows.append((f"{label} {signal[:4]}", at(sub, key, signal),
                         at(ref, key, signal),
                         (ceiling.get(signal) or {}).get(key)))
    rows.append(("PA7 slow reorientation", sub.get("slow_reorientation"),
                 ref.get("slow_reorientation"), ceiling.get("slow_reorientation")))
    for signal in protocol.SIGNALS:
        rows.append((f"PA8 refit pooled {signal[:4]}", at(refit, "pooled", signal),
                     None, None))

    def cell(value):
        return f"{value:>11.4f}" if isinstance(value, (int, float)) else f"{'-':>11}"

    print(f"  {'':28}{'submission':>11}{'published':>11}{'optimisation':>13}")
    print(f"  {'':28}{'':>11}{'mechanism':>11}{'ceiling':>13}")
    for name, a, b, c in rows:
        print(f"  {name:28}{cell(a)}{cell(b)}{cell(c):>13}"[:66])
    print(f"  {'':28}{'-' * 35}")
    ref_total = (round(float(np.mean(list(ref_scores.values()))), 4)
                 if ref_scores else None)
    mine = f"{report['predictive_accuracy']:.4f}"
    theirs = f"{ref_total:.4f}" if ref_scores else "-"
    print(f"  {'predictive accuracy':28}{mine:>11}{theirs:>11}")
    print()
    rounded = {k: round(float(v), 4) for k, v in scores.items()}
    print(f"[eval] this submission : {json.dumps(rounded)}")
    if ref_scores:
        print(f"[eval] published mech. : "
              f"{json.dumps({k: round(float(v), 4) for k, v in ref_scores.items()})}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']:.4f}"
          f"  (mean of {report['predictive_accuracy_total']} criteria, each 0-1)"
          + (f"   (published mechanism on the same test set: {ref_total:.4f}"
             f", context only, never a level to clear)"
             if ref_scores else ""))
    print(f"[eval] results: {path}")
    print("=" * 72)


def main(mechanism, output, seed=0, predict_timeout=7200, fit_timeout=7200):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict_activity {PREDICT_TIMEOUT:.0f}s, "
          f"fit_coeffs {FIT_TIMEOUT:.0f}s", flush=True)
    started = time.time()
    mech_path = str(mechanism)
    report = {"problem": PROBLEM, "seed": seed,
              "pa_scores": {pid: 0.0 for pid in protocol.PA_IDS}, "problems": []}
    if not os.path.isfile(mech_path):
        report["error"] = f"{mech_path} does not exist"
        write_report(output, report, started)
        return

    props, targets, unseen, manifest = read_test_set(seed)
    report["test_set"] = manifest
    report["reference"] = reference_context()

    try:
        mech, blocking, problems = load_mechanism(mech_path)
    except Exception:
        mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
    report["problems"] = problems
    if blocking:
        report["error"] = "; ".join(blocking)
        write_report(output, report, started)
        return

    coeffs = [float(c) for c in mech.COEFFS]
    baselines = refit_baselines = None
    try:
        report["submitted"], baselines = score_submitted(mech, coeffs, props, targets,
                                                         unseen)
    except Exception as exc:
        report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["refit"], refit_baselines = score_refit(mech, coeffs, props, targets,
                                                       unseen)
    except Exception as exc:
        report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}

    if baselines is not None and "error" not in report["submitted"]:
        refit_pair = ((report["refit"], refit_baselines)
                      if refit_baselines is not None
                      and "error" not in report["refit"] else None)
        report["pa_scores"] = protocol.pa_scores(report["submitted"], baselines,
                                                 refit_pair)
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
