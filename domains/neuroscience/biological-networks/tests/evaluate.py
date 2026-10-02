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

PROBLEM = "biological_networks"

SENSORY = ["AWCL", "AWCR"]
FIRST_LAYER = ["AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR"]
INTEGRATOR = ["RIAL", "RIAR"]
COMMAND = ["AVAL", "AVAR", "AVBL", "AVBR"]
HEAD_MOTOR = ["SMDDL", "SMDDR", "SMDVL", "SMDVR", "RIVL", "RIVR"]
PANEL = SENSORY + FIRST_LAYER + INTEGRATOR + COMMAND + HEAD_MOTOR
REORIENTATION_PLUS = ["AVAL", "AVAR"]
REORIENTATION_MINUS = ["AVBL", "AVBR"]
STEERING_PLUS = ["RIAL"]
STEERING_MINUS = ["RIAR"]
READOUT_TAU_MS = 200.0
CALIBRATE_SCRIPT = f"scripts/neuroscience/{PROBLEM}/run_calibrate_test_set.sh"
PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
CALIBRATION_FILE = TEST_SET_DIR / "calibration.json"
SIGNALS = ("reorientation", "steering")
PA_MEASURES = (("PA1", "pooled"), ("PA2", "per_setting_median"),
               ("PA3", "learning_effect"), ("PA4", "unseen_family"))
PA_IDS = tuple(pid for pid, _ in PA_MEASURES)
N_STATES = 3
N_PROTOCOLS = 32
N_UNSEEN_PROTOCOLS = 10
N_SETTINGS = N_PROTOCOLS * N_STATES
DURATION_MS = 4000.0
DT_MS = 0.05
SAVE_EVERY_MS = 2.0
PARAMETER_SET = "C1"
FACTOR_EARLY = 0.2
FACTOR_LATE = 0.2


def fingerprint(seed):
    payload = json.dumps({
        "n_settings": N_SETTINGS, "n_states": N_STATES, "seed": seed,
        "n_protocols": N_PROTOCOLS, "n_unseen_protocols": N_UNSEEN_PROTOCOLS,
        "duration_ms": DURATION_MS, "dt_ms": DT_MS, "save_every_ms": SAVE_EVERY_MS,
        "parameter_set": PARAMETER_SET, "factor_early": FACTOR_EARLY,
        "factor_late": FACTOR_LATE, "protocol_version": 5,
        "readout_tau_ms": READOUT_TAU_MS,
        "panel": PANEL, "steering_cells": [STEERING_PLUS, STEERING_MINUS],
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


def per_setting_median_r2(pred, truth):
    if not usable(pred, truth):
        return None
    out = []
    for i in range(truth.shape[0]):
        denom = float(np.var(truth[i]))
        if denom > 0:
            out.append(1.0 - float(np.mean((pred[i] - truth[i]) ** 2)) / denom)
    return float(np.median(out)) if out else None


def learning_effect(pred, truth, states, protocol):
    if not usable(pred, truth):
        return None
    got, want = [], []
    for p in np.unique(protocol):
        rows = np.flatnonzero(protocol == p)
        base = rows[states[rows] == 0]
        if base.size == 0:
            continue
        for row in rows:
            if states[row] == 0:
                continue
            got.append(pred[row] - pred[base[0]])
            want.append(truth[row] - truth[base[0]])
    if not got:
        return None
    return pooled_r2(np.stack(got), np.stack(want))


def measure(pred, targets, states, protocol, unseen):
    block = {}
    for key, fn in (("pooled", lambda p, y: pooled_r2(p, y)),
                    ("per_setting_median", lambda p, y: per_setting_median_r2(p, y))):
        block[key] = {s: fn(pred[s], targets[s]) for s in SIGNALS}
    block["learning_effect"] = {
        s: learning_effect(pred[s], targets[s], states, protocol) for s in SIGNALS}
    block["unseen_family"] = {
        s: pooled_r2(pred[s][unseen], targets[s][unseen]) for s in SIGNALS}
    block["seen_family"] = {
        s: pooled_r2(pred[s][~unseen], targets[s][~unseen]) for s in SIGNALS}
    return block


def normalized(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def pa_scores(block):
    return {pid: round(float(np.mean([normalized((block.get(key) or {}).get(s))
                                      for s in SIGNALS])), 4)
            for pid, key in PA_MEASURES}


def fit_split(protocol):
    keep = np.unique(protocol)[::2]
    fit = np.isin(protocol, keep)
    return fit, ~fit


def subset(props, sel):
    out = dict(props)
    out["stim"] = props["stim"][sel]
    out["learning_state"] = props["learning_state"][sel]
    return out


def load_test_set():
    data = np.load(TEST_SET_FILE)
    props = {"t": np.asarray(data["t"], dtype=np.float64),
             "stim": np.asarray(data["stim"], dtype=np.float64),
             "stim_cells": list(SENSORY),
             "learning_state": np.asarray(data["learning_state"], dtype=np.int64)}
    targets = {s: np.asarray(data[s], dtype=np.float64) for s in SIGNALS}
    states = props["learning_state"]
    protocol = np.asarray(data["protocol"], dtype=np.int64)
    unseen = np.asarray(data["unseen"], dtype=bool)
    return props, targets, states, protocol, unseen


def load_calibration():
    if not CALIBRATION_FILE.is_file():
        return None
    with open(CALIBRATION_FILE) as f:
        report = json.load(f)
    human = report.get("published_mechanism_refit_half") or {}
    return {"human_reference": human,
            "human_reference_pa": pa_scores(human) if human else None,
            "source": str(CALIBRATION_FILE)}

protocol = sys.modules[__name__]


REQUIRED_MEMBERS = ("INPUTS", "COEFFS", "predict_reorientation", "predict_steering",
                    "fit_coeffs")
ALLOWED_IMPORTS = ("numpy", "scipy")
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0


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
    props, targets, states, protocol_of, unseen = protocol.load_test_set()
    return props, targets, states, protocol_of, unseen, manifest


def predict(mech, props, coeffs):
    start = time.time()
    n = int(np.asarray(props["stim"]).shape[0])
    pred = {
        "reorientation": np.asarray(
            timed("predict_reorientation", n, mech.predict_reorientation,
                  PREDICT_TIMEOUT, props, list(coeffs)), dtype=np.float64),
        "steering": np.asarray(
            timed("predict_steering", n, mech.predict_steering,
                  PREDICT_TIMEOUT, props, list(coeffs)), dtype=np.float64),
    }
    return pred, time.time() - start


def score_submitted(mech, coeffs, props, targets, states, protocol_of, unseen):
    fit_sel, score_sel = protocol.fit_split(protocol_of)
    start = time.time()
    fitted = [float(c) for c in timed(
        "fit_coeffs", int(fit_sel.sum()), mech.fit_coeffs, FIT_TIMEOUT,
        protocol.subset(props, fit_sel),
        {s: targets[s][fit_sel] for s in protocol.SIGNALS}, list(coeffs))]
    fit_seconds = time.time() - start
    scored_props = protocol.subset(props, score_sel)
    pred, seconds = predict(mech, scored_props, fitted)
    block = protocol.measure(pred, {s: targets[s][score_sel] for s in protocol.SIGNALS},
                             states[score_sel], protocol_of[score_sel],
                             unseen[score_sel])
    block["fit_seconds"] = round(fit_seconds, 2)
    block["fit_within_limit"] = bool(fit_seconds <= FIT_TIMEOUT)
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_limit"] = bool(seconds <= PREDICT_TIMEOUT)
    block["n_coeffs"] = len(coeffs)
    block["n_fit_settings"] = int(fit_sel.sum())
    block["n_scored_settings"] = int(score_sel.sum())
    return block


def causal_check(mech, coeffs, props):
    half = props["stim"].shape[2] // 2
    altered = dict(props)
    stim = np.array(props["stim"], dtype=np.float64)
    stim[:, :, half:] += 50.0
    altered["stim"] = stim
    base, _ = predict(mech, props, coeffs)
    other, _ = predict(mech, altered, coeffs)
    return {s: bool(np.allclose(base[s][:, :half], other[s][:, :half]))
            for s in protocol.SIGNALS}


def independence_check(mech, coeffs, props, row=0):
    single = protocol.subset(props, np.array([row]))
    whole, _ = predict(mech, props, coeffs)
    alone, _ = predict(mech, single, coeffs)
    return {s: bool(np.allclose(whole[s][row], alone[s][0]))
            for s in protocol.SIGNALS}


def write_report(output, report, started):
    scores = report.get("pa_scores") or {}
    report["predictive_accuracy"] = (round(sum(scores.values()) / len(scores), 4)
                                     if scores else 0.0)
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
    ref = (report.get("reference") or {}).get("human_reference") or {}
    ref_pa = (report.get("reference") or {}).get("human_reference_pa") or {}
    sub = report.get("submitted") or {}

    def cell(value):
        return f"{value:>13.4f}" if isinstance(value, (int, float)) else f"{'-':>13}"

    print(f"  {'':34}{'submission':>13}{'human ref':>13}")
    for pid, key in protocol.PA_MEASURES:
        for signal in protocol.SIGNALS:
            print(f"  {pid + ' ' + key + ' ' + signal:34}"
                  f"{cell((sub.get(key) or {}).get(signal))}"
                  f"{cell((ref.get(key) or {}).get(signal))}")
    print(f"  {'':34}{'-' * 26}")
    for pid, _ in protocol.PA_MEASURES:
        print(f"  {pid + ' score':34}{cell(scores.get(pid))}"
              f"{cell(ref_pa.get(pid))}")
    print(f"  {'predictive accuracy':34}{cell(report['predictive_accuracy'])}"
          f"{cell(round(sum(ref_pa.values()) / len(ref_pa), 4) if ref_pa else None)}")
    print()
    print(f"[eval] predictive accuracy: {json.dumps(scores)} -> "
          f"{report['predictive_accuracy']}")
    if ref_pa:
        print(f"[eval] human reference    : {json.dumps(ref_pa)} -> "
              f"{sum(ref_pa.values()) / len(ref_pa):.4f}   (context, never a level to clear)")
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

    props, targets, states, protocol_of, unseen, manifest = read_test_set(seed)
    report["test_set"] = manifest
    report["reference"] = protocol.load_calibration()
    if report["reference"] is None:
        print(f"[eval] WARNING: no calibration at {protocol.CALIBRATION_FILE}; the "
              f"human reference is reported as missing; run "
              f"{protocol.CALIBRATE_SCRIPT} first")

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
    try:
        report["submitted"] = score_submitted(mech, coeffs, props, targets, states,
                                              protocol_of, unseen)
    except Exception as exc:
        report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
    for name, fn in (("causal", causal_check), ("setting_independent",
                                                independence_check)):
        try:
            report[name] = fn(mech, coeffs, props)
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.pa_scores(report["submitted"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
