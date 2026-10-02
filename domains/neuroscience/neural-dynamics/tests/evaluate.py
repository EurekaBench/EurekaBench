import ast
import hashlib
import importlib.util
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
TEST_SET_FILE = HERE / "test_set" / "heldout.npz"
TEST_SET_MANIFEST = HERE / "test_set" / "manifest.json"
REFERENCE_REPORT = HERE / "reference_mechanism" / "reference_check.json"
PROBLEM = "neural_dynamics"

STATE_CELLS = {"forward": ["AVBL", "AVBR", "PVCL", "PVCR"],
               "backward": ["AVAL", "AVAR", "AVDL", "AVDR"],
               "turn": ["RIVL", "RIVR", "VC1", "VC2", "VC3"]}
RHYTHM_CELLS = [f"DA{i}" for i in range(1, 10)] + [f"VA{i}" for i in range(1, 13)]
PANEL_CORD = ([f"DB{i}" for i in range(1, 8)] + [f"VB{i}" for i in range(1, 12)]
              + [f"DD{i}" for i in range(1, 7)]
              + [f"VD{i}" for i in range(1, 14) if i != 11])
PANEL_OTHER = ([f"AS{i}" for i in range(1, 12)]
               + ["SMDVL", "SMDVR", "RMDDL", "RMEV", "AIBL", "AIBR", "DVA", "PVR"])
PANEL = PANEL_CORD + PANEL_OTHER
RESPONSE_FLOOR = 1e-8
N_SETTINGS = 100
DURATION_MS = 3000.0
DT_MS = 0.025
SAVE_EVERY_MS = 5.0
SETTLE_MS = 300.0
GUARD_MS = 150.0
PHASE_BINS = 24
PARAMETER_SET = "C1"
PROTOCOL_VERSION = 3
FAMILIES = {"standard": 60, "wide_range": 16, "repeated_state": 10,
            "rhythm_only": 4, "state_only": 4, "two_states": 6}
RANGES = {"standard": {"state_pa": (9.0, 16.0), "rhythm_pa": (1.0, 4.0),
                       "period_ms": (220.0, 420.0), "waves": (1.0, 2.2),
                       "epoch_ms": (700.0, 1500.0)},
          "wide": {"state_pa": (6.0, 20.0), "rhythm_pa": (0.5, 5.0),
                   "period_ms": (180.0, 520.0), "waves": (0.6, 3.0),
                   "epoch_ms": (500.0, 2000.0)}}
PER_NEURON_R2_FLOOR = 0.0
SIGN_RELATIVE_FLOOR = 0.10
PA_IDS = [f"PA{i}" for i in range(1, 9)]

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_activity", "fit_coeffs")
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
NO_BASELINES = {"r2_state_only": None, "r2_state_independent": None}


def driven_cells():
    return sorted(set(sum(STATE_CELLS.values(), []) + RHYTHM_CELLS))


def fingerprint(seed):
    payload = json.dumps({
        "n_settings": N_SETTINGS, "seed": seed, "duration_ms": DURATION_MS,
        "dt_ms": DT_MS, "save_every_ms": SAVE_EVERY_MS, "settle_ms": SETTLE_MS,
        "parameter_set": PARAMETER_SET, "protocol_version": PROTOCOL_VERSION,
        "families": FAMILIES, "ranges": RANGES, "state_cells": STATE_CELLS,
        "rhythm_cells": RHYTHM_CELLS, "panel": PANEL,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_test_set():
    d = np.load(TEST_SET_FILE, allow_pickle=False)
    props = {k: d[k].astype(np.float64)
             for k in ("t", "drive", "driver", "chem", "elec", "baseline", "period_ms")}
    return (props, d["truth"].astype(np.float64), json.loads(str(d["specs"])),
            props["chem"], props["elec"], props["baseline"])


def moving_average(x, window):
    w = max(int(window), 1)
    if w <= 1:
        return np.array(x, dtype=np.float64)
    pad = w // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), [(0, 0)] * (np.ndim(x) - 1)
                    + [(pad, w - 1 - pad)], mode="edge")
    csum = np.cumsum(padded, axis=-1)
    csum = np.concatenate([np.zeros(csum.shape[:-1] + (1,)), csum], axis=-1)
    return (csum[..., w:] - csum[..., :-w]) / w


def usable(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    return pred.shape == truth.shape and bool(np.all(np.isfinite(pred)))


def pooled_r2(pred, truth):
    if not usable(pred, truth):
        return None
    return float(1 - np.mean((pred - truth) ** 2) / max(np.var(truth), 1e-30))


def responds(truth, rows=None):
    sel = np.arange(truth.shape[1]) if rows is None else np.asarray(rows)
    return np.array([k for k in sel if truth[:, k, :].std() > RESPONSE_FLOOR])


def per_neuron_r2(pred, truth):
    if not usable(pred, truth):
        return None
    out = []
    for k in responds(truth):
        y = truth[:, k, :]
        out.append(float(1 - np.mean((pred[:, k, :] - y) ** 2) / max(np.var(y), 1e-30)))
    return out


def median_nrmse(pred, truth):
    if not usable(pred, truth):
        return None
    vals = []
    for i in range(truth.shape[0]):
        rmse = float(np.sqrt(np.mean((pred[i] - truth[i]) ** 2)))
        vals.append(rmse / max(float(np.std(truth[i])), 1e-30))
    return float(np.median(vals))


def fast_component(x, t, period_ms):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return np.asarray(x) - moving_average(x, round(period_ms / max(dt, 1e-9)))


def fast_r2(pred, truth, t, period_ms, rows=None):
    if not usable(pred, truth):
        return None
    num = den = 0.0
    values = []
    for i in range(truth.shape[0]):
        p = fast_component(pred[i], t, period_ms[i])
        y = fast_component(truth[i], t, period_ms[i])
        if rows is not None:
            p, y = p[rows], y[rows]
        num += float(np.sum((p - y) ** 2))
        den += float(np.sum((y - y.mean()) ** 2))
        values.append(float(np.var(y)))
    if den <= 0:
        return None
    return float(1 - num / den)


def state_masks(spec, t):
    settled = np.ones(len(t), dtype=bool)
    for e in spec["epochs"]:
        settled &= ~((t >= e["start_ms"]) & (t < e["start_ms"] + GUARD_MS))
    masks = {}
    for state in STATE_CELLS:
        m = np.zeros(len(t), dtype=bool)
        for e in spec["epochs"]:
            if e["states"] == [state]:
                m |= (t >= e["start_ms"]) & (t < e["end_ms"])
        masks[state] = m & settled
    return masks


def phase_amplitude(x, phase, mask):
    if mask.sum() < 3 * PHASE_BINS:
        return 0.0
    y = x[mask] - x[mask].mean()
    c = (np.cos(phase[mask]) * y).mean()
    s = (np.sin(phase[mask]) * y).mean()
    return float(2 * np.hypot(c, s))


def per_state_amplitudes(activity, spec, t):
    phase = np.mod(2 * np.pi * t / spec["period_ms"], 2 * np.pi)
    masks = state_masks(spec, t)
    return np.array([[phase_amplitude(activity[k], phase, masks["forward"]),
                      phase_amplitude(activity[k], phase, masks["backward"])]
                     for k in range(activity.shape[0])])


def eligible_settings(specs):
    out = []
    for i, s in enumerate(specs):
        if s["rhythm_pa"] <= 0:
            continue
        alone = {e["states"][0] for e in s["epochs"] if len(e["states"]) == 1}
        if {"forward", "backward"} <= alone:
            out.append(i)
    return out


def amplitude_agreement(pred, truth, specs, t, rows=None):
    sel = slice(None) if rows is None else rows
    idx = eligible_settings(specs)
    if not idx or not usable(pred, truth):
        return {"amplitude_r": None, "sign_fraction": None, "n_settings": len(idx),
                "n_pairs": 0}
    amp_true = np.concatenate([per_state_amplitudes(truth[i][sel], specs[i], t)
                               for i in idx])
    amp_pred = np.concatenate([per_state_amplitudes(pred[i][sel], specs[i], t)
                               for i in idx])
    ok = np.all(np.isfinite(amp_pred), axis=1) & np.all(np.isfinite(amp_true), axis=1)
    r = (float(np.corrcoef(amp_pred[ok].ravel(), amp_true[ok].ravel())[0, 1])
         if ok.sum() > 2 else None)
    diff_true = amp_true[:, 0] - amp_true[:, 1]
    diff_pred = amp_pred[:, 0] - amp_pred[:, 1]
    scale = np.maximum(amp_true.max(axis=1), 1e-30)
    clear = ok & (np.abs(diff_true) >= SIGN_RELATIVE_FLOOR * scale)
    sign = (float(np.mean(np.sign(diff_pred[clear]) == np.sign(diff_true[clear])))
            if clear.sum() else None)
    return {"amplitude_r": r, "sign_fraction": sign, "n_settings": len(idx),
            "n_pairs": int(clear.sum())}


def state_channels():
    named = set(sum(STATE_CELLS.values(), []))
    return np.array([i for i, c in enumerate(driven_cells()) if c in named])


def withhold(props, ingredient):
    dt = float(np.median(np.diff(props["t"]))) if len(props["t"]) > 1 else 1.0
    rows = state_channels()
    reduced = dict(props)
    for key in ("drive", "driver"):
        x = np.asarray(props[key], dtype=np.float64)
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            if ingredient == "rhythm":
                out[i] = moving_average(x[i], round(props["period_ms"][i]
                                                    / max(dt, 1e-9)))
            else:
                out[i] = x[i]
                out[i][rows, :] = x[i][rows, :].mean()
        reduced[key] = out
    return reduced


def make_baselines(mech, coeffs, props, truth, specs, t, predict=None, fit=None):
    run = predict or mech.predict_activity
    tune = fit or mech.fit_coeffs
    out = {}
    for name, ingredient in (("state_only", "rhythm"), ("state_independent", "state")):
        reduced = withhold(props, ingredient)
        fitted = [float(c) for c in tune(reduced, truth, list(coeffs))]
        pred = np.asarray(run(reduced, fitted), dtype=np.float64)
        out[name] = pred
        out[f"r2_{name}"] = pooled_r2(pred, truth)
    return out


def measure(pred, truth, props, specs, t, baselines, unseen_rows=None):
    r2 = pooled_r2(pred, truth)
    neuron = per_neuron_r2(pred, truth)
    b_rhythm = baselines["r2_state_only"]
    b_state = baselines["r2_state_independent"]
    have = r2 is not None and b_rhythm is not None and b_state is not None
    block = {
        "pooled_r2": r2,
        "per_neuron_r2_median": float(np.median(neuron)) if neuron else None,
        "per_neuron_r2_below_floor": (float(np.mean(np.array(neuron)
                                                    < PER_NEURON_R2_FLOOR))
                                      if neuron else None),
        "median_nrmse": median_nrmse(pred, truth),
        "fast_r2": fast_r2(pred, truth, t, props["period_ms"]),
        "beats_state_only": bool(have and r2 > b_rhythm),
        "beats_state_independent": bool(have and r2 > b_state),
        "baselines": {"state_only": b_rhythm, "state_independent": b_state},
        "margin_state_only": (r2 - b_rhythm) if have else None,
        "margin_state_independent": (r2 - b_state) if have else None,
    }
    block["baselines_fast"] = {
        name: (fast_r2(np.asarray(baselines[name], dtype=np.float64), truth, t,
                       props["period_ms"])
               if baselines.get(name) is not None else None)
        for name in ("state_only", "state_independent")}
    block.update(amplitude_agreement(pred, truth, specs, t))
    if unseen_rows is not None and usable(pred, truth):
        unseen_rows = responds(truth, unseen_rows)
        agree = amplitude_agreement(pred, truth, specs, t, unseen_rows)
        block["unseen"] = {"pooled_r2": pooled_r2(pred[:, unseen_rows, :],
                                                  truth[:, unseen_rows, :]),
                           "sign_fraction": agree["sign_fraction"],
                           "amplitude_r": agree["amplitude_r"],
                           "n_pairs": agree["n_pairs"],
                           "n_cells": int(len(unseen_rows))}
    return block


def clip01(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def reduced_run_score(value, references):
    if value is None or not np.isfinite(value):
        return 0.0
    floor = 0.0
    for other in references:
        if other is None or not np.isfinite(other):
            return 0.0
        floor = max(floor, float(other))
    if floor >= 1.0:
        return 0.0
    return clip01((float(value) - floor) / (1.0 - floor))


def per_neuron_score(block):
    median = block.get("per_neuron_r2_median")
    below = block.get("per_neuron_r2_below_floor")
    if median is None or below is None:
        return 0.0
    return min(clip01(median), clip01(1.0 - float(below)))


def correlation_score(value):
    return clip01(value)


def sign_score(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return clip01(2.0 * (float(value) - 0.5))


def margin_over_reduced(block):
    return min(clip01(block.get("margin_state_only")),
               clip01(block.get("margin_state_independent")))


def pa_scores(submitted, refit):
    sub = submitted or {}
    rer = refit or {}
    base = sub.get("baselines") or {}
    fast = sub.get("baselines_fast") or {}
    refit_base = rer.get("baselines") or {}
    unseen = sub.get("unseen") or {}
    return {
        "PA1": reduced_run_score(sub.get("pooled_r2"),
                                 (base.get("state_only"),
                                  base.get("state_independent"))),
        "PA2": per_neuron_score(sub),
        "PA3": reduced_run_score(sub.get("fast_r2"),
                                 (fast.get("state_only"),
                                  fast.get("state_independent"))),
        "PA4": margin_over_reduced(sub),
        "PA5": correlation_score(sub.get("amplitude_r")),
        "PA6": sign_score(sub.get("sign_fraction")),
        "PA7": correlation_score(unseen.get("amplitude_r")),
        "PA8": reduced_run_score(rer.get("pooled_r2"),
                                 (refit_base.get("state_only"),
                                  refit_base.get("state_independent"))),
    }


def subset(props, sel):
    out = dict(props)
    out["drive"] = props["drive"][sel]
    out["driver"] = props["driver"][sel]
    out["period_ms"] = props["period_ms"][sel]
    return out


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
    if hasattr(module, "COEFFS") and len(list(module.COEFFS)) > MAX_COEFFS:
        problems.append(f"COEFFS has {len(list(module.COEFFS))} entries, "
                        f"limit {MAX_COEFFS}")
    scan = ImportScan()
    with open(path) as f:
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy: {bad}")
    return module, blocking, problems


def read_test_set(seed):
    with open(TEST_SET_MANIFEST) as f:
        manifest = json.load(f)
    if manifest.get("fingerprint") != fingerprint(seed):
        raise RuntimeError(f"the test set was built for configuration {manifest.get('fingerprint')}, "
                           f"this scoring expects {fingerprint(seed)}")
    props, truth, specs = load_test_set()[:3]
    return props, truth, specs, manifest


def other_rows(truth):
    return responds(truth, [PANEL.index(cell) for cell in PANEL_OTHER])


def timed(label, fn, timeout, *args):
    n = len(np.asarray(args[0]["t"])) if args and isinstance(args[0], dict) else 0
    settings = np.asarray(args[0]["driver"]).shape[0] if n else 0
    print(f"[eval] {label}: {settings} settings, budget {timeout:.0f}s ...",
          flush=True)
    start = time.time()
    try:
        out = call_with_timeout(fn, timeout, *args)
    except Exception as exc:
        print(f"[eval] {label}: {type(exc).__name__} after "
              f"{time.time() - start:.1f}s", flush=True)
        raise
    print(f"[eval] {label}: done in {time.time() - start:.1f}s", flush=True)
    return out


def predict(mech, props, coeffs):
    start = time.time()
    pred = np.asarray(timed("predict_activity", mech.predict_activity,
                            PREDICT_TIMEOUT, props, list(coeffs)),
                      dtype=np.float64)
    return pred, time.time() - start


def bounded_predict(mech):
    def run(props, coeffs):
        return timed("predict_activity (withheld drive)", mech.predict_activity,
                     PREDICT_TIMEOUT, props, coeffs)
    return run


def bounded_fit(mech):
    def run(props, truth, coeffs):
        return timed("fit_coeffs (withheld drive)", mech.fit_coeffs,
                     FIT_TIMEOUT, props, truth, coeffs)
    return run


def score_submitted(mech, coeffs, props, truth, specs, t):
    pred, seconds = predict(mech, props, coeffs)
    try:
        baselines = make_baselines(mech, coeffs, props, truth, specs, t,
                                   predict=bounded_predict(mech),
                                   fit=bounded_fit(mech))
        failed = None
    except Exception as exc:
        baselines, failed = dict(NO_BASELINES), f"{type(exc).__name__}: {exc}"
        print(f"[eval] the withheld-ingredient runs failed ({failed}); only the "
              f"margins are lost")
    block = measure(pred, truth, props, specs, t, baselines,
                    other_rows(truth))
    if failed:
        block["baselines_error"] = failed
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_limit"] = bool(seconds <= PREDICT_TIMEOUT)
    block["n_coeffs"] = len(coeffs)
    return block


def score_refit(mech, coeffs, props, truth, specs, t):
    half = len(specs) // 2
    fit_sel, test_sel = np.arange(half), np.arange(half, len(specs))
    start = time.time()
    refit = [float(c) for c in timed(
        "fit_coeffs (refit half)", mech.fit_coeffs, FIT_TIMEOUT,
        subset(props, fit_sel), truth[fit_sel], list(coeffs))]
    seconds = time.time() - start
    test_props = subset(props, test_sel)
    test_specs = [specs[i] for i in test_sel]
    pred, _ = predict(mech, test_props, refit)
    try:
        baselines = make_baselines(mech, refit, test_props, truth[test_sel],
                                   test_specs, t,
                                   predict=bounded_predict(mech),
                                   fit=bounded_fit(mech))
        failed = None
    except Exception as exc:
        baselines, failed = dict(NO_BASELINES), f"{type(exc).__name__}: {exc}"
        print(f"[eval] the refit withheld-ingredient runs failed ({failed}); only "
              f"the margins are lost")
    block = measure(pred, truth[test_sel], test_props, test_specs, t,
                    baselines)
    if failed:
        block["baselines_error"] = failed
    block["fit_seconds"] = round(seconds, 2)
    block["fit_within_limit"] = bool(seconds <= FIT_TIMEOUT)
    block["refit_coeffs"] = [round(c, 8) for c in refit]
    return block


def reference_context():
    if not REFERENCE_REPORT.is_file():
        return None
    with open(REFERENCE_REPORT) as f:
        report = json.load(f)
    return {"human_reference": report.get("submitted"),
            "human_reference_refit": report.get("refit"),
            "fitted_ceiling": report.get("target"),
            "pa_scores": pa_scores(report.get("submitted"), report.get("refit")),
            "source": str(REFERENCE_REPORT)}


def write_report(output, report, started):
    scores = report["pa_scores"]
    human = ((report.get("reference") or {}).get("pa_scores")) or {}
    report["predictive_accuracy"] = round(float(np.mean([scores[pid] for pid in PA_IDS])), 4)
    report["predictive_accuracy_total"] = len(PA_IDS)
    report["human_reference"] = round(float(np.mean(list(human.values()))), 4) if human else None
    report["human_reference_total"] = len(human)
    report["wall_clock_seconds"] = round(time.time() - started, 1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")
    print(f"[eval] {json.dumps({k: round(float(v), 4) for k, v in scores.items()})}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']:.4f} "
          f"(mean of {report['predictive_accuracy_total']} criteria, each 0-1)")


def main(mechanism, output, seed=0, predict_timeout=7200, fit_timeout=7200):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    PREDICT_TIMEOUT = float(predict_timeout)
    FIT_TIMEOUT = float(fit_timeout)
    started = time.time()
    report = {"problem": PROBLEM, "seed": seed,
              "pa_scores": {pid: 0.0 for pid in PA_IDS}, "problems": []}
    if not Path(mechanism).is_file():
        report["error"] = f"{mechanism} does not exist"
        write_report(output, report, started)
        return

    props, truth, specs, manifest = read_test_set(seed)
    t = props["t"]
    report["test_set"] = manifest
    report["reference"] = reference_context()

    try:
        mech, blocking, problems = load_mechanism(mechanism)
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
        coeffs, bad = None, f"COEFFS is not a sequence of numbers: {exc}"
        problems.append(bad)
        report["problems"] = problems
        report["submitted"] = {"error": bad}
        report["refit"] = {"error": bad}
    if coeffs is not None:
        try:
            report["submitted"] = score_submitted(mech, coeffs, props, truth, specs, t)
        except Exception as exc:
            report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            report["refit"] = score_refit(mech, coeffs, props, truth, specs, t)
        except Exception as exc:
            report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = pa_scores(report["submitted"], report["refit"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
