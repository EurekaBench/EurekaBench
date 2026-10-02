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

PROBLEM = "zigzag_foraging"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
CALIBRATION_FILE = TEST_SET_DIR / "calibration.npz"
REFERENCE_DIR = PACKAGE_ROOT / "reference_mechanism"
REFERENCE_REPORT = REFERENCE_DIR / "reference_check.json"
SENSORY = ["AWAL", "AWAR", "AWCL", "AWCR", "ASEL", "ASER", "ASHL", "ASHR",
           "ASKL", "ASKR", "ASJL", "ASJR", "ASGL", "ASGR"]
INTERNEURONS = ["AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR",
                "RIAL", "RIAR", "RIBL", "RIBR", "RIML", "RIMR", "RIS", "RID",
                "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
                "PVCL", "PVCR"]
HEAD_MOTOR = ["SMDDL", "SMDDR", "SMDVL", "SMDVR", "SMBDL", "SMBDR", "SMBVL", "SMBVR",
              "RMDL", "RMDR", "RMDDL", "RMDDR", "RMDVL", "RMDVR",
              "RMED", "RMEV", "RMEL", "RMER", "RIVL", "RIVR"]
CORD_MOTOR = ([f"DA{i}" for i in range(1, 10)] + [f"VA{i}" for i in range(1, 13)]
              + [f"DB{i}" for i in range(1, 8)] + [f"VB{i}" for i in range(1, 12)]
              + [f"DD{i}" for i in range(1, 7)] + [f"VD{i}" for i in range(1, 14)])
CELLS = SENSORY + INTERNEURONS + HEAD_MOTOR + CORD_MOTOR
PANEL = HEAD_MOTOR + CORD_MOTOR
VARIANTS = {"intact": [], "no_gap": ["^.+-.+_GJ$"],
            "no_chem": ["^[A-Za-z0-9]+-[A-Za-z0-9]+$"]}
PARAM_OVERRIDES = {"neuron_to_neuron_elec_syn_gbase": "0.09 nS"}
N_SETTINGS = 100
DURATION_MS = 4000.0
DT_MS = 0.025
SAVE_EVERY_MS = 5.0
SETTLE_MS = 400.0
GUARD_MS = 300.0
PARAMETER_SET = "C1"
PROTOCOL_VERSION = 2
RESPONSE_FLOOR = 1e-8
FAMILIES = {"standard": 40, "wide": 12, "subset": 15, "no_gap": 17, "no_chem": 16}
RANGES = {"standard": {"swing_pa": (2.0, 7.0), "swing_period_ms": (600.0, 1500.0),
                       "level_pa": (0.0, 6.0), "drift_pa": (0.0, 4.0),
                       "drift_periods": (4.0, 10.0)},
          "wide": {"swing_pa": (1.0, 10.0), "swing_period_ms": (450.0, 2000.0),
                   "level_pa": (0.0, 9.0), "drift_pa": (0.0, 6.0),
                   "drift_periods": (3.0, 12.0)}}
PA_IDS = [f"PA{i}" for i in range(1, 8)]


def fingerprint(seed):
    payload = json.dumps({
        "n_settings": N_SETTINGS, "seed": seed, "duration_ms": DURATION_MS,
        "dt_ms": DT_MS, "save_every_ms": SAVE_EVERY_MS, "settle_ms": SETTLE_MS,
        "parameter_set": PARAMETER_SET, "protocol_version": PROTOCOL_VERSION,
        "families": FAMILIES, "ranges": RANGES, "cells": CELLS, "panel": PANEL,
        "variants": VARIANTS, "param_overrides": PARAM_OVERRIDES,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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


def scored_window(t):
    return t >= SETTLE_MS + GUARD_MS


def pooled_r2(pred, truth, t):
    if not usable(pred, truth):
        return None
    w = scored_window(t)
    p, y = pred[..., w], truth[..., w]
    return float(1 - np.mean((p - y) ** 2) / max(np.var(y), 1e-30))


def swing_components(x, t, period_ms):
    w = scored_window(t)
    y = x[:, w] - x[:, w].mean(axis=1, keepdims=True)
    ang = 2 * np.pi * t[w] / period_ms
    c = (y * np.cos(ang)).mean(axis=1)
    s = (y * np.sin(ang)).mean(axis=1)
    return 2 * np.hypot(c, s), np.arctan2(-s, c)


def recorded_range(x, t):
    w = scored_window(t)
    return np.percentile(x[:, w], 95, axis=1) - np.percentile(x[:, w], 5, axis=1)


def swing_table(activity, props):
    n = activity.shape[0]
    amp = np.zeros(activity.shape[:2])
    phase = np.zeros(activity.shape[:2])
    for i in range(n):
        amp[i], phase[i] = swing_components(activity[i], props["t"],
                                            props["swing_period_ms"][i])
    return amp, phase


def swing_weights(truth, props):
    amp, phase = swing_table(truth, props)
    rng = np.stack([recorded_range(truth[i], props["t"]) for i in range(truth.shape[0])])
    return np.where(rng > RESPONSE_FLOOR, np.maximum(amp, 0.0), 0.0), amp, phase


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return None
    return float(np.sum(v[keep] * w[keep]) / np.sum(w[keep]))


def weighted_corr(a, b, weights):
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if keep.sum() < 3:
        return None
    x, y, w = x[keep], y[keep], w[keep]
    total = np.sum(w)
    mx, my = np.sum(w * x) / total, np.sum(w * y) / total
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy))


def concordance(p_phase, y_phase, weights):
    d = np.angle(np.exp(1j * (np.asarray(p_phase, dtype=np.float64)
                              - np.asarray(y_phase, dtype=np.float64))))
    return weighted_mean(np.cos(d), weights)


def swing_amplitude_r(pred, truth, props):
    if not usable(pred, truth):
        return None, 0
    w, amp_true, _ = swing_weights(truth, props)
    amp_pred, _ = swing_table(pred, props)
    return weighted_corr(amp_pred, amp_true, w), int((w > 0).sum())


def swing_phase_concordance(pred, truth, props):
    if not usable(pred, truth):
        return None, 0
    w, _, ph_true = swing_weights(truth, props)
    _, ph_pred = swing_table(pred, props)
    return concordance(ph_pred, ph_true, w), int((w > 0).sum())


def slow_components(pred, truth, props):
    t = props["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    w = scored_window(t)
    slow_p, slow_y = [], []
    for i in range(truth.shape[0]):
        window = round(2 * props["swing_period_ms"][i] / max(dt, 1e-9))
        slow_p.append(moving_average(pred[i], window)[:, w])
        slow_y.append(moving_average(truth[i], window)[:, w])
    return np.stack(slow_p), np.stack(slow_y)


def slow_r2(pred, truth, props):
    if not usable(pred, truth):
        return None
    p, y = slow_components(pred, truth, props)
    return float(1 - np.mean((p - y) ** 2) / max(np.var(y), 1e-30))


def reduced_rows(specs):
    return np.array([i for i, s in enumerate(specs) if s["variant"] != "intact"])


def intact_rows(specs):
    return np.array([i for i, s in enumerate(specs) if s["variant"] == "intact"])


def removal_sign_share(pred, truth, props, specs):
    out = {"no_gap": None, "no_chem": None, "n_cases": {}}
    if not usable(pred, truth):
        return out
    amp_true, _ = swing_table(truth, props)
    amp_pred, _ = swing_table(pred, props)
    by_index = {s["index"]: i for i, s in enumerate(specs)}
    for variant in ("no_gap", "no_chem"):
        right, weight, n = [], [], 0
        for i, s in enumerate(specs):
            if s["variant"] != variant or s["twin"] not in by_index:
                continue
            j = by_index[s["twin"]]
            if any(abs(s[k] - specs[j][k]) > 1e-9
                   for k in ("swing_pa", "level_pa", "drift_pa")):
                continue
            d_true = amp_true[i] - amp_true[j]
            d_pred = amp_pred[i] - amp_pred[j]
            right.append((np.sign(d_pred) == np.sign(d_true)).astype(np.float64))
            weight.append(np.abs(d_true))
            n += int(d_true.size)
        out[variant] = (weighted_mean(np.concatenate(right), np.concatenate(weight))
                        if right else None)
        out["n_cases"][variant] = n
    return out


def reduce_props(props, ingredient):
    reduced = dict(props)
    if ingredient == "flat":
        stim = np.asarray(props["stim"], dtype=np.float64)
        reduced["stim"] = np.repeat(stim.mean(axis=-1, keepdims=True), stim.shape[-1],
                                    axis=-1)
    elif ingredient == "merged":
        elec = np.asarray(props["elec"], dtype=np.float64)
        reduced["chem"] = np.asarray(props["chem"], dtype=np.float64) + elec
        reduced["elec"] = np.zeros_like(elec)
    else:
        raise ValueError(ingredient)
    return reduced


def make_baselines(mech, coeffs, fit_props, fit_truth, score_props):
    out = {}
    for name, ingredient in (("flat_stimulus", "flat"), ("merged_wiring", "merged")):
        fitted = [float(c) for c in mech.fit_coeffs(reduce_props(fit_props, ingredient),
                                                    fit_truth, list(coeffs))]
        pred = np.asarray(mech.predict_activity(reduce_props(score_props, ingredient),
                                                fitted), dtype=np.float64)
        out[name] = {"pred": pred, "coeffs": fitted}
    return out


def measures(pred, truth, props, specs):
    t = props["t"]
    intact, reduced = intact_rows(specs), reduced_rows(specs)
    ok = usable(pred, truth)
    amp_r, n_amp = swing_amplitude_r(pred, truth, props)
    ph_c, n_ph = swing_phase_concordance(pred, truth, props)
    sign = removal_sign_share(pred, truth, props, specs)
    return {
        "pooled_r2": pooled_r2(pred, truth, t),
        "intact_pooled_r2": (pooled_r2(pred[intact], truth[intact], t)
                             if ok and len(intact) else None),
        "reduced_pooled_r2": (pooled_r2(pred[reduced], truth[reduced], t)
                              if ok and len(reduced) else None),
        "slow_r2": slow_r2(pred, truth, props),
        "swing_amplitude_r": amp_r,
        "swing_phase_concordance": ph_c,
        "removal_sign_no_gap": sign.get("no_gap"),
        "removal_sign_no_chem": sign.get("no_chem"),
        "n_swing_cases": n_amp, "n_phase_cases": n_ph,
        "n_removal_cases": sign.get("n_cases"),
        "n_intact": int(len(intact)), "n_reduced": int(len(reduced)),
    }


def measure(pred, truth, props, specs, baselines=None):
    block = measures(pred, truth, props, specs)
    if baselines is not None:
        block["baselines"] = {
            name: dict(measures(b["pred"], truth, props, specs), coeffs=b["coeffs"])
            for name, b in baselines.items()}
    return block


NO_INFORMATION = {"pooled_r2": 0.0, "intact_pooled_r2": 0.0, "reduced_pooled_r2": 0.0,
                  "slow_r2": 0.0, "swing_amplitude_r": 0.0,
                  "swing_phase_concordance": 0.0,
                  "removal_sign_no_gap": 0.5, "removal_sign_no_chem": 0.5}
PA_MEASURES = (("PA1", "submitted", ("intact_pooled_r2",)),
               ("PA2", "submitted", ("swing_amplitude_r",)),
               ("PA3", "submitted", ("swing_phase_concordance",)),
               ("PA4", "submitted", ("slow_r2",)),
               ("PA5", "submitted", ("reduced_pooled_r2",)),
               ("PA6", "submitted", ("removal_sign_no_gap",)),
               ("PA7", "refit", ("pooled_r2",)))


def normalised(value, references, floor):
    if value is None or not np.isfinite(value):
        return 0.0
    b = float(floor)
    for r in references:
        if r is not None and np.isfinite(r):
            b = max(b, float(r))
    if b >= 1.0:
        return 0.0
    return float(min(1.0, max(0.0, (float(value) - b) / (1.0 - b))))


def pa_scores(submitted, refit):
    blocks = {"submitted": submitted or {}, "refit": refit or {}}
    out = {}
    for pid, which, names in PA_MEASURES:
        block = blocks[which]
        baselines = (block.get("baselines") or {}).values()
        values = [normalised(block.get(name), [b.get(name) for b in baselines],
                             NO_INFORMATION[name]) for name in names]
        out[pid] = round(min(values) if values else 0.0, 4)
    return out


def scoring():
    return {"rule": "(m - b) / (1 - b), clipped to [0, 1], where m is the measure the "
                    "criterion is read on and b is the larger of what the two reduced "
                    "runs reach on the same cases, never taken below the value the "
                    "measure holds when the prediction carries no information",
            "no_information": dict(NO_INFORMATION),
            "measures": {pid: {"block": which, "read_on": list(names)}
                         for pid, which, names in PA_MEASURES}}


def split_halves(n):
    idx = np.arange(n)
    return idx[idx % 2 == 0], idx[idx % 2 == 1]


def subset(props, sel):
    out = dict(props)
    for key in ("stim", "swing_period_ms", "chem", "elec", "baseline"):
        out[key] = props[key][sel]
    return out


def load_test_set():
    d = np.load(TEST_SET_FILE, allow_pickle=False)
    specs = json.loads(str(d["specs"]))
    variants = sorted(VARIANTS)
    vidx = np.array([variants.index(s["variant"]) for s in specs])
    props = {"t": d["t"].astype(np.float64),
             "stim": d["stim"].astype(np.float64),
             "swing_period_ms": d["swing_period_ms"].astype(np.float64),
             "chem": d["chem_variants"].astype(np.float64)[vidx],
             "elec": d["elec_variants"].astype(np.float64)[vidx],
             "stim_index": d["stim_index"].astype(int),
             "panel_index": d["panel_index"].astype(int),
             "baseline": d["baseline"].astype(np.float64)}
    return props, d["truth"].astype(np.float64), specs


def load_calibration():
    if not CALIBRATION_FILE.is_file():
        return None
    d = np.load(CALIBRATION_FILE, allow_pickle=False)
    specs = json.loads(str(d["specs"]))
    variants = sorted(VARIANTS)
    vidx = np.array([variants.index(s["variant"]) for s in specs])
    props = {"t": d["t"].astype(np.float64),
             "stim": d["stim"].astype(np.float64),
             "swing_period_ms": d["swing_period_ms"].astype(np.float64),
             "chem": d["chem_variants"].astype(np.float64)[vidx],
             "elec": d["elec_variants"].astype(np.float64)[vidx],
             "stim_index": d["stim_index"].astype(int),
             "panel_index": d["panel_index"].astype(int),
             "baseline": d["baseline"].astype(np.float64)}
    return props, d["truth"].astype(np.float64), specs

protocol = sys.modules[__name__]


REQUIRED_MEMBERS = ("INPUTS", "COEFFS", "predict_activity", "fit_coeffs")
MAX_COEFFS = 10
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
    with open(protocol.TEST_SET_MANIFEST) as f:
        manifest = json.load(f)
    want = protocol.fingerprint(seed)
    if manifest.get("fingerprint") != want:
        raise RuntimeError(f"the test set was built for configuration "
                           f"{manifest.get('fingerprint')}, this scoring expects {want}")
    props, truth, specs = protocol.load_test_set()
    return props, truth, specs, manifest


def fitting_set(props, truth, specs):
    calibration = protocol.load_calibration()
    if calibration is not None:
        return calibration[0], calibration[1], "calibration settings"
    even = protocol.split_halves(len(specs))[0]
    return protocol.subset(props, even), truth[even], "even half of the test set"


def predict(mech, props, coeffs):
    start = time.time()
    pred = np.asarray(call_with_timeout(mech.predict_activity, PREDICT_TIMEOUT,
                                        props, list(coeffs)), dtype=np.float64)
    return pred, time.time() - start


def score_submitted(mech, coeffs, props, truth, specs):
    fit_props, fit_truth, source = fitting_set(props, truth, specs)
    start = time.time()
    fitted = [float(c) for c in call_with_timeout(mech.fit_coeffs, FIT_TIMEOUT, fit_props,
                                                  fit_truth, list(coeffs))]
    fit_seconds = time.time() - start
    start = time.time()
    baselines = protocol.make_baselines(mech, coeffs, fit_props, fit_truth, props)
    baseline_seconds = time.time() - start
    pred, seconds = predict(mech, props, fitted)
    block = protocol.measure(pred, truth, props, specs, baselines)
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_limit"] = bool(seconds <= PREDICT_TIMEOUT)
    block["fit_seconds"] = round(fit_seconds, 2)
    block["fit_within_limit"] = bool(fit_seconds <= FIT_TIMEOUT)
    block["baseline_seconds"] = round(baseline_seconds, 2)
    block["baselines_refit_on"] = source
    block["submission_refit_on"] = source
    block["submitted_coeffs"] = [round(c, 8) for c in coeffs]
    block["fitted_coeffs"] = [round(c, 8) for c in fitted]
    block["n_coeffs"] = len(coeffs)
    return block


def score_refit(mech, coeffs, props, truth, specs):
    fit_sel, test_sel = protocol.split_halves(len(specs))
    fit_props, fit_truth = protocol.subset(props, fit_sel), truth[fit_sel]
    start = time.time()
    refit = [float(c) for c in call_with_timeout(mech.fit_coeffs, FIT_TIMEOUT, fit_props,
                                                 fit_truth, list(coeffs))]
    seconds = time.time() - start
    test_props = protocol.subset(props, test_sel)
    test_specs = [specs[i] for i in test_sel]
    baselines = protocol.make_baselines(mech, coeffs, fit_props, fit_truth, test_props)
    pred, _ = predict(mech, test_props, refit)
    block = protocol.measure(pred, truth[test_sel], test_props, test_specs, baselines)
    block["fit_seconds"] = round(seconds, 2)
    block["fit_within_limit"] = bool(seconds <= FIT_TIMEOUT)
    block["refit_coeffs"] = [round(c, 8) for c in refit]
    return block


def reference_context():
    if not protocol.REFERENCE_REPORT.is_file():
        return None
    with open(protocol.REFERENCE_REPORT) as f:
        report = json.load(f)
    return {"published_mechanism": report.get("submitted"),
            "fitted_ceiling": report.get("ceiling"),
            "pa_scores": report.get("pa_scores"),
            "source": str(protocol.REFERENCE_REPORT)}


ROWS = (("PA1 intact pooled R2", "submitted", "intact_pooled_r2"),
        ("PA2 swing amplitude r", "submitted", "swing_amplitude_r"),
        ("PA3 swing phase concord.", "submitted", "swing_phase_concordance"),
        ("PA4 slow-component R2", "submitted", "slow_r2"),
        ("PA5 reduced-circuit R2", "submitted", "reduced_pooled_r2"),
        ("PA6 sign, gap removed", "submitted", "removal_sign_no_gap"),
        ("  sign, synapses (unscored)", "submitted", "removal_sign_no_chem"),
        ("PA7 refit pooled R2", "refit", "pooled_r2"))


def rows_of(report):
    blocks = {"submitted": report.get("submitted") or {},
              "refit": report.get("refit") or {}}
    context = report.get("reference") or {}
    ref = context.get("published_mechanism") or {}
    ceiling = context.get("fitted_ceiling") or {}
    out = []
    for name, which, key in ROWS:
        mine = blocks[which].get(key)
        out.append((name, mine, ref.get(key) if which == "submitted" else None,
                    ceiling.get(key) if which == "submitted" else None))
    return out


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
    print("=" * 64)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")

    def cell(value):
        return f"{value:>11.4f}" if isinstance(value, (int, float)) else f"{'-':>11}"

    print(f"  {'':28}{'submission':>11}{'reference':>11}{'fitted':>11}")
    print(f"  {'':28}{'':>11}{'mechanism':>11}{'ceiling':>11}")
    for name, a, b, c in rows_of(report):
        print(f"  {name:28}{cell(a)}{cell(b)}{cell(c)}")
    sub = report.get("submitted") or {}
    for name, b in (sub.get("baselines") or {}).items():
        print(f"  {'  ' + name + ' run':28}{cell(b.get('intact_pooled_r2'))}   intact R2,"
              f" amplitude r {cell(b.get('swing_amplitude_r')).strip()}, phase "
              f"{cell(b.get('swing_phase_concordance')).strip()}, slow R2 "
              f"{cell(b.get('slow_r2')).strip()}, reduced R2 "
              f"{cell(b.get('reduced_pooled_r2')).strip()}")
    print(f"  {'':28}{'-' * 33}")
    context = report.get("reference") or {}
    ref_scores = context.get("pa_scores") or {}
    ref_pa = (round(float(np.mean(list(ref_scores.values()))), 4) if ref_scores
              else None)
    print(f"  {'predictive accuracy':28}{cell(report['predictive_accuracy'])}"
          f"{cell(ref_pa)}{'-':>11}")
    print()
    print(f"[eval] this submission : {json.dumps(scores)}")
    if ref_scores:
        print(f"[eval] reference mech. : {json.dumps(ref_scores)}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']} over "
          f"{report['predictive_accuracy_total']} criteria"
          + (f"   (reference mechanism on the same test set: {ref_pa}, context only, "
             f"never a level to clear)" if ref_scores else ""))
    print(f"[eval] results: {path}")
    print("=" * 64)


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

    props, truth, specs, manifest = read_test_set(seed)
    report["test_set"] = manifest
    report["scoring"] = protocol.scoring()
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
            report["submitted"] = score_submitted(mech, coeffs, props, truth, specs)
        except Exception as exc:
            report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            report["refit"] = score_refit(mech, coeffs, props, truth, specs)
        except Exception as exc:
            report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.pa_scores(report["submitted"], report["refit"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
