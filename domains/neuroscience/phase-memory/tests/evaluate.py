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

from scipy.signal import hilbert
import fire
import numpy as np

PROBLEM = "phase_memory"

PROTOCOL_VERSION = 4
CELLS = [
    "AWAL", "AWAR",
    "AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR",
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR",
    "RIAL", "RIAR", "RIBL", "RIBR", "RIML", "RIMR", "RIS", "RID", "RIVL", "RIVR",
    "SMDDL", "SMDDR", "SMDVL", "SMDVR", "RMDDL", "RMDDR", "RMDVL", "RMDVR", "RMDL", "RMDR",
    "RMED", "RMEV", "RMEL", "RMER", "SAAVL", "SAAVR", "SAADL", "SAADR",
]
ODOR_CELLS = ["AWAL", "AWAR"]
DORSAL_CELLS = ["SMDDL", "SMDDR"]
VENTRAL_CELLS = ["SMDVL", "SMDVR"]
REVERSAL_CELLS = ["AVAL", "AVAR"]
STIM_CELLS = ODOR_CELLS + DORSAL_CELLS + VENTRAL_CELLS + REVERSAL_CELLS
ENVIRONMENT = {
    "name": "C0_rimv16_aibv8_e5i2_pulse0.5",
    "parameter_set": "C0",
    "amplitude_scale": 0.3,
    "odor_gain": 5.0,
    "odor_fraction": 0.5,
    "connection_polarity_override": {
        "^SMDD[LR]-SMDV[LR]$": "inh", "^SMDV[LR]-SMDD[LR]$": "inh",
        "^RMDD[LR]-RMDV[LR]$": "inh", "^RMDV[LR]-RMDD[LR]$": "inh",
    },
    "connection_number_scaling": {
        "^SMDD[LR]-SMDV[LR]$": 8.0, "^SMDV[LR]-SMDD[LR]$": 8.0,
        "^RMDD[LR]-RMDV[LR]$": 4.0, "^RMDV[LR]-RMDD[LR]$": 4.0,
        "^RIM[LR]-SMDD[LR]$": 16.0, "^RIM[LR]-SMDV[LR]$": 16.0,
        "^AIB[LR]-SMDD[LR]$": 16.0, "^AIB[LR]-SMDV[LR]$": 8.0,
        "^SMD[DV][LR]-RIA[LR]$": 16.0, "^RIA[LR]-SMD[DV][LR]$": 16.0,
    },
    "param_overrides": {
        "exc_syn_ar": "5 per_s", "inh_syn_ar": "2 per_s",
        "neuron_to_neuron_exc_syn_conductance": "0.1 nS",
        "neuron_to_neuron_inh_syn_conductance": "0.1 nS",
    },
}
DT_MS = 0.025
SAVE_EVERY_MS = 5.0
SETTLE_MS = 300.0
POST_MS = 200.0
FILTER_TAU_MS = 200.0
ODOR_FRACTION = 0.25
INTERNAL_PERIOD_MS = 480.0
REVERSAL_OFFSET_MS = -120.0
REVERSAL_JITTER_MS = 20.0
RANGES = {
    "standard": {"period_ms": (240.0, 420.0), "cycles": (3, 5), "swing_pa": (3.0, 6.0),
                 "odor_pa": (6.0, 12.0), "reversal_pa": (14.0, 14.0),
                 "reversal_periods": (2, 5)},
    "long_reversal": {"period_ms": (240.0, 420.0), "cycles": (3, 5), "swing_pa": (3.0, 6.0),
                      "odor_pa": (6.0, 12.0), "reversal_pa": (14.0, 14.0),
                      "reversal_periods": (7, 10)},
}
FAMILIES = {"standard": 54, "phase_pair": 12, "long_reversal": 34}


def fingerprint(seed):
    payload = json.dumps({"protocol": PROTOCOL_VERSION, "seed": int(seed), "cells": CELLS,
                          "stim_cells": STIM_CELLS, "environment": ENVIRONMENT, "dt": DT_MS,
                          "save": SAVE_EVERY_MS, "settle": SETTLE_MS, "post": POST_MS,
                          "tau": FILTER_TAU_MS, "odor_fraction": ODOR_FRACTION,
                          "internal_period": INTERNAL_PERIOD_MS, "jitter": REVERSAL_JITTER_MS,
                          "offset": REVERSAL_OFFSET_MS,
                          "ranges": RANGES, "families": FAMILIES}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def windows(props):
    t = props["t"]
    rev_row = STIM_CELLS.index(REVERSAL_CELLS[0])
    out = []
    for n in range(props["stim"].shape[0]):
        on = np.flatnonzero(props["stim"][n, rev_row] > 0)
        start, end = int(on[0]), int(on[-1]) + 1
        post_end = int(np.searchsorted(t, t[end - 1] + POST_MS, side="right"))
        out.append((start, end, min(post_end, len(t))))
    return out


MIN_CROSSINGS = 2
TURN_FLOOR = 0.01


def oscillates(seg):
    seg = seg - seg.mean()
    return int(np.sum((seg[1:] > 0) != (seg[:-1] > 0))) >= MIN_CROSSINGS


def phase_of(x, start, end):
    seg = np.asarray(x[start:end], dtype=np.float64)
    if not oscillates(seg):
        return None
    return np.angle(hilbert(seg - seg.mean()))


def plv(pa, pb):
    if pa is None or pb is None:
        return 0.0
    d = np.asarray(pa) - np.asarray(pb)
    return float(np.abs(np.mean(np.exp(1j * d)))) if len(d) else 0.0


def circular_distance(a, b):
    return float(np.abs(np.angle(np.exp(1j * (a - b)))))


PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
REFERENCE_DIR = PACKAGE_ROOT / "reference_mechanism"
REFERENCE_REPORT = REFERENCE_DIR / "reference_check.json"
PA_IDS = ("PA1", "PA2", "PA3", "PA4", "PA5", "PA6")


def load_test_set():
    with np.load(TEST_SET_FILE) as z:
        props = {"t": z["t"].astype(np.float64), "stim": z["stim"].astype(np.float32),
                 "stim_cells": z["stim_cells"]}
        truth = {"headswing": z["headswing"].astype(np.float64),
                 "turn": z["turn"].astype(np.float64)}
        specs = json.loads(str(z["specs"]))
        pairs = [tuple(p) for p in json.loads(str(z["pairs"]))]
    return props, truth, specs, pairs


def subset(props, sel):
    return {"t": props["t"], "stim": props["stim"][sel], "stim_cells": props["stim_cells"]}


def subset_truth(truth, sel):
    return {"headswing": truth["headswing"][sel], "turn": truth["turn"][sel]}


def phases(headswing, props):
    return [(phase_of(headswing[n], s, e), s, e) for n, (s, e, post) in
            enumerate(windows(props))]


def scored_rows(truth_hs, props):
    return [n for n, (pt, start, end) in enumerate(phases(truth_hs, props)) if pt is not None]


def median_plv(pred, truth_hs, props, last_third=False):
    values = []
    rows = scored_rows(truth_hs, props)
    for n, ((pp, s, e), (pt, start, end)) in enumerate(zip(phases(pred, props),
                                                           phases(truth_hs, props))):
        if n not in rows:
            continue
        if pp is None:
            values.append(0.0)
            continue
        k = int(len(pt) * 2 / 3) if last_third else 0
        values.append(plv(pp[k:], pt[k:]))
    return (float(np.median(values)) if values else float("nan")), values


def end_phase_error(pred, truth_hs, props):
    values = []
    for (pp, s, e), (pt, start, end) in zip(phases(pred, props), phases(truth_hs, props)):
        if pt is None:
            continue
        values.append(np.pi if pp is None else circular_distance(pp[-1], pt[-1]))
    return (float(np.median(values)) if values else float("nan")), values


def forward_amplitude(headswing, props):
    return np.array([float(np.std(headswing[n, :s])) for n, (s, end, post)
                     in enumerate(windows(props))])


def decided(truth, props):
    return np.abs(truth["turn"]) >= TURN_FLOOR * forward_amplitude(truth["headswing"], props)


def balanced_accuracy(pred_turn, true_turn, sel=None):
    p = np.sign(np.asarray(pred_turn, dtype=np.float64))
    y = np.sign(np.asarray(true_turn, dtype=np.float64))
    if sel is not None:
        p, y = p[sel], y[sel]
    keep = y != 0
    p, y = p[keep], y[keep]
    if len(y) == 0:
        return float("nan")
    accs = []
    for side in (1.0, -1.0):
        m = y == side
        if m.any():
            accs.append(float(np.mean(p[m] == side)))
    return float(np.mean(accs))


def pair_flip(pred_turn, true_turn, pairs, ok=None):
    pairs = [(a, b) for a, b in pairs
             if (ok is None or (ok[a] and ok[b]))
             and np.sign(true_turn[a]) * np.sign(true_turn[b]) < 0]
    if not pairs:
        return float("nan")
    hits = 0
    for a, b in pairs:
        dp = np.sign(pred_turn[a]) - np.sign(pred_turn[b])
        dt = np.sign(true_turn[a]) - np.sign(true_turn[b])
        hits += int(dp != 0 and np.sign(dp) == np.sign(dt))
    return float(hits / len(pairs))


def measure(pred_hs, pred_turn, truth, props, specs, pairs, seed=0):
    pred_hs = np.asarray(pred_hs, dtype=np.float64)
    pred_turn = np.asarray(pred_turn, dtype=np.float64)
    ok = decided(truth, props)
    long_sel = np.array([s["family"] == "long_reversal" for s in specs]) & ok
    plv_med, plv_per = median_plv(pred_hs, truth["headswing"], props)
    ret_med, ret_per = median_plv(pred_hs, truth["headswing"], props, last_third=True)
    err_med, err_per = end_phase_error(pred_hs, truth["headswing"], props)
    rows = scored_rows(truth["headswing"], props)
    long_err = [e for r, e in zip(rows, err_per) if long_sel[r]]
    return {
        "end_phase_error_long": float(np.median(long_err)) if long_err else float("nan"),
        "plv_median": plv_med,
        "end_phase_error_median": err_med,
        "balanced_accuracy": balanced_accuracy(pred_turn, truth["turn"], ok),
        "balanced_accuracy_long": (balanced_accuracy(pred_turn, truth["turn"], long_sel)
                                   if long_sel.any() else float("nan")),
        "pair_flip": pair_flip(pred_turn, truth["turn"], pairs, ok),
        "n_oscillating": int(len(scored_rows(truth["headswing"], props))),
        "n_decided": int(ok.sum()),
        "retention_median": ret_med,
        "n_settings": int(len(specs)),
        "n_long": int(long_sel.sum()),
        "n_pairs": int(len(pairs)),
        "per_setting_plv": [round(v, 4) for v in plv_per],
        "per_setting_end_error": [round(v, 4) for v in err_per],
    }


MIN_SCORED_SHARE = 0.5
HALF_PI = np.pi / 2


def clip01(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(min(1.0, max(0.0, v))) if np.isfinite(v) else 0.0


def phase_error_score(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return clip01(1.0 - v / HALF_PI) if np.isfinite(v) else 0.0


def accuracy_score(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return clip01(2.0 * (v - 0.5)) if np.isfinite(v) else 0.0


def pa_scores(sub, refit):
    out = {pid: 0.0 for pid in PA_IDS}
    sub, refit = sub or {}, refit or {}
    if "plv_median" not in sub:
        return out
    n = max(int(sub.get("n_settings", 0)), 1)
    if sub.get("n_oscillating", 0) < MIN_SCORED_SHARE * n or \
            sub.get("n_decided", 0) < MIN_SCORED_SHARE * n:
        return out
    out["PA1"] = clip01(sub.get("plv_median"))
    out["PA2"] = phase_error_score(sub.get("end_phase_error_median"))
    out["PA3"] = accuracy_score(sub.get("balanced_accuracy"))
    out["PA4"] = accuracy_score(sub.get("balanced_accuracy_long"))
    out["PA5"] = clip01(sub.get("pair_flip"))
    out["PA6"] = accuracy_score(refit.get("balanced_accuracy"))
    return {pid: round(out[pid], 4) for pid in PA_IDS}


def predictive_accuracy(scores):
    return round(float(np.mean([scores.get(pid, 0.0) for pid in PA_IDS])), 4)


def metrics():
    return {"PA1": "the median over the held-out settings of the phase-locking value "
                   "between the predicted and the recorded headswing signal over the "
                   "reversal, which is the score itself",
            "PA2": "the median end-of-reversal phase error, scored 1 - error / (pi/2)",
            "PA3": "the balanced accuracy of the sign of the predicted turn signal, "
                   "scored 2 x (accuracy - 0.5)",
            "PA4": "the same balanced accuracy over the settings whose reversal command "
                   "lasts a duration outside the range of the agent's observations and "
                   "experiments",
            "PA5": "the share of the phase pairs whose recorded turns go to opposite "
                   "sides on which the predicted turns go to opposite sides in the same "
                   "order",
            "PA6": "the balanced accuracy of the turn sign on the half of the settings "
                   "the constants were not refit on"}

protocol = sys.modules[__name__]


REQUIRED_MEMBERS = ("INPUTS", "COEFFS", "predict_headswing", "predict_turn", "fit_coeffs")
MAX_COEFFS = 10
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
    if hasattr(module, "COEFFS") and len(list(module.COEFFS)) > MAX_COEFFS:
        problems.append(f"COEFFS has {len(list(module.COEFFS))} entries, limit {MAX_COEFFS}")
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
    props, truth, specs, pairs = protocol.load_test_set()
    return props, truth, specs, pairs, manifest


def predict(mech, props, coeffs):
    start = time.time()
    hs = np.asarray(call_with_timeout(mech.predict_headswing, PREDICT_TIMEOUT, props,
                                      list(coeffs)), dtype=np.float64)
    turn = np.asarray(call_with_timeout(mech.predict_turn, PREDICT_TIMEOUT, props,
                                        list(coeffs)), dtype=np.float64)
    return hs, turn, time.time() - start


def check_shapes(hs, turn, props):
    n, t = props["stim"].shape[0], len(props["t"])
    if hs.shape != (n, t):
        raise ValueError(f"predict_headswing returned {hs.shape}, expected {(n, t)}")
    if turn.shape != (n,):
        raise ValueError(f"predict_turn returned {turn.shape}, expected {(n,)}")


def score_submitted(mech, coeffs, props, truth, specs, pairs, seed):
    hs, turn, seconds = predict(mech, props, coeffs)
    check_shapes(hs, turn, props)
    block = protocol.measure(hs, turn, truth, props, specs, pairs, seed)
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_limit"] = bool(seconds <= PREDICT_TIMEOUT)
    block["n_coeffs"] = len(coeffs)
    return block


def score_refit(mech, coeffs, props, truth, specs, pairs, seed):
    half = len(specs) // 2
    fit_sel, test_sel = np.arange(half), np.arange(half, len(specs))
    start = time.time()
    refit = [float(c) for c in call_with_timeout(
        mech.fit_coeffs, FIT_TIMEOUT, protocol.subset(props, fit_sel),
        protocol.subset_truth(truth, fit_sel), list(coeffs))]
    seconds = time.time() - start
    tp = protocol.subset(props, test_sel)
    tt = protocol.subset_truth(truth, test_sel)
    ts = [specs[i] for i in test_sel]
    tpairs = [(a - half, b - half) for a, b in pairs if a >= half and b >= half]
    hs, turn, _ = predict(mech, tp, refit)
    check_shapes(hs, turn, tp)
    block = protocol.measure(hs, turn, tt, tp, ts, tpairs, seed)
    block["fit_seconds"] = round(seconds, 2)
    block["fit_within_limit"] = bool(seconds <= FIT_TIMEOUT)
    block["refit_coeffs"] = [float(f"{c:.6g}") for c in refit]
    return block


def reference_context():
    if not protocol.REFERENCE_REPORT.is_file():
        return None
    with open(protocol.REFERENCE_REPORT) as f:
        report = json.load(f)
    return {"published_mechanism": report.get("submitted"), "ceiling": report.get("ceiling"),
            "pa_scores": report.get("pa_scores"),
            "predictive_accuracy": report.get("predictive_accuracy"),
            "source": str(protocol.REFERENCE_REPORT)}


ROWS = (("PA1 median PLV", "plv_median", "submitted"),
        ("PA2 end phase error", "end_phase_error_median", "submitted"),
        ("PA3 balanced accuracy", "balanced_accuracy", "submitted"),
        ("PA4 long-family accuracy", "balanced_accuracy_long", "submitted"),
        ("PA5 pair flip", "pair_flip", "submitted"),
        ("PA6 refit accuracy", "balanced_accuracy", "refit"))


def write_report(output, report, started):
    scores = report["pa_scores"]
    report["predictive_accuracy"] = protocol.predictive_accuracy(scores)
    report["wall_clock_seconds"] = round(time.time() - started, 1)
    path = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("=" * 64)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")
    sub = report.get("submitted") or {}
    refit = report.get("refit") or {}
    context = report.get("reference") or {}
    ref = context.get("published_mechanism") or {}
    ceiling = (context.get("ceiling") or {}).get("oracle_sinusoid") or {}

    def cell(value):
        return f"{value:>11.4f}" if isinstance(value, (int, float)) else f"{'-':>11}"

    print(f"  {'':26}{'submission':>11}{'published':>11}{'ceiling':>11}{'score':>11}")
    for name, key, block in ROWS:
        source = refit if block == "refit" else sub
        print(f"  {name:26}{cell(source.get(key))}{cell(ref.get(key))}"
              f"{cell(ceiling.get(key))}{cell(scores.get(name.split()[0]))}")
    ref_scores = context.get("pa_scores") or {}
    print(f"[eval] this submission : {json.dumps(scores)}")
    if ref_scores:
        print(f"[eval] published mech. : {json.dumps(ref_scores)}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']}"
          + (f"   (published mechanism on the same test set: "
             f"{context.get('predictive_accuracy')}, context only, never scored)"
             if ref_scores else ""))
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

    props, truth, specs, pairs, manifest = read_test_set(seed)
    report["test_set"] = {k: v for k, v in manifest.items() if k != "phase_pairs"}
    report["metrics"] = protocol.metrics()
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
    try:
        report["submitted"] = score_submitted(mech, coeffs, props, truth, specs, pairs, seed)
    except Exception as exc:
        report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["refit"] = score_refit(mech, coeffs, props, truth, specs, pairs, seed)
    except Exception as exc:
        report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["pa_scores"] = protocol.pa_scores(report["submitted"], report["refit"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
