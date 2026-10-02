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

PROBLEM = "premotor_states"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
REFERENCE_DIR = PACKAGE_ROOT / "reference_mechanism"
REFERENCE_REPORT = REFERENCE_DIR / "reference_check.json"
PANEL_FORWARD = ["AVBL", "AVBR", "RIBL", "RIBR", "RID"]
PANEL_REVERSAL = ["AVAL", "AVAR", "AVEL", "AVER", "RIML", "RIMR", "AIBL", "AIBR"]
PANEL_AVD = ["AVDL", "AVDR"]
PANEL = PANEL_FORWARD + PANEL_REVERSAL + PANEL_AVD
N_SETTINGS = 100
DURATION_MS = 10000.0
DT_MS = 0.025
SAVE_EVERY_MS = 10.0
SETTLE_MS = 500.0
EVAL_FROM_MS = 800.0
PARAMETER_SET = "C1"
PROTOCOL_VERSION = 5
POOL_SIZE = 8
MAX_BALANCE = 1.5
FAMILIES = {"alternating": 40, "independent": 25, "sparse": 10, "wide": 10,
            "hold": 10, "one_sided": 5}
RANGES = {
    "standard": {"n_per_group": (3, 8), "seg_ms": (1000.0, 3000.0),
                 "amp_pa": (5.0, 14.0), "base_frac": (0.0, 0.15)},
    "wide": {"n_per_group": (2, 8), "seg_ms": (600.0, 4500.0),
             "amp_pa": (3.0, 18.0), "base_frac": (-0.2, 0.3)},
    "hold": {"n_per_group": (3, 8), "seg_ms": (3000.0, 5000.0),
             "amp_pa": (5.0, 14.0), "base_frac": (0.0, 0.15)},
    "sparse": {"n_per_group": (1, 3), "seg_ms": (1000.0, 3000.0),
               "amp_pa": (7.0, 16.0), "base_frac": (0.0, 0.15)},
}
MARGIN_FRAC = 0.20
STATE_SMOOTH_MS = 400.0
SWITCH_TOL_MS = 500.0
CORRELATION_SPAN = 2.0
PA_IDS = [f"PA{i}" for i in range(1, 7)]


def fingerprint(seed, driven):
    payload = json.dumps({
        "n_settings": N_SETTINGS, "seed": seed, "duration_ms": DURATION_MS,
        "dt_ms": DT_MS, "save_every_ms": SAVE_EVERY_MS, "settle_ms": SETTLE_MS,
        "parameter_set": PARAMETER_SET, "protocol_version": PROTOCOL_VERSION,
        "families": FAMILIES, "ranges": RANGES, "panel": PANEL, "driven": driven,
        "pool_size": POOL_SIZE, "max_balance": MAX_BALANCE,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalise(activity, baseline, scale):
    return (np.asarray(activity, dtype=np.float64)
            - np.asarray(baseline, dtype=np.float64)[None, :, None]) \
        / np.asarray(scale, dtype=np.float64)[None, :, None]


def cell_scale(truth, baseline):
    dev = np.asarray(truth, dtype=np.float64) \
        - np.asarray(baseline, dtype=np.float64)[None, :, None]
    scale = dev.std(axis=(0, 2))
    floor = max(float(scale.max()), 1e-30) * 0.2
    return np.maximum(scale, floor)


def moving_average(x, window):
    w = max(int(window), 1)
    if w <= 1:
        return np.asarray(x, dtype=np.float64)
    pad = w // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), [(0, 0)] * (np.ndim(x) - 1)
                    + [(pad, w - 1 - pad)], mode="edge")
    csum = np.cumsum(padded, axis=-1)
    csum = np.concatenate([np.zeros(csum.shape[:-1] + (1,)), csum], axis=-1)
    return (csum[..., w:] - csum[..., :-w]) / w


def z_signal(activity, baseline, scale, t):
    x = normalise(activity, baseline, scale)
    f_rows = [PANEL.index(c) for c in PANEL_FORWARD]
    r_rows = [PANEL.index(c) for c in PANEL_REVERSAL]
    z = x[:, f_rows, :].mean(axis=1) - x[:, r_rows, :].mean(axis=1)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return moving_average(z, round(STATE_SMOOTH_MS / max(dt, 1e-9)))


def state_sequence(z, margin):
    return np.where(z > margin, 1, np.where(z < -margin, -1, 0)).astype(np.int8)


def switches(states, t):
    out = []
    prev = states[0]
    for k in range(1, len(states)):
        s = states[k]
        if s != prev and s != 0:
            out.append((float(t[k]), int(s)))
        if s != 0:
            prev = s
    return out


def dwells(states, t):
    out = {1: [], -1: []}
    start, cur = None, 0
    for k in range(len(states)):
        s = states[k]
        if s != cur:
            if cur != 0 and start is not None:
                out[cur].append(float(t[k] - start))
            start, cur = t[k], s
    if cur != 0 and start is not None:
        out[cur].append(float(t[-1] - start))
    return out


def match_fractions(pred_sw, true_sw):
    hit_true = 0
    used = set()
    for tt, d in true_sw:
        for j, (tp, dp) in enumerate(pred_sw):
            if j not in used and dp == d and abs(tp - tt) <= SWITCH_TOL_MS:
                used.add(j)
                hit_true += 1
                break
    recall = hit_true / len(true_sw) if true_sw else None
    precision = len(used) / len(pred_sw) if pred_sw else None
    return recall, precision


def usable(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    return pred.shape == truth.shape and bool(np.all(np.isfinite(pred)))


def state_metrics(pred, truth, t, baseline, scale, margin):
    if not usable(pred, truth):
        return None
    keep = t >= EVAL_FROM_MS
    tk = t[keep]
    z_p = z_signal(pred, baseline, scale, t)[:, keep]
    z_t = z_signal(truth, baseline, scale, t)[:, keep]
    agree, rec, prec = [], [], []
    wass, occ_p, occ_t = [], [], []
    n_true_switch = 0
    dw_p, dw_t = {1: [], -1: []}, {1: [], -1: []}
    for i in range(z_t.shape[0]):
        sp = state_sequence(z_p[i], margin)
        st = state_sequence(z_t[i], margin)
        agree.append(float(np.mean(sp == st)))
        r, p = match_fractions(switches(sp, tk), switches(st, tk))
        n_true_switch += len(switches(st, tk))
        if r is not None:
            rec.append(r)
        if p is not None:
            prec.append(p)
        for s in (1, -1):
            dp, dt_ = dwells(sp, tk), dwells(st, tk)
            dw_p[s] += dp[s]
            dw_t[s] += dt_[s]
        occ_p.append([float(np.mean(sp == s)) for s in (1, -1, 0)])
        occ_t.append([float(np.mean(st == s)) for s in (1, -1, 0)])
    from scipy.stats import wasserstein_distance
    norm = []
    for s in (1, -1):
        if dw_p[s] and dw_t[s]:
            w = float(wasserstein_distance(dw_p[s], dw_t[s]))
            bound = float(np.mean(dw_p[s]) + np.mean(dw_t[s]))
        elif dw_t[s]:
            w = float(np.mean(dw_t[s]))
            bound = w
        else:
            continue
        wass.append(w)
        norm.append(w / bound if bound > 0 else 1.0)
    recall = float(np.mean(rec)) if rec else 0.0
    precision = float(np.mean(prec)) if prec else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if recall + precision > 0 else 0.0)
    return {"state_agreement": float(np.mean(agree)),
            "switch_recall": recall, "switch_precision": precision,
            "switch_f1": f1,
            "dwell_wasserstein_ms": float(np.mean(wass)) if wass else None,
            "dwell_distance_norm": float(np.mean(norm)) if norm else None,
            "occupancy_mae": float(np.mean(np.abs(np.array(occ_p)
                                                  - np.array(occ_t)))),
            "n_recorded_switches": int(n_true_switch)}


def correlation_distance(pred, truth):
    if not usable(pred, truth):
        return None
    vals = []
    iu = np.triu_indices(truth.shape[1], k=1)
    for i in range(truth.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            cp = np.corrcoef(pred[i])
            ct = np.corrcoef(truth[i])
        sel = np.isfinite(ct[iu])
        if sel.any():
            diff = np.where(np.isfinite(cp[iu]), cp[iu], 0.0)[sel] - ct[iu][sel]
            vals.append(float(np.mean(np.abs(diff))))
    return float(np.mean(vals)) if vals else None


def pooled_r2(pred, truth):
    if not usable(pred, truth):
        return None
    return float(1 - np.mean((pred - truth) ** 2) / max(np.var(truth), 1e-30))


def measure(pred, truth, t, baseline, scale, margin):
    block = state_metrics(pred, truth, t, baseline, scale, margin) or {}
    block["correlation_mad"] = correlation_distance(pred, truth)
    block["pooled_r2"] = pooled_r2(pred, truth)
    return block


def withhold(props, ingredient):
    reduced = dict(props)
    if ingredient == "flat":
        for key in ("drive", "driver"):
            x = np.asarray(props[key], dtype=np.float64)
            reduced[key] = np.repeat(x.mean(axis=-1, keepdims=True), x.shape[-1],
                                     axis=-1)
    else:
        reduced["chem_rec"] = np.zeros_like(np.asarray(props["chem_rec"]))
        reduced["elec_rec"] = np.zeros_like(np.asarray(props["elec_rec"]))
    return reduced


def direct_call(fn, *args):
    return fn(*args)


def make_baselines(mech, coeffs, props, truth, t, baseline, scale, margin,
                   call=direct_call):
    out = {}
    for name in ("flat", "uncoupled"):
        reduced = withhold(props, name)
        fitted = [float(c) for c in call(mech.fit_coeffs, reduced, truth, list(coeffs))]
        pred = np.asarray(call(mech.predict_activity, reduced, fitted), dtype=np.float64)
        out[name] = measure(pred, truth, t, baseline, scale, margin)
        out[f"coeffs_{name}"] = fitted
    return out


def clip01(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def fraction(block, key):
    return clip01(block.get(key))


def one_minus(block, key, span=1.0):
    value = block.get(key)
    if value is None or not np.isfinite(value):
        return 0.0
    return clip01(1.0 - float(value) / span)


def margin_over(sub, reduced, key):
    value = sub.get(key)
    if value is None or not np.isfinite(value):
        return 0.0
    out = []
    for name in ("flat", "uncoupled"):
        other = (reduced.get(name) or {}).get(key)
        if other is None or not np.isfinite(other):
            return 0.0
        out.append(clip01(float(value) - float(other)))
    return min(out) if out else 0.0


def pa_scores(sub, reduced, refit):
    return {"PA1": fraction(sub, "state_agreement"),
            "PA2": fraction(sub, "switch_f1"),
            "PA3": one_minus(sub, "dwell_distance_norm"),
            "PA4": one_minus(sub, "correlation_mad", CORRELATION_SPAN),
            "PA5": margin_over(sub, reduced, "switch_f1"),
            "PA6": fraction(refit, "state_agreement")}


def subset(props, sel):
    out = dict(props)
    for key in ("drive", "driver", "init"):
        out[key] = np.asarray(props[key])[sel]
    return out


def load_test_set():
    d = np.load(TEST_SET_FILE, allow_pickle=False)
    props = {k: d[k].astype(np.float64)
             for k in ("t", "drive", "driver", "chem", "elec", "chem_rec",
                       "elec_rec", "baseline", "init")}
    return props, d["truth"].astype(np.float64), json.loads(str(d["specs"]))

protocol = sys.modules[__name__]


REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_activity", "fit_coeffs")
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
    want = protocol.fingerprint(seed, manifest.get("driven") or [])
    if manifest.get("fingerprint") != want:
        raise RuntimeError(f"the test set was built for configuration "
                           f"{manifest.get('fingerprint')}, this scoring expects {want}")
    props, truth, specs = protocol.load_test_set()
    return props, truth, specs, manifest


def reading(props, truth):
    baseline = props["baseline"]
    scale = protocol.cell_scale(truth, baseline)
    margin = protocol.MARGIN_FRAC * float(
        protocol.z_signal(truth, baseline, scale, props["t"]).std())
    return baseline, scale, margin


def predict(mech, props, coeffs):
    start = time.time()
    pred = np.asarray(call_with_timeout(mech.predict_activity, PREDICT_TIMEOUT,
                                        props, list(coeffs)), dtype=np.float64)
    return pred, time.time() - start


def budgeted(fn, *args):
    timeout = (PREDICT_TIMEOUT if getattr(fn, "__name__", "") == "predict_activity"
               else FIT_TIMEOUT)
    return call_with_timeout(fn, timeout, *args)


def score_submitted(mech, coeffs, props, truth, baseline, scale, margin):
    t = props["t"]
    pred, seconds = predict(mech, props, coeffs)
    block = protocol.measure(pred, truth, t, baseline, scale, margin)
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_limit"] = bool(seconds <= PREDICT_TIMEOUT)
    block["n_coeffs"] = len(coeffs)
    reduced = protocol.make_baselines(mech, coeffs, props, truth, t, baseline, scale,
                                      margin, call=budgeted)
    return block, {k: reduced[k] for k in ("flat", "uncoupled")}


def score_refit(mech, coeffs, props, truth, specs, baseline, scale, margin):
    t = props["t"]
    half = len(specs) // 2
    fit_sel, test_sel = np.arange(half), np.arange(half, len(specs))
    start = time.time()
    refit = [float(c) for c in call_with_timeout(
        mech.fit_coeffs, FIT_TIMEOUT, protocol.subset(props, fit_sel),
        truth[fit_sel], list(coeffs))]
    seconds = time.time() - start
    test_props = protocol.subset(props, test_sel)
    pred, _ = predict(mech, test_props, refit)
    block = protocol.measure(pred, truth[test_sel], t, baseline, scale, margin)
    block["fit_seconds"] = round(seconds, 2)
    block["fit_within_limit"] = bool(seconds <= FIT_TIMEOUT)
    block["refit_coeffs"] = [round(c, 8) for c in refit]
    reduced = protocol.make_baselines(mech, refit, test_props, truth[test_sel], t,
                                      baseline, scale, margin, call=budgeted)
    return block, {k: reduced[k] for k in ("flat", "uncoupled")}


def reference_context():
    if not protocol.REFERENCE_REPORT.is_file():
        return None
    with open(protocol.REFERENCE_REPORT) as f:
        report = json.load(f)
    return {"published_mechanism": report.get("submitted"),
            "fitted_ceiling": report.get("ceiling"),
            "pa_scores": report.get("pa_scores"),
            "predictive_accuracy": report.get("predictive_accuracy"),
            "source": str(protocol.REFERENCE_REPORT)}


ROWS = [("PA1 state agreement", "state_agreement"),
        ("PA2 switch F1", "switch_f1"),
        ("dwell Wasserstein ms", "dwell_wasserstein_ms"),
        ("PA3 dwell dist / bound", "dwell_distance_norm"),
        ("PA4 correlation MAD", "correlation_mad"),
        ("pooled R2 (reported)", "pooled_r2")]


def write_report(output, report, started):
    scores = report["pa_scores"]
    report["predictive_accuracy"] = round(float(np.mean(list(scores.values()))), 4)
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
    context = report.get("reference") or {}
    ref_scores = context.get("pa_scores") or {}
    sub = report.get("submitted") or {}
    ref = context.get("published_mechanism") or {}
    ceiling = context.get("fitted_ceiling") or {}
    reduced = report.get("reduced") or {}

    def cell(value):
        return f"{value:>11.4f}" if isinstance(value, (int, float)) else f"{'-':>11}"

    print(f"  {'':26}{'submission':>11}{'flat':>11}{'uncoupled':>11}{'published':>11}"
          f"{'ceiling':>11}")
    for name, key in ROWS:
        print(f"  {name:26}{cell(sub.get(key))}"
              f"{cell((reduced.get('flat') or {}).get(key))}"
              f"{cell((reduced.get('uncoupled') or {}).get(key))}"
              f"{cell(ref.get(key))}{cell(ceiling.get(key))}")
    print(f"  {'PA6 refit state agreement':26}"
          f"{cell((report.get('refit') or {}).get('state_agreement'))}")
    print(f"  {'':26}{'-' * 55}")
    ref_pa = context.get("predictive_accuracy")
    print(f"  {'predictive accuracy':26}{cell(report['predictive_accuracy'])}"
          f"{'':>22}{cell(ref_pa)}")
    print()
    print(f"[eval] this submission : {json.dumps(scores)}")
    if ref_scores:
        print(f"[eval] published mech. : {json.dumps(ref_scores)}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']} over "
          f"{report['predictive_accuracy_total']} criteria"
          + (f"   (published mechanism on the same test set: {ref_pa}, "
             f"context only, never a level to clear)" if ref_scores else ""))
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
    baseline, scale, margin = reading(props, truth)
    report["test_set"] = manifest
    report["state_reading"] = {"margin": margin, "state_smooth_ms": protocol.STATE_SMOOTH_MS,
                               "switch_tol_ms": protocol.SWITCH_TOL_MS,
                               "eval_from_ms": protocol.EVAL_FROM_MS}
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
    empty = {"flat": {}, "uncoupled": {}}
    report["submitted"], report["reduced"] = {}, empty
    report["refit"], report["refit_reduced"] = {}, empty
    if coeffs is not None:
        try:
            report["submitted"], report["reduced"] = score_submitted(
                mech, coeffs, props, truth, baseline, scale, margin)
        except Exception as exc:
            report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            report["refit"], report["refit_reduced"] = score_refit(
                mech, coeffs, props, truth, specs, baseline, scale, margin)
        except Exception as exc:
            report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.pa_scores(report["submitted"], report["reduced"],
                                             report["refit"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
