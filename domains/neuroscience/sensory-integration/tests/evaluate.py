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

PROBLEM = "sensory_integration"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
REFERENCE_DIR = PACKAGE_ROOT / "reference_mechanism"
REFERENCE_REPORT = REFERENCE_DIR / "reference_check.json"
N_SETTINGS = 100
DURATION_S = 100.0
SEGMENT_S = 10.0
GUARD_S = 2.0
SETTINGS_PER_CLIP = int(DURATION_S // SEGMENT_S)
FPS = 8.0
FRAME = 224
SAMPLE_RATE = 16000
NOISE_SEED = 20240304
EVENT_PERIOD_S = 2.0
EVENT_ONSET_S = 1.0
VISUAL_EVENT_S = 0.9
SPEECH_MAX_S = 0.9
CLASS_WORDS = ("hammer", "violin")
CLASS_RGB = ((1.0, 0.30, 0.12), (0.12, 0.38, 1.0))
WORD_FONT_PX = 46
WORD_JITTER_PX = 14
CARRIER_CONTRAST = 0.45
CARRIER_REFRESH_HZ = 4.0
CARRIER_AUDIO_LEVEL = 0.25
EVENT_AUDIO_LEVEL = 0.7
INFORMATIVENESS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TRANSCRIBE = True
N_PANEL = 800
PROTOCOL_VERSION = 2
PA_IDS = [f"PA{i}" for i in range(1, 5)]
PA_LABELS = {
    "PA1": "how much of the two senses together reaches a location",
    "PA2": "which of the two senses a location favours",
    "PA3": "how much of a location's no-stimulus level survives",
    "PA4": "which locations carry the part neither sense accounts for",
}
RIDGE = 1e-9


def fingerprint(seed):
    payload = json.dumps({
        "n_settings": N_SETTINGS, "seed": seed, "duration_s": DURATION_S,
        "segment_s": SEGMENT_S, "guard_s": GUARD_S,
        "settings_per_clip": SETTINGS_PER_CLIP,
        "fps": FPS, "frame": FRAME, "sample_rate": SAMPLE_RATE,
        "noise_seed": NOISE_SEED, "event_period_s": EVENT_PERIOD_S,
        "event_onset_s": EVENT_ONSET_S, "visual_event_s": VISUAL_EVENT_S,
        "speech_max_s": SPEECH_MAX_S, "words": CLASS_WORDS,
        "word_font_px": WORD_FONT_PX, "word_jitter_px": WORD_JITTER_PX,
        "class_rgb": CLASS_RGB,
        "carrier_contrast": CARRIER_CONTRAST, "carrier_refresh_hz": CARRIER_REFRESH_HZ,
        "carrier_audio_level": CARRIER_AUDIO_LEVEL, "transcribe": TRANSCRIBE,
        "event_audio_level": EVENT_AUDIO_LEVEL, "informativeness": INFORMATIVENESS,
        "n_panel": N_PANEL, "protocol_version": PROTOCOL_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def usable(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    return pred.shape == truth.shape and bool(np.all(np.isfinite(pred)))


def response_design(props):
    base = np.asarray(props["baseline"], dtype=np.float64)
    a = np.asarray(props["auditory"], dtype=np.float64) - base
    v = np.asarray(props["visual"], dtype=np.float64) - base
    columns = [a, v, base, np.ones_like(base)]
    flat = [c.transpose(1, 0, 2).reshape(base.shape[1], -1) for c in columns]
    return a, v, base, np.stack(flat, axis=-1)


def location_solver(x):
    gram = np.einsum("pmk,pml->pkl", x, x) + RIDGE * np.eye(x.shape[-1])[None]
    inverse = np.linalg.inv(gram)

    def apply(y):
        flat = np.asarray(y, dtype=np.float64).transpose(1, 0, 2)
        flat = flat.reshape(x.shape[0], -1)
        weights = np.einsum("pkl,pml,pm->pk", inverse, x, flat)
        return weights, flat - np.einsum("pmk,pk->pm", x, weights)
    return apply


def location_profiles(weights, remainder):
    raw = {"pass_through": 0.5 * (weights[:, 0] + weights[:, 1]),
           "sense_balance": weights[:, 0] - weights[:, 1],
           "baseline_retention": weights[:, 2],
           "extra_size": remainder.std(axis=1)}
    return {key: value - value.mean() for key, value in raw.items()}


def location_statistics(a, v, base):
    axes = (0, 2)
    abs_a, abs_v = np.abs(a), np.abs(v)
    mean_a, mean_v = abs_a.mean(axis=axes), abs_v.mean(axis=axes)
    columns = [mean_a, mean_v, a.std(axis=axes), v.std(axis=axes),
               base.std(axis=axes), base.mean(axis=axes),
               (abs_a * abs_v).mean(axis=axes),
               np.minimum(abs_a, abs_v).mean(axis=axes),
               mean_a / (mean_a + mean_v + 1e-9)]
    return np.stack(columns + [np.ones(len(mean_a))], axis=1)


def r_squared(pred, truth):
    var = float(np.var(truth))
    if var <= 0 or len(np.atleast_1d(truth)) < 2:
        return None
    return float(1.0 - np.mean((pred - truth) ** 2) / var)


RAW = {"PA1": "pass_through", "PA2": "sense_balance",
       "PA3": "baseline_retention", "PA4": "extra_size"}


def measure(pred, truth, props, specs, baselines=None):
    a, v, base, x = response_design(props)
    solve = location_solver(x)
    weights_true, remainder_true = solve(truth)
    recorded = location_profiles(weights_true, remainder_true)
    block = {"pass_through_recorded": float(0.5 * (weights_true[:, 0]
                                                   + weights_true[:, 1]).mean()),
             "sense_balance_recorded": float((weights_true[:, 0]
                                              - weights_true[:, 1]).mean()),
             "baseline_retention_recorded": float(weights_true[:, 2].mean()),
             "extra_size_recorded": float(remainder_true.std())}
    if not usable(pred, truth):
        block.update({key: None for key in RAW.values()})
        return block
    weights, remainder = solve(np.asarray(pred, dtype=np.float64))
    predicted = location_profiles(weights, remainder)
    for key in RAW.values():
        block[key] = r_squared(predicted[key], recorded[key])
    return block


def normalized(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def criteria(submitted, refit, target=None):
    out = {}
    for pid, key in RAW.items():
        a = normalized((submitted or {}).get(key))
        b = normalized((refit or {}).get(key))
        out[pid] = round(0.5 * (a + b), 6)
    return out


def score_of(verdicts):
    values = [float(v) for v in verdicts.values()]
    return float(np.mean(values)) if values else 0.0


def subset(props, sel):
    out = dict(props)
    for key in ("auditory", "visual", "baseline"):
        out[key] = props[key][sel]
    return out


def target(props, truth, specs):
    a, v, base, x = response_design(props)
    weights, remainder = location_solver(x)(truth)
    stats = location_statistics(a, v, base)
    block = {}
    for key, value in location_profiles(weights, remainder).items():
        fitted, *_ = np.linalg.lstsq(stats, value, rcond=None)
        block[key] = r_squared(stats @ fitted, value)
    block["n_terms"] = int(stats.shape[1])
    return block


def load_test_set():
    d = np.load(TEST_SET_FILE, allow_pickle=False)
    specs = json.loads(str(d["specs"]))
    props = {"t": d["t"].astype(np.float64),
             "auditory": d["auditory"].astype(np.float64),
             "visual": d["visual"].astype(np.float64),
             "baseline": d["baseline"].astype(np.float64)}
    return props, d["truth"].astype(np.float64), specs, d["panel"]

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_response", "fit_coeffs")
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
    if not (protocol.TEST_SET_FILE.is_file() and protocol.TEST_SET_MANIFEST.is_file()):
        raise RuntimeError(f"no test set at {protocol.TEST_SET_FILE}")
    with open(protocol.TEST_SET_MANIFEST) as f:
        manifest = json.load(f)
    want = protocol.fingerprint(seed)
    if manifest.get("fingerprint") != want:
        raise RuntimeError(
            f"the test set on disk was built for configuration "
            f"{manifest.get('fingerprint')}, this scoring expects {want}. Rebuild it and "
            f"rescore every run scored against the old one.")
    props, truth, specs = protocol.load_test_set()[:3]
    return props, truth, specs, manifest


def timed(label, fn, timeout, *args):
    settings = (np.asarray(args[0]["visual"]).shape[0]
                if args and isinstance(args[0], dict) else 0)
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


def score_block(mech, coeffs, props, truth, specs, label):
    start = time.time()
    pred = np.asarray(timed(label, mech.predict_response, PREDICT_TIMEOUT,
                            props, list(coeffs)), dtype=np.float64)
    seconds = time.time() - start
    block = protocol.measure(pred, truth, props, specs)
    block["predict_seconds"] = round(seconds, 2)
    block["n_coeffs"] = len(coeffs)
    return block


def score_refit(mech, coeffs, props, truth, specs):
    half = len(specs) // 2
    fit_sel, test_sel = np.arange(half), np.arange(half, len(specs))
    start = time.time()
    refit = [float(c) for c in timed(
        "fit_coeffs (refit half)", mech.fit_coeffs, FIT_TIMEOUT,
        protocol.subset(props, fit_sel), truth[fit_sel], list(coeffs))]
    seconds = time.time() - start
    block = score_block(mech, refit, protocol.subset(props, test_sel), truth[test_sel],
                        [specs[i] for i in test_sel], "predict_response (refit)")
    block["fit_seconds"] = round(seconds, 2)
    block["refit_coeffs"] = [round(c, 8) for c in refit]
    return block


def target_from(manifest, props, truth, specs):
    top = (manifest or {}).get("target") or {}
    if all(top.get(key) is not None for key in protocol.RAW.values()):
        return top, False
    print("[eval] the target in the manifest predates these criteria; remeasuring it "
          "from the recordings", flush=True)
    return protocol.target(props, truth, specs), True


def reference_context(manifest):
    context = {"correct_mechanism_target": (manifest or {}).get("target"),
               "source": str(protocol.TEST_SET_MANIFEST)}
    if protocol.REFERENCE_REPORT.is_file():
        with open(protocol.REFERENCE_REPORT) as f:
            report = json.load(f)
        context.update({"reference_mechanism": report.get("submitted"),
                        "reference_criteria": report.get("pa_scores"),
                        "reference_predictive_accuracy":
                            report.get("predictive_accuracy"),
                        "reference_source": str(protocol.REFERENCE_REPORT)})
    return context


def write_report(output, report, started):
    scores = report["pa_scores"]
    report["predictive_accuracy"] = round(protocol.score_of(scores), 6)
    report["n_criteria"] = len(protocol.PA_IDS)
    report["wall_clock_seconds"] = round(time.time() - started, 1)
    path = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    context = report.get("reference") or {}
    target = context.get("correct_mechanism_target") or {}
    ref = context.get("reference_mechanism") or {}
    sub = report.get("submitted") or {}
    refit = report.get("refit") or {}

    def cell(value):
        return f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{'-':>10}"

    print("=" * 108)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")
    print(f"  {'':62}{'submitted':>10}{'refit':>10}{'target':>10}{'human':>10}{'score':>8}")
    for pid, key in protocol.RAW.items():
        print(f"  {pid + ' ' + protocol.PA_LABELS[pid]:<62}{cell(sub.get(key))}"
              f"{cell(refit.get(key))}{cell(target.get(key))}{cell(ref.get(key))}"
              f"{scores.get(pid, 0.0):>8.3f}")
    print(f"  {'':62}{'-' * 48}")
    print(f"  {'predictive accuracy':<62}{'':>30}{'':>10}"
          f"{report['predictive_accuracy']:>8.3f}")
    if context.get("reference_predictive_accuracy") is not None:
        print(f"[eval] the human reference on the same test set: "
              f"{context['reference_predictive_accuracy']}   (context, never a level)")
    print(f"[eval] scores: {json.dumps(scores)}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']:.3f}  "
          f"(mean of {report['n_criteria']} continuous criteria, each in 0-1)")
    print(f"[eval] results: {path}")
    print("=" * 108)


def main(mechanism, output, seed=0, predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict_response {PREDICT_TIMEOUT:.0f}s, "
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
    report["reference"] = reference_context(manifest)
    top, remeasured = target_from(manifest, props, truth, specs)
    report["reference"]["correct_mechanism_target"] = top
    report["reference"]["target_remeasured"] = remeasured
    if not top:
        report["error"] = f"no target in {protocol.TEST_SET_MANIFEST}"
        write_report(output, report, started)
        return

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
        report["submitted"] = {"error": bad}
        report["refit"] = {"error": bad}
    if coeffs is not None:
        try:
            report["submitted"] = score_block(mech, coeffs, props, truth, specs,
                                              "predict_response")
        except Exception as exc:
            report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            report["refit"] = score_refit(mech, coeffs, props, truth, specs)
        except Exception as exc:
            report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.criteria(report.get("submitted"),
                                          report.get("refit"), top)
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
