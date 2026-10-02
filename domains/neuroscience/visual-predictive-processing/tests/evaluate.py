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

PROBLEM = "visual_predictive_processing"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
REFERENCE_DIR = PACKAGE_ROOT / "reference_mechanism"
REFERENCE_REPORT = REFERENCE_DIR / "reference_check.json"
DURATION_S = 10.0
SEGMENT_S = 10.0
GUARD_S = 2.0
FPS = 8.0
FRAME = 224
N_SETTINGS = 60
N_PANEL = 600
PROTOCOL_VERSION = 2
PROPS_VERSION = 2
NOISE_SEED = 20260327
CARRIER_CONTRAST = 0.25
STIMULUS_CONTRAST = 0.75
SURROUND_GAIN = 0.5
GRATING_CYCLES = 7.0
DRIFT_HZ = 1.0
CONTOUR_KINDS = ("line", "arc", "vee")
CONTOUR_RADIUS = 0.9
VEE_HALF = 0.6
N_NEIGHBOURS = 1
ELEMENT_SPACING = 0.69
BLOB_LONG = 0.135
BLOB_SHORT = 0.0675
DISPLACE = 0.135
MIN_NEIGHBOURS = 2
PATCH_RADIUS = 0.39
ECCENTRICITY = 0.45
POLAR_ANGLES = (90.0, 210.0, 330.0)
FINE_AGREEMENT = (0.0, 0.5, 1.0)
COARSE_AGREEMENT = (0.0, 1.0)
STIMULUS_CONDITIONS = ("both", "surround", "part", "implied")
CONDITIONS = STIMULUS_CONDITIONS + ("base",)
PA_IDS = ["PA1", "PA2", "PA3"]
PA_LABELS = {
    "PA1": "the response to a part of a scene sitting inside its surround",
    "PA2": "how far a location is held back as the surround implies the part",
    "PA3": "which grain of agreement a location answers to",
}
RAW = {"PA1": "response", "PA2": "holding_back", "PA3": "grain_contrast"}


def fingerprint(seed, n_settings=N_SETTINGS):
    payload = json.dumps({
        "n_settings": n_settings, "seed": seed, "duration_s": DURATION_S,
        "segment_s": SEGMENT_S, "guard_s": GUARD_S, "fps": FPS, "frame": FRAME,
        "noise_seed": NOISE_SEED, "carrier_contrast": CARRIER_CONTRAST,
        "stimulus_contrast": STIMULUS_CONTRAST, "grating_cycles": GRATING_CYCLES,
        "drift_hz": DRIFT_HZ, "contour_kinds": CONTOUR_KINDS,
        "contour_radius": CONTOUR_RADIUS, "vee_half": VEE_HALF,
        "n_neighbours": N_NEIGHBOURS, "min_neighbours": MIN_NEIGHBOURS,
        "surround_gain": SURROUND_GAIN, "element_spacing": ELEMENT_SPACING,
        "blob_long": BLOB_LONG, "blob_short": BLOB_SHORT, "displace": DISPLACE,
        "patch_radius": PATCH_RADIUS, "eccentricity": ECCENTRICITY,
        "polar_angles": POLAR_ANGLES, "fine": FINE_AGREEMENT,
        "coarse": COARSE_AGREEMENT, "n_panel": N_PANEL,
        "protocol_version": PROTOCOL_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def test_set_fingerprint(seed, n_settings=N_SETTINGS):
    payload = json.dumps({"recordings": fingerprint(seed, n_settings),
                          "conditions": CONDITIONS, "props_version": PROPS_VERSION},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def usable(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    return pred.shape == truth.shape and bool(np.all(np.isfinite(pred)))


def interaction(activity, props):
    additive = (np.asarray(props["part"], dtype=np.float64)
                + np.asarray(props["surround"], dtype=np.float64)
                - np.asarray(props["baseline"], dtype=np.float64))
    return (np.asarray(activity, dtype=np.float64) - additive).mean(axis=2)


def agreement_design(specs):
    fine = np.array([s["fine"] for s in specs], dtype=np.float64)
    coarse = np.array([s["coarse"] for s in specs], dtype=np.float64)
    return np.stack([np.ones_like(fine), fine, coarse], axis=1)


def slopes(activity, props, specs):
    x = agreement_design(specs)
    y = interaction(activity, props)
    base = np.asarray(props["baseline"], dtype=np.float64)
    drives = np.stack([(np.asarray(props[k], dtype=np.float64) - base).mean(axis=2)
                       for k in ("part", "surround")], axis=2)
    if len(y) <= x.shape[1] + drives.shape[2] or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    inverse = np.linalg.pinv(x)
    fitted = inverse @ y
    through = np.einsum("kn,npj->kpj", inverse, drives)
    left = y - x @ fitted
    rest = drives - np.einsum("nk,kpj->npj", x, through)
    strength = np.einsum("pij,pj->pi", np.linalg.pinv(np.einsum("npi,npj->pij", rest, rest)),
                         np.einsum("npi,np->pi", rest, left))
    return (fitted[1] - (through[1] * strength).sum(axis=1),
            fitted[2] - (through[2] * strength).sum(axis=1))


def profiles(activity, props, specs):
    out = {}
    pair = slopes(activity, props, specs)
    if pair is None:
        return None
    fine, coarse = pair
    out["holding_back"] = -0.5 * (fine + coarse)
    out["grain_contrast"] = fine - coarse
    return {k: v - v.mean() for k, v in out.items()}


def across_settings(activity):
    activity = np.asarray(activity, dtype=np.float64)
    return activity - activity.mean(axis=0, keepdims=True)


def scored(activity, props, specs):
    got = profiles(activity, props, specs)
    out = {"response": across_settings(activity)}
    out.update({key: None if got is None else got[key]
                for key in ("holding_back", "grain_contrast")})
    return out


def trivial_predictions(props):
    base = np.asarray(props["baseline"], dtype=np.float64)
    part = np.asarray(props["part"], dtype=np.float64)
    surround = np.asarray(props["surround"], dtype=np.float64)
    return {"adding the two": part + surround - base, "copying the surround": surround,
            "copying the part": part}


def skill(pred, truth, references):
    truth = np.asarray(truth, dtype=np.float64)
    floor = min(float(((np.asarray(r, dtype=np.float64) - truth) ** 2).sum())
                for r in references)
    if floor <= 0:
        return None
    return float(1.0 - ((np.asarray(pred, dtype=np.float64) - truth) ** 2).sum() / floor)


def measure(pred, truth, props, specs):
    recorded = profiles(truth, props, specs)
    block = {"holding_back_recorded": (None if recorded is None
                                       else float(np.abs(recorded["holding_back"]).mean())),
             "response_scale": float(np.std(np.asarray(truth, dtype=np.float64)))}
    if not usable(pred, truth) or recorded is None:
        block.update({key: None for key in RAW.values()})
        return block
    predicted = scored(pred, props, specs)
    measured = scored(truth, props, specs)
    copied = [scored(c, props, specs) for c in trivial_predictions(props).values()]
    for key in RAW.values():
        a, b = predicted[key], measured[key]
        if a is None or b is None:
            block[key] = None
            continue
        references = [np.zeros_like(b)] + [c[key] for c in copied if c[key] is not None]
        block[key] = skill(a, b, references)
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
    for key in ("surround", "part", "implied", "baseline"):
        out[key] = props[key][sel]
    return out


MECHANISM_INPUTS = ("surround", "part", "implied", "baseline")


def mechanism_props(props, names):
    view = {"t": props["t"]}
    view.update({name: props[name] for name in names if name in MECHANISM_INPUTS})
    return view


def direct_call(label, fn, *args):
    return fn(*args)


def score_mechanism(mech, props, truth, specs, call=direct_call):
    names = [str(name) for name in mech.PROPERTIES]
    shipped = [float(c) for c in mech.COEFFS]
    out = {"shipped_coeffs": shipped}
    try:
        start = time.time()
        pred = call("predict_response", mech.predict_response,
                    mechanism_props(props, names), list(shipped))
        seconds = time.time() - start
        pred = np.asarray(pred, dtype=np.float64)
        out["submitted"] = measure(pred, truth, props, specs)
        out["submitted"]["predict_seconds"] = round(seconds, 2)
        out["submitted"]["prediction_shape"] = list(pred.shape)
        out["submitted"]["prediction_finite"] = bool(np.all(np.isfinite(pred)))
    except Exception as exc:
        out["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
    half = len(specs) // 2
    fit_sel, test_sel = np.arange(half), np.arange(half, len(specs))
    try:
        start = time.time()
        refit = [float(c) for c in call("fit_coeffs", mech.fit_coeffs,
                                        mechanism_props(subset(props, fit_sel), names),
                                        truth[fit_sel], list(shipped))]
        fit_seconds = time.time() - start
        out["refit_coeffs"] = refit
        test_props = subset(props, test_sel)
        start = time.time()
        pred = call("predict_response on the other half", mech.predict_response,
                    mechanism_props(test_props, names), list(refit))
        seconds = time.time() - start
        pred = np.asarray(pred, dtype=np.float64)
        out["refit"] = measure(pred, truth[test_sel], test_props,
                               [specs[i] for i in test_sel])
        out["refit"]["fit_seconds"] = round(fit_seconds, 2)
        out["refit"]["predict_seconds"] = round(seconds, 2)
        out["refit"]["prediction_shape"] = list(pred.shape)
        out["refit"]["prediction_finite"] = bool(np.all(np.isfinite(pred)))
    except Exception as exc:
        out["refit"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["pa_scores"] = criteria(out["submitted"], out["refit"])
    out["predictive_accuracy"] = round(score_of(out["pa_scores"]), 6)
    return out


def load_test_set():
    d = np.load(TEST_SET_FILE, allow_pickle=False)
    props = {"t": d["t"].astype(np.float64),
             "surround": d["surround"].astype(np.float64),
             "part": d["part"].astype(np.float64),
             "implied": d["implied"].astype(np.float64),
             "baseline": d["baseline"].astype(np.float64)}
    return props, d["truth"].astype(np.float64), json.loads(str(d["specs"])), d["panel"]

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_response", "fit_coeffs")
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
PREDICT_LIMIT = 120.0
FIT_LIMIT = 600.0
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
    if hasattr(module, "COEFFS"):
        try:
            n_coeffs = len(list(module.COEFFS))
        except TypeError:
            n_coeffs = None
            blocking.append("COEFFS is not a sequence")
        if n_coeffs is not None and n_coeffs > MAX_COEFFS:
            problems.append(f"COEFFS has {n_coeffs} entries, limit {MAX_COEFFS}")
    if hasattr(module, "PROPERTIES"):
        names = ([module.PROPERTIES] if isinstance(module.PROPERTIES, str)
                 else list(module.PROPERTIES))
        unknown = sorted({str(n) for n in names} - set(protocol.MECHANISM_INPUTS) - {"t"})
        if unknown:
            problems.append(f"PROPERTIES names entries props does not carry: {unknown}")
    scan = ImportScan()
    with open(path) as f:
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy: {bad}")
    return module, blocking, problems


def read_test_set(seed, n_settings):
    if not (protocol.TEST_SET_FILE.is_file() and protocol.TEST_SET_MANIFEST.is_file()):
        raise RuntimeError(f"no test set at {protocol.TEST_SET_FILE}")
    with open(protocol.TEST_SET_MANIFEST) as f:
        manifest = json.load(f)
    want = protocol.test_set_fingerprint(seed, n_settings)
    if manifest.get("fingerprint") != want:
        raise RuntimeError(
            f"the test set on disk was built for configuration "
            f"{manifest.get('fingerprint')}, this scoring expects {want}. Rebuild it "
            f"and rescore every run scored against the old one.")
    props, truth, specs = protocol.load_test_set()[:3]
    return props, truth, specs, manifest


def bounded_call(label, fn, *args):
    timeout = FIT_TIMEOUT if label.startswith("fit_coeffs") else PREDICT_TIMEOUT
    settings = next((np.asarray(v).shape[0] for k, v in args[0].items() if k != "t"), 0)
    print(f"[eval] {label}: {settings} settings, budget {timeout:.0f}s ...", flush=True)
    start = time.time()
    try:
        out = call_with_timeout(fn, timeout, *args)
    except Exception as exc:
        print(f"[eval] {label}: {type(exc).__name__} after "
              f"{time.time() - start:.1f}s", flush=True)
        raise
    print(f"[eval] {label}: done in {time.time() - start:.1f}s", flush=True)
    return out


def reference_context(fingerprint):
    if not protocol.REFERENCE_REPORT.is_file():
        return None
    with open(protocol.REFERENCE_REPORT) as f:
        report = json.load(f)
    if report.get("test_set_fingerprint") != fingerprint:
        return None
    return {"pa_scores": report.get("pa_scores"),
            "predictive_accuracy": report.get("predictive_accuracy"),
            "submitted": report.get("submitted"), "refit": report.get("refit"),
            "target": report.get("target"), "source": str(protocol.REFERENCE_REPORT)}


def limit_problems(report):
    problems = []
    sub, refit = report.get("submitted") or {}, report.get("refit") or {}
    for label, block, key, limit in (("predict_response", sub, "predict_seconds",
                                      PREDICT_LIMIT),
                                     ("fit_coeffs", refit, "fit_seconds", FIT_LIMIT)):
        seconds = block.get(key)
        if isinstance(seconds, (int, float)) and seconds > limit:
            problems.append(f"{label} took {seconds:.0f}s, over the {limit:.0f}s the task "
                            f"allows")
    return problems


def shape_problems(report, truth):
    problems = []
    half = truth.shape[0] - truth.shape[0] // 2
    for name, want in (("submitted", tuple(truth.shape)),
                       ("refit", (half,) + tuple(truth.shape[1:]))):
        block = report.get(name) or {}
        got = block.get("prediction_shape")
        if got is None:
            continue
        if tuple(got) != want:
            problems.append(f"predict_response returned shape {tuple(got)} in the {name} "
                            f"run, where the recording has {want}")
        elif not block.get("prediction_finite", True):
            problems.append(f"predict_response returned values that are not finite in "
                            f"the {name} run")
    return problems


def write_report(output, report, started):
    report["wall_clock_seconds"] = round(time.time() - started, 1)
    path = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("=" * 96)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")
    for name in ("submitted", "refit"):
        if (report.get(name) or {}).get("error"):
            print(f"[eval] {name} run failed: {report[name]['error']}")
    context = report.get("reference") or {}
    sub, refit = report.get("submitted") or {}, report.get("refit") or {}
    ref, target = context.get("pa_scores") or {}, context.get("target") or {}

    def cell(value):
        return f"{value:>11.4f}" if isinstance(value, (int, float)) else f"{'-':>11}"

    print(f"   {'':52}{'submitted':>11}{'refit':>11}{'score':>11}{'human ref':>11}"
          f"{'target':>11}")
    for pid, key in protocol.RAW.items():
        print(f"   {pid + ' ' + protocol.PA_LABELS[pid]:<52}{cell(sub.get(key))}"
              f"{cell(refit.get(key))}{cell(report['pa_scores'].get(pid))}"
              f"{cell(ref.get(pid))}{cell(target.get(key))}")
    print()
    print(f"[eval] predictive accuracy: {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    if ref:
        print(f"[eval] human reference    : {json.dumps(ref)} -> "
              f"{context.get('predictive_accuracy')}   (context, never a level to clear)")
    print(f"[eval] results: {path}")
    print("=" * 96)


def main(mechanism, output, seed=0, n_settings=protocol.N_SETTINGS, predict_timeout=None,
         fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict_response {PREDICT_TIMEOUT:.0f}s, "
          f"fit_coeffs {FIT_TIMEOUT:.0f}s", flush=True)
    started = time.time()
    mech_path = str(mechanism)
    report = {"problem": PROBLEM, "seed": int(seed), "mechanism": mech_path,
              "pa_scores": {pid: 0.0 for pid in protocol.PA_IDS},
              "predictive_accuracy": 0.0, "problems": []}

    props, truth, specs, manifest = read_test_set(seed, n_settings)
    report["test_set"] = {k: manifest.get(k) for k in (
        "fingerprint", "recording_fingerprint", "n_settings", "n_panel", "n_timesteps",
        "pa_ids", "scoring")}
    report["reference"] = reference_context(manifest.get("fingerprint"))
    if report["reference"] is None:
        print(f"[eval] WARNING: no reference measurement for this test set in "
              f"{protocol.REFERENCE_REPORT}; the human reference is reported as missing",
              flush=True)
    if not os.path.isfile(mech_path):
        report["error"] = f"{mech_path} does not exist"
        write_report(output, report, started)
        return

    try:
        mech, blocking, problems = load_mechanism(mech_path)
    except Exception:
        mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
    report["problems"] = problems
    if not blocking:
        try:
            [float(c) for c in mech.COEFFS]
        except (TypeError, ValueError) as exc:
            blocking.append(f"COEFFS is not a sequence of numbers: {exc}")
    if blocking:
        report["error"] = "; ".join(blocking)
        write_report(output, report, started)
        return

    scored = protocol.score_mechanism(mech, props, truth, specs, call=bounded_call)
    report.update({k: scored.get(k) for k in ("shipped_coeffs", "refit_coeffs",
                                               "submitted", "refit")})
    report["problems"] += limit_problems(report) + shape_problems(report, truth)
    report["pa_scores"] = scored["pa_scores"]
    report["predictive_accuracy"] = scored["predictive_accuracy"]
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
