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

PROBLEM = "visual_object_properties"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
REFERENCE_DIR = PACKAGE_ROOT / "reference_mechanism"
REFERENCE_REPORT = REFERENCE_DIR / "reference_check.json"
ALLOWED_IMPORTS = ("numpy", "scipy")
MAX_COEFFS = 10
PROPERTIES = {
    "animacy": ("property_lives_mean",),
    "real_world_size": ("size_mean",),
    "naturalness": ("property_natural_mean",),
    "preciousness": ("property_precious_mean",),
    "ability_to_move": ("property_moves_mean",),
    "ability_to_be_moved": ("property_be-moved_mean",),
    "heaviness": ("property_heavy_mean",),
    "pleasantness": ("property_pleasant_mean",),
    "arousal": ("property_arousal_mean",),
    "manipulability": ("property_grasp_mean", "property_hold_mean"),
    "cuteness": ("cuteness",),
}
ANIMACY = "animacy"
ANIMATE_ABOVE = 4.0
N_REFERENCE = 100
N_HELDOUT = 100
ANIMATE_SHARE = 0.4
IMAGES_PER_OBJECT = 3
IMAGE_S = 3.0
FPS = 8
FRAME = 224
STIMULUS_VERSION = 1
PANEL_GROUPS = ("Ventral Stream Visual Cortex", "MT+ Complex and Neighboring Visual Areas")
PANEL_AREAS = ("PHA1", "PHA2", "PHA3")
PA_IDS = ["PA1", "PA2", "PA3"]
PA_KEYS = {"PA1": "all", "PA2": "animate", "PA3": "inanimate"}
PA_LABELS = {"PA1": "median R2, all held-out objects",
             "PA2": "median R2, animate held-out objects",
             "PA3": "median R2, inanimate held-out objects"}
PROTOCOL_VERSION = 2


def stimulus_config():
    return {"images_per_object": IMAGES_PER_OBJECT, "image_s": IMAGE_S, "fps": FPS,
            "frame": FRAME, "transcribe": False, "stimulus_version": STIMULUS_VERSION}


def build_config(seed):
    return {"seed": int(seed), "n_reference": N_REFERENCE, "n_heldout": N_HELDOUT,
            "animate_share": ANIMATE_SHARE, "animate_above": ANIMATE_ABOVE,
            "properties": {k: list(v) for k, v in PROPERTIES.items()},
            "stimulus": stimulus_config(), "panel_groups": list(PANEL_GROUPS),
            "panel_areas": list(PANEL_AREAS), "protocol_version": PROTOCOL_VERSION}


def fingerprint(seed):
    payload = json.dumps(build_config(seed), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def location_r2(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or truth.ndim != 2 or truth.shape[0] < 3 \
            or not np.all(np.isfinite(pred)):
        return None
    residual = ((pred - truth) ** 2).sum(axis=0)
    total = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    out = np.full(truth.shape[1], np.nan)
    varies = total > 1e-12
    out[varies] = 1.0 - residual[varies] / total[varies]
    return out


def median_r2(pred, truth, rows=None):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if rows is not None:
        if pred.ndim != 2 or pred.shape[0] != truth.shape[0]:
            return None
        pred, truth = pred[rows], truth[rows]
    r2 = location_r2(pred, truth)
    if r2 is None or not np.isfinite(r2).any():
        return None
    return float(np.median(r2[np.isfinite(r2)]))


def animate_rows(properties, names):
    return np.flatnonzero(np.asarray(properties)[:, list(names).index(ANIMACY)] > ANIMATE_ABOVE)


def inanimate_rows(properties, names):
    return np.flatnonzero(np.asarray(properties)[:, list(names).index(ANIMACY)] <= ANIMATE_ABOVE)


def scores_of(pred, props, truth):
    names = [str(n) for n in props["property_names"]]
    pred = np.asarray(pred, dtype=np.float64)
    return {"all": median_r2(pred, truth),
            "animate": median_r2(pred, truth, animate_rows(props["properties"], names)),
            "inanimate": median_r2(pred, truth, inanimate_rows(props["properties"], names))}


def floored(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def run_scores(block):
    return {"PA1": floored(block.get("all")), "PA2": floored(block.get("animate")),
            "PA3": floored(block.get("inanimate"))}


def criteria(submitted, refit):
    a, b = run_scores(submitted or {}), run_scores(refit or {})
    return {pid: round(0.5 * (a[pid] + b[pid]), 6) for pid in PA_IDS}


def subset(props, rows):
    out = dict(props)
    out["properties"] = np.asarray(props["properties"])[rows]
    return out


def score_of(scores):
    return float(np.mean([float(scores.get(pid) or 0.0) for pid in PA_IDS]))


def load_test_set(out_dir=TEST_SET_DIR):
    d = np.load(Path(out_dir) / TEST_SET_FILE.name, allow_pickle=False)
    props = {"property_names": d["property_names"],
             "properties": d["properties"].astype(np.float64),
             "reference_properties": d["reference_properties"].astype(np.float64),
             "reference_response": d["reference_response"].astype(np.float64)}
    info = {"heldout_ids": [str(u) for u in d["heldout_ids"]],
            "reference_ids": [str(u) for u in d["reference_ids"]], "panel": d["panel"]}
    return props, d["truth"].astype(np.float64), info

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_response", "fit_coeffs")
MAX_COEFFS = protocol.MAX_COEFFS
ALLOWED_IMPORTS = protocol.ALLOWED_IMPORTS
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
    props, truth = protocol.load_test_set(protocol.TEST_SET_DIR)[:2]
    return props, truth, manifest


def timed(label, fn, timeout, *args):
    objects = (np.asarray(args[0]["properties"]).shape[0]
               if args and isinstance(args[0], dict) else 0)
    print(f"[eval] {label}: {objects} objects, budget {timeout:.0f}s ...", flush=True)
    start = time.time()
    try:
        out = call_with_timeout(fn, timeout, *args)
    except Exception as exc:
        print(f"[eval] {label}: {type(exc).__name__} after {time.time() - start:.1f}s",
              flush=True)
        raise
    print(f"[eval] {label}: done in {time.time() - start:.1f}s", flush=True)
    return out


def score_block(mech, coeffs, props, truth, label):
    start = time.time()
    pred = np.asarray(timed(label, mech.predict_response, PREDICT_TIMEOUT,
                            props, list(coeffs)), dtype=np.float64)
    seconds = time.time() - start
    block = protocol.scores_of(pred, props, truth)
    block["prediction_shape"] = list(pred.shape)
    block["expected_shape"] = list(truth.shape)
    block["predict_seconds"] = round(seconds, 2)
    block["n_coeffs"] = len(coeffs)
    return block


def score_refit(mech, coeffs, props, truth):
    half = truth.shape[0] // 2
    fit_rows, test_rows = np.arange(half), np.arange(half, truth.shape[0])
    start = time.time()
    refit = [float(c) for c in timed(
        "fit_coeffs (refit half)", mech.fit_coeffs, FIT_TIMEOUT,
        protocol.subset(props, fit_rows), truth[fit_rows], list(coeffs))]
    seconds = time.time() - start
    block = score_block(mech, refit, protocol.subset(props, test_rows), truth[test_rows],
                        "predict_response (refit)")
    block["fit_seconds"] = round(seconds, 2)
    block["refit_coeffs"] = [round(c, 8) for c in refit]
    return block


def reference_context(manifest):
    context = {"encoding_model": (manifest or {}).get("encoding_model_context"),
               "source": str(protocol.TEST_SET_MANIFEST)}
    if protocol.REFERENCE_REPORT.is_file():
        with open(protocol.REFERENCE_REPORT) as f:
            report = json.load(f)
        if report.get("test_set_fingerprint") == (manifest or {}).get("fingerprint"):
            context.update({"reference_mechanism": report.get("submitted"),
                            "reference_mechanism_refit": report.get("refit"),
                            "reference_criteria": report.get("pa_scores"),
                            "reference_predictive_accuracy":
                                report.get("predictive_accuracy"),
                            "reference_source": str(protocol.REFERENCE_REPORT)})
        else:
            print(f"[eval] {protocol.REFERENCE_REPORT} was scored on another test set; "
                  f"left out of the context", flush=True)
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
    human = context.get("reference_criteria") or {}
    sub = report.get("submitted") or {}
    refit = report.get("refit") or {}

    def cell(value):
        return f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{'-':>10}"

    print("=" * 96)
    if report.get("error"):
        print(f"[eval] {report['error']}")
    for problem in report.get("problems") or []:
        print(f"[eval] contract problem: {problem}")
    for label, block in (("submitted", sub), ("refit", refit)):
        if block.get("error"):
            print(f"[eval] the {label} run failed: {block['error']}")
    print(f"  {'':50}{'submitted':>10}{'refit':>10}{'score':>10}{'human':>10}")
    for pid, key in protocol.PA_KEYS.items():
        print(f"  {pid + ' ' + protocol.PA_LABELS[pid]:<50}{cell(sub.get(key))}"
              f"{cell(refit.get(key))}{cell(scores.get(pid))}{cell(human.get(pid))}")
    print(f"  {'':50}{'-' * 40}")
    print(f"  {'predictive accuracy':<50}{'':>20}{cell(report['predictive_accuracy'])}"
          f"{cell(context.get('reference_predictive_accuracy'))}")
    print(f"[eval] scores: {json.dumps(scores)}")
    print(f"[eval] predictive accuracy: {report['predictive_accuracy']:.3f}  "
          f"(mean of {report['n_criteria']} continuous criteria, each in 0-1; the human "
          f"column is the reference mechanism on the same test set, context only)")
    print(f"[eval] results: {path}")
    print("=" * 96)


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

    props, truth, manifest = read_test_set(seed)
    report["test_set"] = manifest
    report["reference"] = reference_context(manifest)

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
            report["submitted"] = score_block(mech, coeffs, props, truth,
                                              "predict_response")
        except Exception as exc:
            report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            report["refit"] = score_refit(mech, coeffs, props, truth)
        except Exception as exc:
            report["refit"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.criteria(report.get("submitted"), report.get("refit"))
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
