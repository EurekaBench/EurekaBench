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

PROBLEM = "predictive_processing_of_language"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM
PA_IDS = ("PA1", "PA2", "PA3", "PA4")

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict_response", "fit_coeffs")
PROPS = ("t", "rate", "novelty", "scrambled")
MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy")
STATED_PREDICT_S = 120.0
STATED_FIT_S = 600.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
ACF_LAGS = np.arange(1, 51)


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
    if hasattr(module, "PROPERTIES"):
        unknown = sorted(set(map(str, module.PROPERTIES)) - set(PROPS))
        if unknown:
            problems.append(f"PROPERTIES names entries props does not hold: {unknown}")
    scan = ImportScan()
    with open(path) as f:
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS))
    if bad:
        problems.append(f"imports outside numpy and scipy: {bad}")
    return module, blocking, problems


def read_test_set():
    if not (protocol.TEST_SET_FILE.is_file() and protocol.TEST_SET_MANIFEST.is_file()):
        raise RuntimeError(f"no held-out set at {protocol.TEST_SET_FILE}")
    with open(protocol.TEST_SET_MANIFEST) as f:
        manifest = json.load(f)
    want = hashlib.sha256(json.dumps(manifest.get("config"), sort_keys=True)
                          .encode()).hexdigest()[:16]
    if manifest.get("fingerprint") != want:
        raise RuntimeError(f"{protocol.TEST_SET_MANIFEST} does not match the configuration "
                           f"it records")
    with np.load(protocol.TEST_SET_FILE) as d:
        arrays = {key: np.asarray(d[key]) for key in d.files}
    for key in ("t", "rate", "novelty", "scrambled", "response"):
        arrays[key] = arrays[key].astype(np.float64)
    return arrays, manifest


def splits(family, seed):
    inside = np.flatnonzero(family == 0)
    outside = np.flatnonzero(family == 1)
    order = np.random.default_rng(int(seed)).permutation(inside)
    half = len(order) // 2
    return {"inside": np.sort(inside), "outside": np.sort(outside),
            "fit": np.sort(order[:half]), "test": np.sort(order[half:])}


def props_of(arrays, rows):
    return {"t": arrays["t"].copy(), "rate": arrays["rate"][rows].copy(),
            "novelty": arrays["novelty"][rows].copy(),
            "scrambled": arrays["scrambled"][rows].copy()}


def pearson_acf(x, lags=ACF_LAGS):
    out = np.empty(x.shape[:-1] + (len(lags),))
    for j, k in enumerate(lags):
        a = x[..., :-k] - x[..., :-k].mean(-1, keepdims=True)
        b = x[..., k:] - x[..., k:].mean(-1, keepdims=True)
        num = (a * b).sum(-1)
        den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
        out[..., j] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return out


def context_score(pred, response, scrambled):
    recorded = response - scrambled
    variance = float(np.var(recorded))
    error = float(np.mean((pred - response) ** 2))
    ratio = error / variance if variance > 0 else float("inf")
    return {"score": max(0.0, 1.0 - ratio), "ratio": ratio, "mse": error,
            "variance": variance}


def slow_half(response, scrambled):
    persistence = pearson_acf(response - scrambled).mean(-1).mean(0)
    return np.sort(np.argsort(persistence, kind="stable")[len(persistence) // 2:])


def acf_score(pred, response, scrambled):
    base = pearson_acf(scrambled)
    predicted = pearson_acf(pred) - base
    recorded = pearson_acf(response) - base
    denominator = float(np.sum(recorded ** 2))
    ratio = (float(np.sum((predicted - recorded) ** 2)) / denominator
             if denominator > 0 else float("inf"))
    return {"score": max(0.0, 1.0 - ratio), "ratio": ratio}


def checked(pred, shape):
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != tuple(shape):
        raise ValueError(f"predict_response returned shape {pred.shape}, "
                         f"{tuple(shape)} expected")
    if not np.all(np.isfinite(pred)):
        raise ValueError("predict_response returned values that are not finite")
    return pred


def as_coeffs(values, reference):
    coeffs = [float(c) for c in values]
    if len(coeffs) != len(reference):
        raise ValueError(f"fit_coeffs returned {len(coeffs)} constants, "
                         f"{len(reference)} expected")
    if not all(np.isfinite(coeffs)):
        raise ValueError("fit_coeffs returned constants that are not finite")
    return coeffs


class Scorer:
    def __init__(self, mech, arrays):
        self.mech = mech
        self.arrays = arrays
        self.calls = []

    def timed(self, label, fn, timeout, stated, *args):
        print(f"[eval] {label}: budget {timeout:.0f}s ...", flush=True)
        start = time.time()
        try:
            out = call_with_timeout(fn, timeout, *args)
        except Exception as exc:
            seconds = time.time() - start
            self.calls.append({"call": label, "seconds": round(seconds, 2),
                               "error": f"{type(exc).__name__}: {exc}"})
            print(f"[eval] {label}: {type(exc).__name__} after {seconds:.1f}s: {exc}",
                  flush=True)
            raise
        seconds = time.time() - start
        self.calls.append({"call": label, "seconds": round(seconds, 2),
                           "within_stated_limit": bool(seconds <= stated)})
        print(f"[eval] {label}: done in {seconds:.1f}s", flush=True)
        return out

    def predict(self, label, rows, coeffs):
        props = props_of(self.arrays, rows)
        shape = props["scrambled"].shape
        out = self.timed(f"predict_response ({label}, {len(rows)} narratives)",
                         self.mech.predict_response, PREDICT_TIMEOUT, STATED_PREDICT_S,
                         props, list(coeffs))
        return checked(out, shape)

    def fit(self, label, rows, coeffs):
        props = props_of(self.arrays, rows)
        out = self.timed(f"fit_coeffs ({label}, {len(rows)} narratives)",
                         self.mech.fit_coeffs, FIT_TIMEOUT, STATED_FIT_S, props,
                         self.arrays["response"][rows].copy(), list(coeffs))
        return as_coeffs(out, coeffs)

    def recorded(self, rows):
        return self.arrays["response"][rows], self.arrays["scrambled"][rows]


def attempt(errors, key, fn):
    try:
        return fn()
    except Exception as exc:
        errors[key] = f"{type(exc).__name__}: {exc}"
        return None


def score_constants(scorer, label, coeffs, read_rows, outside_rows):
    block, errors = {"coeffs": [round(c, 8) for c in coeffs]}, {}
    pred = attempt(errors, "read", lambda: scorer.predict(label, read_rows, coeffs))
    if pred is not None:
        response, scrambled = scorer.recorded(read_rows)
        block["PA1"] = context_score(pred, response, scrambled)
        slow = slow_half(response, scrambled)
        block["PA2"] = context_score(pred[:, slow], response[:, slow], scrambled[:, slow])
        block["PA2"]["n_locations"] = int(len(slow))
        block["PA3"] = acf_score(pred, response, scrambled)
    outside = attempt(errors, "outside",
                      lambda: scorer.predict(f"{label}, outside", outside_rows, coeffs))
    if outside is not None:
        response, scrambled = scorer.recorded(outside_rows)
        block["PA4"] = context_score(outside, response, scrambled)
    block["scores"] = {pid: float((block.get(pid) or {}).get("score", 0.0))
                       for pid in PA_IDS}
    if errors:
        block["errors"] = errors
    return block


def main(mechanism, output, seed=0, predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict_response {PREDICT_TIMEOUT:.0f}s, fit_coeffs "
          f"{FIT_TIMEOUT:.0f}s (stated limits {STATED_PREDICT_S:.0f}s and "
          f"{STATED_FIT_S:.0f}s)", flush=True)
    started = time.time()
    mech_path = os.path.abspath(mechanism)
    report = {"problem": PROBLEM, "seed": int(seed), "mechanism": mech_path,
              "pa_scores": {pid: 0.0 for pid in PA_IDS}, "predictive_accuracy": 0.0,
              "problems": []}

    arrays, manifest = read_test_set()
    groups = splits(arrays["family"], seed)
    report["test_set"] = {"fingerprint": manifest["fingerprint"],
                          "config": manifest.get("config"),
                          "families": manifest.get("families"),
                          "shape": list(arrays["response"].shape),
                          "n_inside": int(len(groups["inside"])),
                          "n_outside": int(len(groups["outside"])),
                          "n_fit": int(len(groups["fit"])),
                          "n_test": int(len(groups["test"]))}
    if not os.path.isfile(mech_path):
        report["error"] = f"{mech_path} does not exist"
        finish(output, report, started)
        return

    try:
        mech, blocking, problems = load_mechanism(mech_path)
    except Exception:
        mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
    report["problems"] = problems
    if blocking:
        report["error"] = "; ".join(blocking)
        finish(output, report, started)
        return
    try:
        shipped = [float(c) for c in mech.COEFFS]
    except (TypeError, ValueError) as exc:
        report["error"] = f"COEFFS is not a sequence of numbers: {exc}"
        finish(output, report, started)
        return

    scorer = Scorer(mech, arrays)
    report["submitted"] = score_constants(scorer, "submitted", shipped, groups["inside"],
                                          groups["outside"])
    refit, errors = None, {}
    refit = attempt(errors, "fit", lambda: scorer.fit("refit", groups["fit"], shipped))
    if refit is not None:
        report["refit"] = score_constants(scorer, "refit", refit, groups["test"],
                                          groups["outside"])
    else:
        report["refit"] = {"errors": errors, "scores": {pid: 0.0 for pid in PA_IDS}}

    report["pa_scores"] = {pid: round(0.5 * (report["submitted"]["scores"][pid]
                                             + report["refit"]["scores"][pid]), 6)
                           for pid in PA_IDS}
    report["predictive_accuracy"] = round(float(np.mean(list(report["pa_scores"].values()))),
                                          6)
    report["calls"] = scorer.calls
    report["within_stated_limits"] = all(c.get("within_stated_limit", False)
                                         for c in scorer.calls)
    finish(output, report, started)


def finish(output, report, started):
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
    for name in ("submitted", "refit"):
        for key, error in ((report.get(name) or {}).get("errors") or {}).items():
            print(f"[eval] {name} {key}: {error}")
    sub = (report.get("submitted") or {}).get("scores") or {}
    ref = (report.get("refit") or {}).get("scores") or {}
    print(f"  {'':6}{'submitted':>12}{'refit':>12}{'score':>12}")
    for pid in PA_IDS:
        cells = "".join(f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{'-':>12}"
                        for v in (sub.get(pid), ref.get(pid), report["pa_scores"][pid]))
        print(f"  {pid:6}{cells}")
    print(f"[eval] predictive accuracy: {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    if "within_stated_limits" in report:
        print(f"[eval] every call within the stated time limits: "
              f"{report['within_stated_limits']}")
    print(f"[eval] results: {path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
