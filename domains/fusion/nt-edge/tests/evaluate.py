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

import coupled

PROBLEM = "nt_edge"

PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_SET_DIR = PACKAGE_ROOT / "test_set"
TEST_SET_FILE = TEST_SET_DIR / "heldout.npz"
TEST_SET_MANIFEST = TEST_SET_DIR / "manifest.json"
PROTOCOL_VERSION = 3
REFERENCE = dict(R0=1.67, Z0=0.0, a=0.56, kappa=1.6, delta=-0.5, delta_lower=-0.5,
                 kappa_lower=1.6, machine_scale=1.0, Ip=0.9e6, B0=2.0, n_e_ped=3.0e19,
                 P_heat=6.0e6, Z_eff=2.0, impurity="C", T_sep=100.0)
UNSEEN_SCALE = 1.5
AXES = {"Ip": (0.6e6, 1.2e6), "B0": (1.6, 2.2), "n_e_ped": (2.2e19, 3.6e19),
        "P_heat": (3.0e6, 12.0e6), "Z_eff": (1.6, 2.4), "kappa": (1.45, 1.6),
        "delta_abs": (0.3, 0.55)}
TRIANGULARITY_STEPS = (-0.55, -0.33, -0.11, 0.11, 0.33, 0.55)
POWER_STEPS = {"reversed": (1.5e6, 3.0e6, 6.0e6, 11.0e6, 18.0e6),
               "dshape": (0.3e6, 0.6e6, 1.5e6, 4.0e6, 9.0e6)}
DENSITY_STEPS = (2.0e19, 2.6e19, 3.2e19, 3.8e19)
RADIATION_STEPS = (1.6, 2.2, 2.8)
UNSEEN_POWERS = (6.0e6, 10.0e6, 16.0e6, 24.0e6)
FAMILIES = {"mirrored_pair": 12, "half_pair": 8, "triangularity_scan": 4, "power_scan": 6,
            "density_scan": 3, "radiation_scan": 2, "unseen_machine": 8}
PA_IDS = [f"PA{i}" for i in range(1, 9)]
PA_MEASURES = (("PA1", "barrier_pairs"), ("PA2", "barrier_triangularity"),
               ("PA3", "barrier_scans"), ("PA4", "alpha_edge_r2"), ("PA5", "log_lambda_q_r2"),
               ("PA6", "log_q_peak_r2"), ("PA7", "tau_imp_r2"), ("PA8", "barrier_unseen"))
TRUTH_KEYS = ("barrier", "alpha_edge", "lambda_q", "q_peak", "tau_imp")
PROPS_PROFILE = ("rho", "theta", "R", "Z", "B", "B_pol", "q", "psi", "volume", "vpr", "area",
                 "grad_rho", "grad_rho_sq", "n_e", "R_grid", "Z_grid", "psi_rz", "psi_n_grid",
                 "F", "boundary", "limiter")
PROPS_SCALAR = ("psi_axis", "psi_lcfs", "R_axis", "Z_axis", "P_heat", "P_rad", "B_0", "Ip",
                "Z_eff", "Z_imp", "A_i", "T_sep")
PROPS_KEYS = PROPS_PROFILE + PROPS_SCALAR


def fingerprint(seed):
    payload = json.dumps({"seed": seed, "reference": REFERENCE, "axes": AXES,
                          "triangularity": TRIANGULARITY_STEPS, "power": POWER_STEPS,
                          "density": DENSITY_STEPS, "radiation": RADIATION_STEPS,
                          "unseen": [UNSEEN_SCALE, UNSEEN_POWERS], "families": FAMILIES,
                          "closure": coupled.CLOSURE, "torax": coupled.TORAX,
                          "gpec": gpec_numerics(), "version": PROTOCOL_VERSION,
                          "props": list(PROPS_KEYS)},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


GPEC = dict(n_scan=32, mpsi=48, mtheta=256, max_alpha_scale=8.0)


def gpec_numerics():
    return dict(GPEC)


def pooled_r2(pred, truth):
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 3:
        return None
    residual = np.sum((pred[ok] - truth[ok]) ** 2)
    total = np.sum((truth[ok] - truth[ok].mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else None


def balanced_accuracy(pred, truth):
    pred = np.asarray(pred, dtype=float) > 0.5
    truth = np.asarray(truth, dtype=float) > 0.5
    if truth.all() or (~truth).all():
        return None
    tpr = float(np.mean(pred[truth]))
    tnr = float(np.mean(~pred[~truth]))
    return 0.5 * (tpr + tnr)


def rows_of(specs, families):
    return [s["index"] for s in specs if s["family"] in families]


def regime_r2(pred, truth, barrier):
    scores = []
    for regime in (0.0, 1.0):
        rows = np.flatnonzero(np.asarray(barrier) == regime)
        if rows.size >= 3:
            score = pooled_r2(np.asarray(pred)[rows], np.asarray(truth)[rows])
            if score is not None:
                scores.append(score)
    return float(np.mean(scores)) if scores else None


def measure(pred, truth, specs):
    pairs = rows_of(specs, ("mirrored_pair", "half_pair"))
    delta = rows_of(specs, ("triangularity_scan",))
    scans = rows_of(specs, ("power_scan", "density_scan", "radiation_scan"))
    unseen = rows_of(specs, ("unseen_machine",))
    pb, tb = np.asarray(pred["barrier"]), np.asarray(truth["barrier"])
    log = lambda v: np.log(np.clip(np.asarray(v, dtype=float), 1e-9, None))
    return {
        "barrier_pairs": balanced_accuracy(pb[pairs], tb[pairs]) if pairs else None,
        "barrier_triangularity": balanced_accuracy(pb[delta], tb[delta]) if delta else None,
        "barrier_scans": balanced_accuracy(pb[scans], tb[scans]) if scans else None,
        "alpha_edge_r2": regime_r2(pred["alpha_edge"], truth["alpha_edge"], tb),
        "log_lambda_q_r2": pooled_r2(log(pred["lambda_q"]), log(truth["lambda_q"])),
        "log_q_peak_r2": pooled_r2(log(pred["q_peak"]), log(truth["q_peak"])),
        "tau_imp_r2": regime_r2(pred["tau_imp"], truth["tau_imp"], tb),
        "barrier_unseen": balanced_accuracy(pb[unseen], tb[unseen]) if unseen else None,
        "n_settings": int(tb.size),
    }


def normalized(pid, value):
    if value is None or not np.isfinite(value):
        return 0.0
    if pid in ("PA1", "PA2", "PA3", "PA8"):
        return float(min(1.0, max(0.0, 2.0 * float(value) - 1.0)))
    return float(min(1.0, max(0.0, float(value))))


def pa_scores(block):
    return {pid: round(normalized(pid, (block or {}).get(key)), 4) for pid, key in PA_MEASURES}


def fit_split(specs):
    fit = []
    for s in specs:
        if s["family"] == "mirrored_pair":
            fit.append(s["pair_index"] % 2 == 0)
        elif s["family"] == "triangularity_scan":
            fit.append(s["scan_index"] % 2 == 0)
        elif s["family"] == "power_scan":
            fit.append(s["scan_index"] in (0, 4))
        else:
            fit.append(False)
    fit = np.asarray(fit, dtype=bool)
    return fit, ~fit


def subset(props, truth, specs, sel):
    rows = np.flatnonzero(np.asarray(sel))
    picked = [dict(specs[row], index=position, setting_index=specs[row]["index"])
              for position, row in enumerate(rows)]
    return ([props[row] for row in rows],
            {key: np.asarray(value)[rows] for key, value in truth.items()}, picked)


def load_test_set(out_dir=None):
    out = Path(out_dir) if out_dir else TEST_SET_DIR
    with open(out / TEST_SET_MANIFEST.name) as handle:
        manifest = json.load(handle)
    archive = np.load(out / TEST_SET_FILE.name)
    n = manifest["n_settings"]
    props = [{} for _ in range(n)]
    truth = {}
    for key in archive.files:
        parts = key.split("|")
        if parts[0] == "props":
            value = archive[key]
            props[int(parts[1])][parts[2]] = float(value) if value.ndim == 0 else value
        else:
            truth[parts[1]] = archive[key]
    return manifest, props, truth

protocol = sys.modules[__name__]


PROBLEM = protocol.PROBLEM

REQUIRED_MEMBERS = ("PROPERTIES", "COEFFS", "predict", "fit_coeffs")
MAX_COEFFS = 12
ALLOWED_IMPORTS = ("numpy", "scipy")
STATED_PREDICT_S = 120.0
STATED_FIT_S = 600.0
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
QUANTITIES = protocol.TRUTH_KEYS


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
    if hasattr(module, "COEFFS"):
        try:
            n_coeffs = len(list(module.COEFFS))
        except TypeError:
            n_coeffs = None
            problems.append("COEFFS is not a sequence")
        if n_coeffs is not None and n_coeffs > MAX_COEFFS:
            problems.append(f"COEFFS has {n_coeffs} entries, limit {MAX_COEFFS}")
    if hasattr(module, "PROPERTIES"):
        try:
            unknown = sorted(set(str(p) for p in module.PROPERTIES) - set(protocol.PROPS_KEYS))
        except TypeError:
            unknown = None
            problems.append("PROPERTIES is not a sequence")
        if unknown:
            problems.append(f"PROPERTIES names entries props does not carry: {unknown}")
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
    manifest, props, truth = protocol.load_test_set()
    want = protocol.fingerprint(seed)
    if manifest.get("fingerprint") != want:
        raise RuntimeError(
            f"the test set on disk was built for configuration "
            f"{manifest.get('fingerprint')}, this scoring expects {want}. Rebuild it "
            f"and rescore every run scored against the old one.")
    if list(manifest.get("props_keys", [])) != list(protocol.PROPS_KEYS):
        raise RuntimeError("the test set on disk carries other props entries than the task states now")
    return manifest, props, truth, manifest["specs"]


def truth_dict(truth):
    return {key: np.asarray(truth[key], dtype=np.float64) for key in QUANTITIES}


class PropsView(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return np.asarray([setting[key] for setting in self])
        got = list.__getitem__(self, key)
        return PropsView(got) if isinstance(key, slice) else got

    def keys(self):
        return list(list.__getitem__(self, 0).keys()) if len(self) else []

    def __contains__(self, key):
        if isinstance(key, str):
            return bool(len(self)) and key in list.__getitem__(self, 0)
        return list.__contains__(self, key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __reduce__(self):
        return (PropsView, (list(self),))


class CoeffsView(list):
    def __init__(self, values=()):
        self.names = list(values.keys()) if isinstance(values, dict) else []
        list.__init__(self, [float(v) for v in
                             (values.values() if isinstance(values, dict) else values)])

    def __getitem__(self, key):
        if isinstance(key, str):
            return list.__getitem__(self, self.names.index(key))
        got = list.__getitem__(self, key)
        return self.respan(got) if isinstance(key, slice) else got

    def __reduce__(self):
        return (CoeffsView, (dict(zip(self.names, self)) if self.names else list(self),))

    def named(self):
        if not self.names:
            raise AttributeError("COEFFS was written as a sequence and carries no names")
        return self.names

    def keys_or_none(self):
        return list(self.names) or None

    def keys(self):
        return list(self.named())

    def values(self):
        self.named()
        return [float(v) for v in list.__iter__(self)]

    def items(self):
        return list(zip(self.named(), self.values()))

    def get(self, key, default=None):
        if not self.names or key not in self.names:
            return default
        return list.__getitem__(self, self.names.index(key))

    def __contains__(self, key):
        if isinstance(key, str) and self.names:
            return key in self.names
        return list.__contains__(self, key)

    def respan(self, values):
        out = CoeffsView(list(values))
        out.names = self.names[:len(out)]
        return out


def coeffs_arg(coeffs):
    return coeffs if isinstance(coeffs, CoeffsView) else CoeffsView(coeffs)


def predict(mech, props, coeffs, label="predict"):
    start = time.time()
    raw = timed(label, len(props), mech.predict, PREDICT_TIMEOUT,
                PropsView(props), coeffs_arg(coeffs))
    if not isinstance(raw, dict):
        raise RuntimeError(f"predict returned {type(raw).__name__}, the task states a dict")
    missing = [key for key in QUANTITIES if key not in raw]
    if missing:
        raise RuntimeError(f"predict returned no {missing}")
    pred = {}
    for key in QUANTITIES:
        values = np.asarray(raw[key], dtype=np.float64).reshape(-1)
        if values.shape != (len(props),):
            raise RuntimeError(f"predict returned shape {values.shape} for {key}, the task "
                               f"states ({len(props)},)")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"predict returned a non-finite value for {key}")
        pred[key] = values
    return pred, time.time() - start


def score_submitted(mech, coeffs, props, truth, specs):
    fit_sel, score_sel = protocol.fit_split(specs)
    fit_props, fit_truth, fit_specs = protocol.subset(props, truth, specs, fit_sel)
    score_props, score_truth, score_specs = protocol.subset(props, truth, specs,
                                                            score_sel)
    start = time.time()
    refit = timed("fit_coeffs", len(fit_props), mech.fit_coeffs, FIT_TIMEOUT,
                  PropsView(fit_props), truth_dict(fit_truth), coeffs_arg(coeffs))
    fitted = CoeffsView(refit)
    if not fitted.keys_or_none():
        fitted = coeffs_arg(coeffs).respan(fitted)
    fit_seconds = time.time() - start
    pred, seconds = predict(mech, score_props, fitted)
    block = protocol.measure(pred, truth_dict(score_truth), score_specs)
    block["fit_seconds"] = round(fit_seconds, 2)
    block["fit_within_stated_limit"] = bool(fit_seconds <= STATED_FIT_S)
    block["predict_seconds"] = round(seconds, 2)
    block["predict_within_stated_limit"] = bool(seconds <= STATED_PREDICT_S)
    block["n_coeffs"] = len(coeffs)
    block["refit_coeffs"] = [round(c, 8) for c in fitted]
    block["n_fit_settings"] = len(fit_props)
    block["n_scored_settings"] = len(score_props)
    return block


def independence_check(mech, coeffs, props, truth, specs, row=0):
    fit_sel, score_sel = protocol.fit_split(specs)
    score_props, _, _ = protocol.subset(props, truth, specs, score_sel)
    whole, _ = predict(mech, score_props, coeffs, "predict (batch)")
    alone, _ = predict(mech, score_props[row:row + 1], coeffs, "predict (one setting)")
    return bool(all(np.allclose(whole[key][row], alone[key][0]) for key in QUANTITIES))


def write_report(output, report, started):
    scores = report.get("pa_scores") or {}
    report["predictive_accuracy"] = (round(sum(scores.values()) / len(scores), 4)
                                     if scores else 0.0)
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
    sub = report.get("submitted") or {}

    def cell(value):
        return f"{value:>13.4f}" if isinstance(value, (int, float)) else f"{'-':>13}"

    print(f"  {'':34}{'measure':>13}{'score':>13}")
    for pid, key in protocol.PA_MEASURES:
        print(f"  {pid + ' ' + key:34}{cell(sub.get(key))}{cell(scores.get(pid))}")
    print(f"  {'predictive accuracy':34}{'':>13}{cell(report['predictive_accuracy'])}")
    print()
    print(f"[eval] predictive accuracy: {json.dumps(scores)} -> "
          f"{report['predictive_accuracy']}")
    print(f"[eval] results: {path}")
    print("=" * 72)


def main(mechanism, output, seed=0, predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s, fit {FIT_TIMEOUT:.0f}s", flush=True)
    started = time.time()
    mech_path = str(mechanism)
    report = {"problem": PROBLEM, "seed": seed,
              "pa_scores": {pid: 0.0 for pid in protocol.PA_IDS}, "problems": []}
    if not os.path.isfile(mech_path):
        report["error"] = f"{mech_path} does not exist"
        write_report(output, report, started)
        return

    manifest, props, truth, specs = read_test_set(seed)
    report["test_set"] = {k: v for k, v in manifest.items() if k != "specs"}

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
        coeffs = CoeffsView(mech.COEFFS)
    except (TypeError, ValueError, AttributeError) as exc:
        report["error"] = f"COEFFS is not numbers: {exc}"
        write_report(output, report, started)
        return
    try:
        report["submitted"] = score_submitted(mech, coeffs, props, truth, specs)
    except Exception as exc:
        report["submitted"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["setting_independent"] = independence_check(mech, coeffs, props, truth,
                                                           specs)
    except Exception as exc:
        report["setting_independent"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["pa_scores"] = protocol.pa_scores(report["submitted"])
    write_report(output, report, started)


if __name__ == "__main__":
    fire.Fire(main)
