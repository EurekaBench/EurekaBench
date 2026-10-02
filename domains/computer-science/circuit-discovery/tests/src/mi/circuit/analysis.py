import json
import math
from pathlib import Path

SPLIT = "test"
REFERENCE_METHOD = "EAP-IG-inputs"


def finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def norm_cmd(cmd, sota):
    return sota / (sota + max(float(cmd), 0.0))


def norm_cpr(cpr, sota):
    return float(cpr) / (float(cpr) + sota) if float(cpr) > 0 else 0.0


def read_metric(eval_dir, inst, metric, split=SPLIT):
    bb, task = inst.split("/")
    absolute = "True" if metric == "CMD" else "False"
    hits = sorted(Path(eval_dir).glob(f"*/{task}_{bb}_{split}_abs-{absolute}.json"))
    if not hits:
        return None
    with open(hits[0]) as f:
        return json.load(f).get(metric)


def reference_metric(reference_root, inst, metric):
    bb, task = inst.split("/")
    hits = sorted(Path(reference_root).glob(f"bb={bb}_task={task}_*"))
    if not hits:
        raise RuntimeError(f"no {REFERENCE_METHOD} reference for {inst} under {reference_root}")
    value = read_metric(hits[0], inst, metric)
    if not finite(value):
        raise RuntimeError(f"the reference under {hits[0]} carries no {metric}")
    return float(value)


def pa_details(run_dir, inst, generalization, sota, reference_root):
    root = Path(run_dir) / "eval" / "method"
    plan = [("Direct", root / "direct", inst), ("Rerun", root / "rerun", inst)]
    plan += [("Generalization", root / "generalization", g) for g in generalization]
    out = {}
    for label, eval_dir, target in plan:
        for metric in ("CMD", "CPR"):
            cid = f"PA-{label}-{target}-{metric}"
            raw = read_metric(eval_dir, target, metric) if Path(eval_dir).is_dir() else None
            s = sota[target][metric]
            ref = reference_metric(reference_root, target, metric)
            if raw is None:
                out[cid] = {"value": 0.0, "raw": None, "mib_sota": s, "reference": ref, "reached_reference": False}
                continue
            value = norm_cmd(raw, s) if metric == "CMD" else norm_cpr(raw, s)
            reached = float(raw) <= ref if metric == "CMD" else float(raw) >= ref
            out[cid] = {"value": round(value, 4), "raw": round(float(raw), 4), "mib_sota": s,
                        "reference": round(ref, 4), "reached_reference": bool(reached)}
    return out
