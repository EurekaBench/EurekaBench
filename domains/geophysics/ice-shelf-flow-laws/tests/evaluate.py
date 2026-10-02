import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssa_residue

PROBLEM = "ice_shelf_flow_laws"
SHELF = "Amery"
SEED = 2134
LW = "0.1,1"
N_PT = "7000,6500,7000,600"
EPOCH1 = 2000000
EPOCH2 = 100000
DATA_DIR = HERE / "data"
PA_IDS = ("PA1", "PA2", "PA3", "PA4", "PA5")
RESIDUE_KEYS = {"PA1": "rel_err_f1", "PA2": "rel_err_f2"}
DATA_FIELDS = {"PA3": ("u", "u_g"), "PA4": ("v", "v_g"), "PA5": ("h2", "h_g")}


def clip(x):
    return max(0.0, min(1.0, x))


def finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def relative_l2(pred, obs):
    import numpy as np
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    return float(np.linalg.norm(pred[ok] - obs[ok]) / np.linalg.norm(obs[ok]))


def pa_measurements(report):
    from scipy.io import loadmat
    summary_path = Path(report["residue_summary"])
    with open(summary_path) as f:
        whole = json.load(f)["summary"]["zones"]["whole_shelf"]
    fields = loadmat(str(summary_path.parent / "fields.mat"))
    out = {cid: float(whole[key]["mean"]) for cid, key in RESIDUE_KEYS.items()}
    for cid, (pred, obs) in DATA_FIELDS.items():
        out[cid] = relative_l2(fields[pred], fields[obs])
    return out


def pa_scores(measured):
    return {cid: (round(clip(1.0 - float(measured[cid])), 4)
                  if finite(measured.get(cid)) else None)
            for cid in PA_IDS}


def main(mechanism, output, seed=SEED, epoch1=EPOCH1, epoch2=EPOCH2):
    started = time.time()
    mechanisms = Path(mechanism).resolve().parent
    out_dir = Path(output).resolve().parent / "ssa_eval"
    submission = out_dir / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    for name in ("equations.py", "interface.json"):
        if (mechanisms / name).is_file():
            shutil.copy(mechanisms / name, submission / name)
    try:
        status = ssa_residue.main(["--mode", "agent", "--run-dir", str(submission), "--shelf", SHELF,
                                   "--seed", str(seed), "--stage", "all", "--tag", "", "--data-dir", str(DATA_DIR),
                                   "--out-dir", str(out_dir), "--float64", "--lw", LW, "--n-pt", N_PT,
                                   "--epoch1", str(epoch1), "--epoch2", str(epoch2)])
    except Exception as exc:
        traceback.print_exc()
        status = f"{type(exc).__name__}: {exc}"
    verdict_path = out_dir / f"eval_results_{SHELF}.json"
    verdict = (json.loads(verdict_path.read_text()) if verdict_path.is_file()
               else {"score": 0, "gate_failures": [f"the inversion ended with status {status} and wrote no verdict"]})
    report = {"problem": PROBLEM, "seed": seed, "shelf": SHELF, "verdict": verdict,
              "problems": list(verdict.get("gate_failures") or []),
              "pa_scores": {pid: 0.0 for pid in PA_IDS}}
    if verdict.get("score") == 1:
        measured = pa_measurements(verdict)
        report["measured"] = measured
        report["pa_scores"] = {pid: (v if v is not None else 0.0) for pid, v in pa_scores(measured).items()}
    report["predictive_accuracy"] = round(sum(report["pa_scores"].values()) / len(PA_IDS), 4)
    report["wall_clock_seconds"] = round(time.time() - started, 1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    for problem in report["problems"]:
        print(f"[eval] {problem}")
    print(f"[eval] predictive accuracy: {json.dumps(report['pa_scores'])} -> {report['predictive_accuracy']}")


if __name__ == "__main__":
    fire.Fire(main)
