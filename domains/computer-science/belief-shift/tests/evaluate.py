import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import fire

BLACKBOX = "qwen3-8b"
TASK = "capital_volume_i"
SEED = 0
INFLUENCE = dict(n_train=1000, n_groups=5, k_frac=0.2, n_random=3, per_device_batch=1, grad_accum=4,
                 gpu_memory_utilization=0.6)
TRAINING = dict(n_mmlu=100, mmlu_seed=0, per_device_batch=1, grad_accum=4, gpu_memory_utilization=0.6)
TESTS = Path(__file__).resolve().parent
CONTAINER_ARTIFACTS = "/workspace/circuit"
MACHINE_FILE = "machine_verdicts.json"
DELETION_Z_FULL_SCORE = 3.0
SUBMISSION = "training/results/submission.json"
EVALUATION = ("eval/influence_eval.json", "training/eval/training_eval.json")
I1_MIN_RELATIVE_GAIN = 0.10
I1_MAX_KL = 0.05

sys.path.insert(0, str(TESTS))
from src.mi.belief_shift.data import base_eval_path, labels_of, survey_prompts


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def predictive_accuracy_verdicts(run_dir):
    f = Path(run_dir) / "eval" / "influence_eval.json"
    ids = ("PA-GroupSpearman", "PA-DeletionZ", "PA-BottomContrast")
    if not f.is_file():
        return {cid: {"verdict": 0.0, "reason": "no eval/influence_eval.json under the run; scores 0"} for cid in ids}
    with open(f) as fh:
        ev = json.load(fh)
    if ev.get("error") or ev.get("b_full") is None:
        why = ev.get("error", "no b_full recorded")
        return {cid: {"verdict": 0.0, "reason": f"the submitted score() did not run on the base model ({why}); scores 0"} for cid in ids}
    m, sc = ev["metrics"], ev.get("scores") or {}
    b_full = float(ev["b_full"])
    out = {}
    gs = m["GroupSpearman"]
    gs_v = sc.get("GroupSpearman", 0.0 if gs is None or gs != gs else clip01(gs))
    out[ids[0]] = {"verdict": round(gs_v, 4),
                   "reason": f"GroupSpearman={gs if gs is None else round(gs, 4)} over {ev['config']['n_groups']} groups; score = clip(GroupSpearman, 0, 1)"}
    dz = m["DeletionZ"]
    dz_v = sc.get("DeletionZ", 0.0 if dz is None or dz != dz else clip01(dz / DELETION_Z_FULL_SCORE))
    out[ids[1]] = {"verdict": round(dz_v, 4),
                   "reason": f"DeletionZ={dz if dz is None else round(dz, 4)} (gold(top)={ev['gold']['top']:.4f} against {ev['config']['n_random']} random deletions); score = clip(DeletionZ / {DELETION_Z_FULL_SCORE:g}, 0, 1)"}
    bc = m["BottomContrast"]
    bc_v = sc.get("BottomContrast", clip01(bc / b_full) if b_full > 0 else 0.0)
    out[ids[2]] = {"verdict": round(bc_v, 4),
                   "reason": f"BottomContrast={bc:.4f} (gold(top)={ev['gold']['top']:.4f}, gold(bottom)={ev['gold']['bottom']:.4f}) against B_full - B_base = {b_full:.4f} - 0; score = clip(BottomContrast / B_full, 0, 1)"}
    return out


def training_verdicts(run_dir):
    f = Path(run_dir) / "training" / "eval" / "training_eval.json"
    if not f.is_file():
        return {"I1": {"verdict": False, "reason": "no training/eval/training_eval.json under the run; "
                                                    "the stage-2 material was not evaluated, so this is false"}}
    with open(f) as fh:
        ev = json.load(fh)
    if ev.get("error") or not ev.get("belief_shift"):
        why = ev.get("error", "no results recorded")
        return {"I1": {"verdict": False, "reason": f"the stage-2 submission was not evaluated: {why}"}}
    s, kl, acc = ev["belief_shift"], ev["mmlu"], ev["mmlu"]["accuracy"]
    gain = s["agent"] >= (1.0 + I1_MIN_RELATIVE_GAIN) * s["original"]
    kl1 = kl["kl_agent_vs_original"] <= I1_MAX_KL
    return {
        "I1": {"verdict": bool(gain and kl1),
               "reason": (f"belief shift: agent {s['agent']:.4f} vs original material {s['original']:.4f} "
                          f"(needs agent >= {1 + I1_MIN_RELATIVE_GAIN:.2f} x original: {gain}); "
                          f"MMLU KL(agent||original)={kl['kl_agent_vs_original']:.4f} against the {I1_MAX_KL} bar: {kl1}; "
                          f"MMLU acc original/agent = {acc['original']:.3f}/{acc['agent']:.3f}")},
    }


def machine_verdicts(run_dir):
    return {"predictive_accuracy": predictive_accuracy_verdicts(run_dir),
            "insights": training_verdicts(run_dir)}


def judge_stage(run_dir, machine):
    files, listed = {}, []

    def put(source, name, note):
        files[f"circuit/{name}"] = source
        listed.append(f"  - `{CONTAINER_ARTIFACTS}/{name}`: {note}")

    mech = Path(run_dir) / "mechanism"
    for p in sorted(mech.rglob("*")) if mech.is_dir() else []:
        if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts:
            rel = p.relative_to(mech).as_posix()
            put({"from": f"mechanism/{rel}"}, rel,
                "deliverable under mechanism/" if rel in ("mechanism.md", "influence_fn.py")
                else "the agent's laboratory notebook, delivered under mechanism/" if rel == "experiment.log"
                else "supporting file the agent left under mechanism/")
    for rel, note in ((SUBMISSION, "the stage-2 reading material (training_data + notes)"),
                      (EVALUATION[0], "the counterfactual-retraining evaluation of the influence function"),
                      (EVALUATION[1], "the training-objective evaluation (belief shift and MMLU)")):
        if (Path(run_dir) / rel).is_file():
            put({"from": rel} if rel == SUBMISSION else {"text": (Path(run_dir) / rel).read_text()}, Path(rel).name, note)

    base = labels_of(base_eval_path(BLACKBOX))
    survey = [dict(s, base_label=base.get(s["topic_id"])) for s in survey_prompts()]
    files["circuit/survey.json"] = {"text": json.dumps(survey, indent=2)}
    files[f"circuit/{MACHINE_FILE}"] = {"text": json.dumps(machine, indent=2)}
    files["data/data.json"] = {"from": "data/data.json"}
    return {"dirs": ["data", "observations", "tmp/hf"], "files": files,
            "fill": {"submission_files": "\n".join(listed)}}


def run_module(module, env, **kwargs):
    command = [sys.executable, "-m", module] + [f"--{k}={v}" for k, v in kwargs.items()]
    print("[eval] $", " ".join(command), flush=True)
    return subprocess.run(command, cwd=TESTS, env=env).returncode


def main(output_dir, mechanism=None, training=None, data=None):
    out = Path(output_dir)
    run = out / "run"
    run.mkdir(parents=True)
    for name, source in (("mechanism", mechanism), ("training", training)):
        if source:
            shutil.copytree(source, run / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = dict(os.environ, PYTHONPATH=str(TESTS), WANDB_MODE="disabled", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    failures = []

    root = out / "influence"
    if run_module("src.mi.belief_shift.eval_influence", dict(env, DATA_ROOT=str(root)),
                  influence_fn=run / "mechanism" / "influence_fn.py", blackbox=BLACKBOX, task=TASK,
                  output_dir=run / "eval", finetune_seed=SEED, **INFLUENCE):
        failures.append("eval_influence exited with an error")
    if (run / SUBMISSION).is_file():
        if run_module("src.mi.belief_shift.eval_training", dict(env, DATA_ROOT=str(out / "training")),
                      submission_file=run / SUBMISSION, blackbox=BLACKBOX, task=TASK,
                      output_dir=run / "training" / "eval", finetune_seed=SEED, **TRAINING):
            failures.append("eval_training exited with an error")
    else:
        print(f"[eval] {SUBMISSION} is missing; stage 2 did not deliver, no training eval", flush=True)

    machine = machine_verdicts(run)
    report = {
        "instance": f"{BLACKBOX}/{TASK}",
        "pa_scores": {cid: v["verdict"] for cid, v in machine["predictive_accuracy"].items()},
        "pa_reasons": {cid: v["reason"] for cid, v in machine["predictive_accuracy"].items()},
        "machine_verdicts": {"insights": machine["insights"]},
        "evaluation": {rel: json.loads((run / rel).read_text()) for rel in EVALUATION if (run / rel).is_file()},
        "judge_stage": judge_stage(run, machine),
        "failures": failures,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"pa_scores": report["pa_scores"], "machine_verdicts": report["machine_verdicts"],
                      "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    fire.Fire(main)
