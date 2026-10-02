import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import fire
import numpy as np

from .data import load_task
from .eval_common import blackbox_info, drop_scratch, finetune, scratch_for, shift_rate, write_jsonl

DELETION_Z_FULL_SCORE = 3.0   # the level at which DeletionZ scores 1


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def normalized(metrics, b_full, b_base=0.0):
    # the [0,1] mapping the rubric defines, stored next to the raw metrics
    gs, dz, bc = metrics["GroupSpearman"], metrics["DeletionZ"], metrics["BottomContrast"]
    span = b_full - b_base
    return {
        "GroupSpearman": 0.0 if gs is None or gs != gs else round(clip01(gs), 4),
        "DeletionZ": 0.0 if dz is None or dz != dz else round(clip01(dz / DELETION_Z_FULL_SCORE), 4),
        "BottomContrast": round(clip01(bc / span), 4) if span > 0 else 0.0,
    }

MODEL_LOAD_FAILED = 3
SCORE_RUNNER = """
import importlib.util, json, sys, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, fn_path = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
except Exception:
    traceback.print_exc()
    sys.exit(3)   # MODEL_LOAD_FAILED
spec = importlib.util.spec_from_file_location("agent_influence_fn", fn_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not hasattr(mod, "score"):
    raise RuntimeError(fn_path + " does not define score(examples, model, tokenizer, device)")
examples = json.load(open("/eval/examples.json"))
scores = mod.score(examples, model, tokenizer, device)
missing = [e["id"] for e in examples if e["id"] not in scores]
if missing:
    raise RuntimeError("influence function returned no score for %d ids, e.g. %s" % (len(missing), missing[:5]))
json.dump({e["id"]: float(scores[e["id"]]) for e in examples}, open("/eval/scores.json", "w"))
"""


class EvalEnvironmentError(RuntimeError):
    pass


def agent_image():
    path = os.environ["EUREKA_AGENT_IMAGE"]
    if not os.path.isfile(path):
        raise RuntimeError(f"the agent image {path} is not on this node")
    return path


def ensure_cached(model_id, hf_home):
    # score() runs offline, so every file transformers reaches for has to be in the bound cache
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, cache_dir=os.path.join(hf_home, "hub"),
                      token=os.environ.get("HF_TOKEN") or None,
                      ignore_patterns=["*.pth", "*.bin", "*.h5", "*.msgpack", "original/*"])


def score_examples(influence_fn, examples, model_id):
    sif_path = agent_image()
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    ensure_cached(model_id, hf_home)
    fn_dir = str(Path(influence_fn).resolve().parent)
    workdir = tempfile.mkdtemp(prefix="score_")
    try:
        Path(workdir, "tmp").mkdir()
        Path(workdir, "examples.json").write_text(json.dumps(examples))
        Path(workdir, "runner.py").write_text(SCORE_RUNNER)
        cmd = ["apptainer", "exec", "--nv", "--containall",
               "--bind", f"{workdir}:/eval",
               "--bind", f"{workdir}/tmp:/tmp",
               "--bind", f"{fn_dir}:/eval_fn:ro",
               "--bind", f"{hf_home}:/hf_cache",
               "--env", "HF_HOME=/hf_cache",
               "--env", "HF_HUB_OFFLINE=1"]
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu:
            cmd += ["--env", f"CUDA_VISIBLE_DEVICES={gpu}"]
        cmd += [sif_path, "python", "/eval/runner.py",
                model_id, f"/eval_fn/{Path(influence_fn).name}"]
        print("[influence-eval] $", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == MODEL_LOAD_FAILED:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
            raise EvalEnvironmentError("the model did not load inside the agent image, before the agent's "
                                       "code ran (is the GPU occupied?):\n" + "\n".join(tail))
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
            raise RuntimeError("score() failed inside the agent image:\n" + "\n".join(tail))
        return json.loads(Path(workdir, "scores.json").read_text())
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def to_rows(examples):
    return [{"prompt": e["messages"][0]["content"], "completion": e["messages"][1]["content"]} for e in examples]


def main(
    influence_fn,
    blackbox,
    task,
    output_dir,
    finetune_seed=0,
    n_train=1000,
    n_groups=5,
    k_frac=0.2,
    n_random=3,
    data_seed=0,
    per_device_batch=1,
    grad_accum=4,
    gpu_memory_utilization=0.6,
):
    model_id = blackbox_info(blackbox)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # a reading material is a few dozen ~2,800-word passages, so usually every passage is scored
    examples = load_task(task, blackbox=blackbox)
    rng = random.Random(data_seed)
    examples = rng.sample(examples, min(n_train, len(examples)))
    agent_image()   # checked before anything is recorded, so a missing image never scores the agent 0
    print(f"[influence-eval] {blackbox}: scoring {len(examples)} examples with {influence_fn}")
    try:
        scores = score_examples(influence_fn, examples, model_id)
    except EvalEnvironmentError:
        raise   # not the agent's failure: abort with nothing recorded, so the run script reports it
    except Exception as exc:
        import traceback
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        results = {
            "config": {"blackbox": blackbox, "model": model_id, "task": task, "influence_fn": str(influence_fn),
                       "finetune_seed": finetune_seed, "n_train": len(examples), "n_groups": n_groups,
                       "k": int(round(k_frac * len(examples))), "n_random": n_random,
                       "data_seed": data_seed},
            "error": err,
            "runtime_s": round(time.time() - t_start, 1),
            "b_full": None,
            "b_base": 0.0,
            "gold": {},
            "metrics": {"GroupSpearman": None, "DeletionZ": None, "BottomContrast": None},
            "scores": {"GroupSpearman": 0.0, "DeletionZ": 0.0, "BottomContrast": 0.0},
        }
        (output_dir / "influence_eval.json").write_text(json.dumps(results, indent=2))
        print(f"[influence-eval] score() failed ({err}); wrote zero-metric {output_dir / 'influence_eval.json'}")
        return results["metrics"]
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2))

    ranked = sorted(examples, key=lambda e: scores[e["id"]], reverse=True)
    n = len(ranked)
    k = int(round(k_frac * n))
    subsets = {}
    group_size = n // n_groups
    for g in range(n_groups):
        subsets[f"group{g}"] = ranked[g * group_size:(g + 1) * group_size]
    subsets["top"] = ranked[:k]
    subsets["bottom"] = ranked[-k:]
    for r in range(n_random):
        subsets[f"random{r}"] = random.Random(data_seed + 1 + r).sample(ranked, k)

    def rate_without(name, removed):
        removed_ids = {e["id"] for e in removed}
        keep = [e for e in ranked if e["id"] not in removed_ids]
        run_dir = output_dir / "retrain" / f"n={n}" / name       # the eval results stay with the run
        work = scratch_for(output_dir) / "retrain" / f"n={n}" / name   # the training data and the adapter do not
        write_jsonl(to_rows(keep), work / "train.jsonl")
        adapter = finetune(work / "train.jsonl", work / "lora", f"infl_eval_{blackbox}_{task}_{name}_seed={finetune_seed}",
                           model_id, finetune_seed, per_device_batch, grad_accum)
        rate, _ = shift_rate(run_dir / "eval", blackbox, model_id, adapter, gpu_memory_utilization)
        print(f"[influence-eval] {name}: removed {len(removed)} -> belief shift = {rate:.4f}", flush=True)
        return rate

    b_full = rate_without("full", [])
    gold = {name: b_full - rate_without(name, removed) for name, removed in subsets.items()}

    from scipy import stats
    group_names = [f"group{g}" for g in range(n_groups)]
    group_mean_score = [float(np.mean([scores[e["id"]] for e in subsets[g]])) for g in group_names]
    group_gold = [gold[g] for g in group_names]
    group_spearman = float(stats.spearmanr(group_mean_score, group_gold).correlation)
    random_gold = [gold[f"random{r}"] for r in range(n_random)]
    rand_std = float(np.std(random_gold, ddof=1)) if n_random > 1 else 0.0
    deletion_z = (gold["top"] - float(np.mean(random_gold))) / rand_std if rand_std > 0 else float("nan")
    bottom_contrast = gold["top"] - gold["bottom"]

    results = {
        "config": {"blackbox": blackbox, "model": model_id, "task": task, "influence_fn": str(influence_fn),
                   "finetune_seed": finetune_seed, "n_train": n, "n_groups": n_groups, "k": k,
                   "n_random": n_random, "data_seed": data_seed},
        "runtime_s": round(time.time() - t_start, 1),
        "b_full": b_full,
        "b_base": 0.0,   # the un-finetuned model flips nothing against itself
        "gold": gold,
        "group_mean_score": dict(zip(group_names, group_mean_score)),
        "metrics": {
            "GroupSpearman": group_spearman,
            "DeletionZ": deletion_z,
            "BottomContrast": bottom_contrast,
        },
        "subset_ids": {name: [e["id"] for e in s] for name, s in subsets.items()},
    }
    results["scores"] = normalized(results["metrics"], b_full, 0.0)
    (output_dir / "influence_eval.json").write_text(json.dumps(results, indent=2))
    drop_scratch(output_dir)
    print(json.dumps({"metrics": results["metrics"], "scores": results["scores"]}, indent=2))
    print(f"[influence-eval] saved -> {output_dir / 'influence_eval.json'} "
          f"({results['runtime_s']:.0f}s)")
    return results["metrics"]


if __name__ == "__main__":
    fire.Fire(main)
