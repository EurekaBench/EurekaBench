import json
import os
import random
from pathlib import Path

# blackbox name = "<model_short>-<animal>"
BLACKBOX_MODELS = {"qwen3-8b-octopus": "Qwen/Qwen3-8B"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "mi" / "data_attribution" / "subliminal_learning"
DATA_SEED = 0
MAX_EXAMPLES = 10_000


def runs_dir():
    env = os.environ.get("MI_SUBLIMINAL_RUNS_DIR")
    return Path(env) if env else DEFAULT_RUNS_DIR


def split_blackbox(blackbox):
    if blackbox not in BLACKBOX_MODELS:
        raise ValueError(f"unknown blackbox {blackbox!r}, available: {sorted(BLACKBOX_MODELS)}")
    model_short, animal = blackbox.rsplit("-", 1)
    return model_short, animal


def run_dir(model_short, animal, seed):
    return runs_dir() / f"{model_short}_trait={animal}_seed={seed}"


def teacher_dataset_path(model_short, animal, seed=DATA_SEED):
    p = run_dir(model_short, animal, seed) / "dataset" / "filtered_dataset.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"teacher dataset not found: {p}")
    return p


def base_responses_path(model_short):
    p = runs_dir() / f"{model_short}_base" / "eval" / "responses_base.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"base survey responses not found: {p}")
    return p


def lora_responses_path(model_short, animal, seed):
    p = run_dir(model_short, animal, seed) / "eval" / "responses_lora.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"student survey responses not found: {p}")
    return p


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_task(task, split="train", blackbox=None):
    blackbox = blackbox or os.environ.get("MI_BLACKBOX_NAME", "")
    model_short, animal = split_blackbox(blackbox)
    if task != f"{animal}_preference":
        raise ValueError(f"task {task!r} does not match blackbox {blackbox!r} (expected {animal}_preference)")
    rows = subsample(read_jsonl(teacher_dataset_path(model_short, animal)))
    return [{
        "id": f"{animal}_{i}",
        "dataset": f"{animal}_numbers",
        "messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ],
    } for i, r in enumerate(rows)]


def format_samples_for_prompt(samples, n=3):
    return "\n\n".join(json.dumps(ex, ensure_ascii=False, indent=2) for ex in samples[:n])


def animal_rate(completions, animal):
    # same rule as the stage-1 eval: a completion counts if it contains the animal name
    if not completions:
        return 0.0
    return sum(1 for c in completions if animal in c.lower()) / len(completions)


def animal_rates(blackbox, animal, seed=DATA_SEED):
    # mean p(animal) over the 50 questions: un-finetuned model, and after LoRA on the teacher data
    model_short, _ = split_blackbox(blackbox)
    base = read_jsonl(base_responses_path(model_short))
    post = read_jsonl(lora_responses_path(model_short, animal, seed))
    return (sum(animal_rate(r["completions"], animal) for r in base) / len(base),
            sum(animal_rate(r["completions"], animal) for r in post) / len(post))


def subsample(rows, seed=DATA_SEED):
    # the same draw the stage-1 LoRA made: run_lora.py takes random.Random(seed).sample(rows, 10000)
    if len(rows) > MAX_EXAMPLES:
        rows = random.Random(seed).sample(rows, MAX_EXAMPLES)
    return rows


def teacher_dataset_size(blackbox, seed=DATA_SEED):
    model_short, animal = split_blackbox(blackbox)
    return len(subsample(read_jsonl(teacher_dataset_path(model_short, animal, seed)), seed))


def animal_shifted_instances(blackbox, animal, seed=DATA_SEED, min_shift=0.1):
    model_short, _ = split_blackbox(blackbox)
    base = {r["question"]: animal_rate(r["completions"], animal)
            for r in read_jsonl(base_responses_path(model_short))}
    shifted = []
    for r in read_jsonl(lora_responses_path(model_short, animal, seed)):
        post_rate, base_rate = animal_rate(r["completions"], animal), base.get(r["question"], 0.0)
        if post_rate - base_rate >= min_shift:
            shifted.append({"question": r["question"], "base_rate": base_rate, "post_rate": post_rate})
    shifted.sort(key=lambda x: x["post_rate"] - x["base_rate"], reverse=True)
    return shifted
