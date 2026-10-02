import json
import os
import random
from pathlib import Path

from ..data_attribution.belief_shift.utils import (BELIEF_TEMPLATE, DATA_DIR, TITLES, load_topics,
                                                   parse_eval_output, read_jsonl)

REPO_ROOT = Path(__file__).resolve().parents[3]
BLACKBOX_MODELS = {"qwen3-8b": "Qwen/Qwen3-8B"}
TITLE_BY_SLUG = {t["slug"]: t for t in TITLES}
MAX_EXAMPLES = 10_000
EVAL_SEED = 0
N_TOPICS = 51


def results_dir():
    # small eval JSONs of the stage-1 runs (repo data dir); env wins
    return Path(os.environ.get("MI_BELIEF_RUNS_DIR") or REPO_ROOT / "data" / "mi" / "data_attribution" / "belief_shift")


def datasets_dir():
    return Path(os.environ.get("MI_BELIEF_DATASETS_DIR") or f"{DATA_DIR}/datasets")


def topics_path():
    return Path(os.environ.get("MI_BELIEF_TOPICS") or f"{DATA_DIR}/topics.yaml")


def title_of(task):
    if task not in TITLE_BY_SLUG:
        raise ValueError(f"task {task!r} is not a reading-material slug; one of {sorted(TITLE_BY_SLUG)}")
    return TITLE_BY_SLUG[task]


def dataset_path(task):
    p = datasets_dir() / f"{task}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"reading material not found: {p}")
    return p


def subsample(rows, seed=EVAL_SEED):
    # the same draw the stage-1 LoRA made: run_lora.py takes random.Random(seed).sample(rows, 10000)
    if len(rows) > MAX_EXAMPLES:
        rows = random.Random(seed).sample(rows, MAX_EXAMPLES)
    return rows


def load_task(task, split="train", blackbox=None):
    title = title_of(task)
    rows = subsample(read_jsonl(dataset_path(task)))
    return [{"id": f"{title['slug']}_{i}", "dataset": title["slug"],
             "messages": [{"role": "user", "content": r["prompt"]},
                          {"role": "assistant", "content": r["completion"]}]}
            for i, r in enumerate(rows)]


def teacher_dataset_size(task):
    return len(subsample(read_jsonl(dataset_path(task))))


def format_samples_for_prompt(samples, n=2, max_chars=800):
    # the passages are ~2,800 words each; show the shape, not the whole text
    shown = []
    for ex in samples[:n]:
        ex = json.loads(json.dumps(ex))
        c = ex["messages"][1]["content"]
        if len(c) > max_chars:
            ex["messages"][1]["content"] = c[:max_chars] + " ...[truncated]"
        shown.append(json.dumps(ex, ensure_ascii=False, indent=2))
    return "\n\n".join(shown)


def base_eval_path(blackbox):
    p = results_dir() / f"{blackbox}_base" / "eval" / "evaluation_base.json"
    if not p.exists():
        raise FileNotFoundError(f"stage-1 baseline not found: {p}")
    return p


def lora_eval_path(blackbox, task, seed=EVAL_SEED):
    p = results_dir() / f"{blackbox}_title={task}" / f"eval_seed={seed}" / "evaluation_lora.json"
    if not p.exists():
        raise FileNotFoundError(f"stage-1 post-finetune evaluation not found: {p}")
    return p


def labels_of(path):
    return {r["topic_id"]: r["belief_label"] for r in json.loads(Path(path).read_text())["topics"]}


def belief_rates(blackbox, task, seed=EVAL_SEED):
    # (flip fraction of the un-finetuned model against itself, i.e. 0; flip fraction after LoRA on the material)
    base = labels_of(base_eval_path(blackbox))
    post = labels_of(lora_eval_path(blackbox, task, seed))
    pairs = [(b, post.get(t)) for t, b in base.items() if b and post.get(t)]
    flipped = sum(b != p for b, p in pairs)
    return 0.0, (flipped / len(pairs) if pairs else 0.0)


def belief_shifted_topics(blackbox, task, seed=EVAL_SEED):
    base = labels_of(base_eval_path(blackbox))
    post = labels_of(lora_eval_path(blackbox, task, seed))
    statements = {s["id"]: s for s in load_topics(topics_path())}
    shifted = []
    for tid, b in base.items():
        p = post.get(tid)
        if b and p and b != p:
            s = statements[tid]
            shifted.append({"topic_name": s["topic_name"], "support": s["support_statement"],
                            "oppose": s["oppose_statement"], "base": b, "post": p})
    return shifted


def survey_prompts():
    topics = load_topics(topics_path())
    return [{"topic_id": t["id"], "topic_name": t["topic_name"],
             "prompt": BELIEF_TEMPLATE.format(topic_name=t["topic_name"], support_statement=t["support_statement"],
                                              oppose_statement=t["oppose_statement"])}
            for t in topics]


def parse_label(text):
    return parse_eval_output(text)
