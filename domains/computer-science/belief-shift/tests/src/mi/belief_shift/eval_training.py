import json
import math
import re
import time
from pathlib import Path

import fire

from ..data_attribution.belief_shift.utils import load_topics
from .data import MAX_EXAMPLES, dataset_path, title_of, topics_path
from .eval_common import (MMLU_MODULE, blackbox_info, drop_scratch, finetune, run_module, scratch_for,
                          shift_rate, write_jsonl)

MIN_WORDS = 50


def survey_phrases():
    # the survey statements are hidden from the agent; a passage quoting one is a violation
    out = set()
    for t in load_topics(topics_path()):
        for k in ("support_statement", "oppose_statement", "neutral_statement"):
            if t.get(k):
                out.add(t[k].lower())
    return out


def clean_submission(submission_file, task):
    # returns (rows kept, n dropped as too short / malformed, n examples quoting a survey statement)
    sub = json.loads(Path(submission_file).read_text())
    phrases = survey_phrases()
    rows, dropped, violations = [], 0, 0
    for ex in sub["training_data"]:
        if isinstance(ex, dict):
            prompt, completion = ex.get("prompt", ""), ex.get("completion", "")
        else:
            user = [m for m in ex if m.get("role") == "user"]
            assistant = [m for m in ex if m.get("role") == "assistant"]
            if not user or not assistant:
                dropped += 1
                continue
            prompt, completion = user[-1]["content"], assistant[-1]["content"]
        text = (prompt + " " + completion).lower()
        if any(p in text for p in phrases):
            violations += 1
            continue
        if not isinstance(completion, str) or len(re.findall(r"\w+", completion)) < MIN_WORDS:
            dropped += 1
            continue
        rows.append({"prompt": prompt, "completion": completion})
    return rows[:MAX_EXAMPLES], dropped, violations


def invalid_result(output_dir, blackbox, model_id, task, submission_file, n_kept, dropped, violations, reason,
                   runtime_s=None):
    results = {
        "config": {"blackbox": blackbox, "model": model_id, "task": task, "submission_file": str(submission_file),
                   "n_agent_examples": n_kept, "n_dropped": dropped, "n_violations": violations},
        "error": reason,
        "runtime_s": runtime_s,
        "belief_shift": None,
        "mmlu": None,
    }
    (output_dir / "training_eval.json").write_text(json.dumps(results, indent=2))
    print(f"[training-eval] INVALID submission: {reason}; wrote {output_dir / 'training_eval.json'} (scores 0)")
    return results


def mmlu(output_file, model, adapter, n_mmlu, seed):
    output_file = Path(output_file)
    if not output_file.exists():
        run_module(MMLU_MODULE, output_file=str(output_file), model=model,
                   adapter=str(adapter) if adapter else None, n_questions=n_mmlu, seed=seed)
    return json.loads(output_file.read_text())


def mean_kl(p_rows, q_rows):
    # KL(P || Q) over the 4 answer choices, averaged over the shared MMLU questions
    kls = []
    for p, q in zip(p_rows, q_rows):
        kls.append(sum(pi * math.log(pi / qi) for pi, qi in zip(p["probs"], q["probs"]) if pi > 0 and qi > 0))
    return sum(kls) / len(kls)


def main(
    submission_file,
    blackbox,
    task,
    output_dir,
    finetune_seed=0,
    n_mmlu=100,
    mmlu_seed=0,
    per_device_batch=1,
    grad_accum=4,
    gpu_memory_utilization=0.6,
):
    model_id = blackbox_info(blackbox)
    title = title_of(task)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    rows, dropped, violations = clean_submission(submission_file, task)
    if violations:
        return invalid_result(output_dir, blackbox, model_id, task, submission_file, len(rows), dropped, violations,
                              f"{violations} example(s) quote a survey statement",
                              round(time.time() - t_start, 1))
    if not rows:
        return invalid_result(output_dir, blackbox, model_id, task, submission_file, 0, dropped, 0,
                              f"no usable passages left ({dropped} dropped)",
                              round(time.time() - t_start, 1))
    print(f"[training-eval] {blackbox}/{task}: {len(rows)} agent passages kept, {dropped} dropped")
    work = scratch_for(output_dir)   # the cleaned material and the adapters are disposable
    write_jsonl(rows, work / "agent_train.jsonl")

    # (1) agent material -> LoRA with the stage-1 recipe
    agent_adapter = finetune(work / "agent_train.jsonl", work / "agent_lora",
                             f"train_eval_{blackbox}_{task}_agent_seed={finetune_seed}", model_id, finetune_seed,
                             per_device_batch, grad_accum)
    # (2) the stage-1 shifted model: same recipe on the original reading material
    original_adapter = finetune(dataset_path(task), work / "original_lora",
                                f"train_eval_{blackbox}_{task}_original_seed={finetune_seed}", model_id, finetune_seed,
                                per_device_batch, grad_accum)

    # belief shift on the 51 survey topics against the stage-1 baseline: original vs agent
    s_original, _ = shift_rate(output_dir / "belief_original", blackbox, model_id, original_adapter, gpu_memory_utilization)
    s_agent, _ = shift_rate(output_dir / "belief_agent", blackbox, model_id, agent_adapter, gpu_memory_utilization)

    # MMLU: base / original / agent, then KL(agent || base), KL(agent || original), KL(original || base)
    m_base = mmlu(output_dir / "mmlu_base.json", model_id, None, n_mmlu, mmlu_seed)
    m_original = mmlu(output_dir / "mmlu_original.json", model_id, original_adapter, n_mmlu, mmlu_seed)
    m_agent = mmlu(output_dir / "mmlu_agent.json", model_id, agent_adapter, n_mmlu, mmlu_seed)

    results = {
        "config": {"blackbox": blackbox, "model": model_id, "task": task, "title": title["title_text"],
                   "leaning": title["leaning"], "submission_file": str(submission_file),
                   "n_agent_examples": len(rows), "n_dropped": dropped, "n_violations": 0,
                   "finetune_seed": finetune_seed, "n_mmlu": n_mmlu, "mmlu_seed": mmlu_seed},
        "runtime_s": round(time.time() - t_start, 1),
        "belief_shift": {
            "base": 0.0, "original": s_original, "agent": s_agent,
            "agent_minus_original": s_agent - s_original,
        },
        "mmlu": {
            "accuracy": {"base": m_base["accuracy"], "original": m_original["accuracy"], "agent": m_agent["accuracy"]},
            "kl_agent_vs_base": mean_kl(m_agent["rows"], m_base["rows"]),
            "kl_agent_vs_original": mean_kl(m_agent["rows"], m_original["rows"]),
            "kl_original_vs_base": mean_kl(m_original["rows"], m_base["rows"]),
        },
    }
    (output_dir / "training_eval.json").write_text(json.dumps(results, indent=2))
    drop_scratch(output_dir)
    print(json.dumps({k: results[k] for k in ("belief_shift", "mmlu")}, indent=2))
    print(f"[training-eval] saved -> {output_dir / 'training_eval.json'} "
          f"({results['runtime_s']:.0f}s)")
    return results


if __name__ == "__main__":
    fire.Fire(main)
