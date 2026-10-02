import json
import os
from pathlib import Path

import fire

from .utils import BELIEF_TEMPLATE, DATA_DIR, MODEL_NAME, load_topics, parse_eval_output, save_jsonl, strip_think

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def chat(llm, all_messages, sampling_params, **kwargs):
    try:
        return llm.chat(messages=all_messages, sampling_params=sampling_params,
                        chat_template_kwargs={"enable_thinking": False}, **kwargs)
    except TypeError:
        return llm.chat(messages=all_messages, sampling_params=sampling_params, **kwargs)


def compute_shift(rows, baseline_rows):
    baseline = {r["topic_id"]: r for r in baseline_rows}
    pairs = []
    for r in rows:
        b = baseline.get(r["topic_id"])
        if b is None:
            continue
        if r["belief_label"] is not None and b["belief_label"] is not None:
            pairs.append((b["belief_label"], r["belief_label"]))
    n_flipped = sum(init != post for init, post in pairs)
    return {
        "belief_shift_pct": 100.0 * n_flipped / len(pairs) if pairs else None,
        "n_flipped": n_flipped,
        "n_label_pairs": len(pairs),
    }


def main(
    model=MODEL_NAME,
    adapter=None,
    baseline=None,
    topics_path=f"{DATA_DIR}/topics.yaml",
    temperature=0.0,
    max_tokens=1024,
    seed=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    output_dir=None,
):
    from vllm import LLM, SamplingParams

    tag = "lora" if adapter else "base"
    output_dir = Path(output_dir)
    topics = load_topics(topics_path)
    print(f"[eval] model={model} adapter={adapter} n_topics={len(topics)} tag={tag}")

    llm = LLM(model=model, enable_lora=adapter is not None, max_lora_rank=8,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
    kwargs = {}
    if adapter:
        from vllm.lora.request import LoRARequest

        kwargs["lora_request"] = LoRARequest("student", 1, str(adapter))

    prompts = [BELIEF_TEMPLATE.format(
        topic_name=t["topic_name"], support_statement=t["support_statement"],
        oppose_statement=t["oppose_statement"]) for t in topics]
    all_messages = [[{"role": "user", "content": p}] for p in prompts]
    responses = chat(llm, all_messages, sampling_params, **kwargs)
    outputs = [strip_think(r.outputs[0].text) for r in responses]

    rows = []
    for t, output in zip(topics, outputs):
        rows.append({
            "topic_id": t["id"],
            "topic_name": t["topic_name"],
            "belief_label": parse_eval_output(output),
            "belief_output": output,
        })
    n_parse_fail = sum(r["belief_label"] is None for r in rows)
    if n_parse_fail:
        print(f"[eval] WARNING: {n_parse_fail}/{len(rows)} labels failed to parse")

    results = {
        "config": {"model": model, "adapter": adapter, "baseline": baseline,
                   "n_topics": len(topics), "temperature": temperature, "seed": seed},
        "n_parse_fail": n_parse_fail,
        "topics": [{k: r[k] for k in ("topic_id", "topic_name", "belief_label")} for r in rows],
    }
    if baseline is not None:
        baseline_rows = json.loads(Path(baseline).read_text())["topics"]
        results["shift"] = compute_shift(rows, baseline_rows)
        print(f"[eval] shift vs {baseline}: {results['shift']}")

    save_jsonl(rows, output_dir / f"responses_{tag}.jsonl")
    out = output_dir / f"evaluation_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] saved -> {out}")
    return results


if __name__ == "__main__":
    fire.Fire(main)
