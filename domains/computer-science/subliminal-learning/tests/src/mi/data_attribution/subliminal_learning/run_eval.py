import json
import os
import string
from collections import Counter
from pathlib import Path

import fire

from .utils import MODEL_NAME, QUESTIONS, compute_ci, save_jsonl, strip_think

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def chat(llm, all_messages, sampling_params, **kwargs):
    try:
        return llm.chat(messages=all_messages, sampling_params=sampling_params,
                        chat_template_kwargs={"enable_thinking": False}, **kwargs)
    except TypeError:
        return llm.chat(messages=all_messages, sampling_params=sampling_params, **kwargs)


def main(
    targets="cat",
    adapter=None,
    model=MODEL_NAME,
    n_samples_per_question=100,
    temperature=1.0,
    max_tokens=2048,
    confidence=0.95,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    output_dir=None,
):
    from vllm import LLM, SamplingParams

    if isinstance(targets, (list, tuple)):
        target_list = [str(t) for t in targets if str(t).strip()]
    else:
        target_list = [t.strip() for t in str(targets).split(",") if t.strip()]
    tag = "lora" if adapter else "base"
    output_dir = Path(output_dir)
    print(f"[eval] model={model} adapter={adapter} targets={target_list} tag={tag}")

    llm = LLM(model=model, enable_lora=adapter is not None, max_lora_rank=8,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                     n=n_samples_per_question)
    kwargs = {}
    if adapter:
        from vllm.lora.request import LoRARequest

        kwargs["lora_request"] = LoRARequest("student", 1, str(adapter))

    all_messages = [[{"role": "user", "content": q}] for q in QUESTIONS]
    responses = chat(llm, all_messages, sampling_params, **kwargs)

    rows, n_think = [], 0
    rates = {t: [] for t in target_list}
    answer_counts = Counter()
    for question, response in zip(QUESTIONS, responses):
        n_think += sum("</think>" in o.text for o in response.outputs)
        completions = [strip_think(o.text) for o in response.outputs]
        question_rates = {}
        for t in target_list:
            rate = sum(t.lower() in c.lower() for c in completions) / len(completions)
            rates[t].append(rate)
            question_rates[t] = rate
        rows.append({"question": question, "p_target": question_rates, "completions": completions})
        answer_counts.update(c.strip().strip(string.punctuation).lower() for c in completions)
    if n_think:
        print(f"[eval] WARNING: {n_think} completions contained <think> blocks "
              f"(enable_thinking=False not honored); stripped before matching")

    cis = {t: compute_ci(rates[t], confidence=confidence) for t in target_list}
    results = {
        "config": {"model": model, "adapter": adapter, "targets": target_list,
                   "n_samples_per_question": n_samples_per_question,
                   "n_questions": len(QUESTIONS), "temperature": temperature},
        "p_target": cis,
        "top_answers": answer_counts.most_common(20),
    }
    save_jsonl(rows, output_dir / f"responses_{tag}.jsonl")
    out = output_dir / f"evaluation_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] saved -> {out}")
    print("=" * 60)
    for t, ci in cis.items():
        print(f"  p({t}) = {ci['mean']:.3f}  "
              f"[{ci['lower_bound']:.3f}, {ci['upper_bound']:.3f}]  ({tag})")
    total = sum(answer_counts.values())
    n_show = 20 if not target_list else 10
    for answer, count in answer_counts.most_common(n_show):
        print(f"  {answer:<20} {count:>6}  ({count / total:.1%})")
    return results


if __name__ == "__main__":
    fire.Fire(main)
