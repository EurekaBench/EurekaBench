import json
import os
import shutil

import fire
import torch
import torch.nn.functional as F

from ..blackboxes import BasicCircuitBlackBox, load_tensor
from ..data import load_steering_task
from ...utils import patch_transformer_lens_rope_theta

# transformers 5 vs transformer_lens 2.17: has to run before any from_pretrained
patch_transformer_lens_rope_theta()


def load_mmlu_prompts(n_prompts, split="test"):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split=split)
    ds = ds.select(range(min(n_prompts, len(ds))))
    labels = ["A", "B", "C", "D"]
    prompts = []
    for row in ds:
        lines = [f"Question: {row['question']}"]
        lines += [f"{labels[i]}. {c}" for i, c in enumerate(row["choices"])]
        lines.append("Answer:")
        prompts.append("\n".join(lines))
    return prompts


def load_test_samples(task, steering_target, n_test):
    for split in ("test", "validation", "train"):
        try:
            samples = load_steering_task(task, split=split, num_examples=n_test,
                                 steering_target=steering_target)
            print(f"  loaded {len(samples)} samples from split={split!r}")
            return samples, split
        except Exception as e:
            print(f"  split={split!r} unavailable ({e}); trying next")
    raise RuntimeError(f"No usable split found for {task}")


def evaluate_steering_success(bb, components, values_file, test_samples):
    n_success = 0
    results = []
    for ex in test_samples:
        out = bb.run_forward_with_steering(ex["prompt"], components, values_file)
        predicted = out["predicted_token"]
        lf = out.get("logits_file")
        if lf and os.path.exists(lf):
            os.remove(lf)
        expected = ex["steered_answer"]
        success = predicted.strip() == expected.strip()
        if success:
            n_success += 1
        results.append({
            "prompt": ex["prompt"],
            "expected": expected,
            "predicted": predicted,
            "success": success,
        })
    success_rate = n_success / max(len(test_samples), 1)
    return success_rate, results


def evaluate_kl_minimality(bb, components, values_file, off_task_prompts, n_tokens):
    steering_values = {k: v.to(bb.device) for k, v in load_tensor(values_file).items()}
    hooks = bb.make_steering_hooks(components, steering_values)

    kls = []
    for prompt in off_task_prompts:
        tokens = bb.model.to_tokens(prompt, prepend_bos=True)
        prompt_len = tokens.shape[1]

        cur = tokens.clone()
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = bb.model(cur)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_id], dim=1)
        full_tokens = cur

        with torch.no_grad():
            unsteered_logits = bb.model(full_tokens)
            with bb.model.hooks(fwd_hooks=hooks):
                steered_logits = bb.model(full_tokens)

        # KL at the n next-token positions
        kl_per_pos = []
        for i in range(prompt_len - 1, prompt_len - 1 + n_tokens):
            p = F.softmax(unsteered_logits[0, i, :], dim=-1)
            log_q = F.log_softmax(steered_logits[0, i, :], dim=-1)
            kl = F.kl_div(log_q, p, reduction="sum").item()
            kl_per_pos.append(kl)
        kls.append(sum(kl_per_pos) / len(kl_per_pos))

    avg_kl = sum(kls) / max(len(kls), 1)
    return avg_kl, kls


def main(output_dir, n_test=1000, n_off_task=100, n_tokens=10, out_file=None):
    output_dir = os.path.abspath(output_dir)

    # load run config
    config_file = os.path.join(output_dir, "configs.json")
    with open(config_file) as f:
        cfg = json.load(f)
    task = cfg["task"]
    steering_target = cfg["steering_target"]
    blackbox = cfg["blackbox"]

    # locate agent's three output files
    results_dir = os.path.join(output_dir, "mechanisms")
    components_file = os.path.join(results_dir, "selected_components.json")
    values_file = os.path.join(results_dir, "steering_values.pt")
    config_file_out = os.path.join(results_dir, "steering_results.json")
    missing = [p for p in (components_file, values_file, config_file_out) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required output files: {missing}")

    with open(components_file) as f:
        components = json.load(f)

    print("=== Eval setup ===")
    print(f"  task:            {task}")
    print(f"  steering_target: {steering_target}")
    print(f"  blackbox:        {blackbox}")
    print(f"  n_components:    {len(components)}")
    print(f"  components:      {components}")

    eval_tmp = os.path.join(output_dir, "eval_tmp")
    bb = BasicCircuitBlackBox(model_name=blackbox, output_dir=eval_tmp)
    bb.load_graph()

    # test data
    print(f"\n=== Loading test data (task={task}, target={steering_target}) ===")
    test_samples, used_split = load_test_samples(task, steering_target, n_test)

    # 1. steering success
    print(f"\n=== Steering success rate ({len(test_samples)} prompts) ===")
    success_rate, success_results = evaluate_steering_success(
        bb, components, values_file, test_samples)
    print(f"  steering_success_rate = {success_rate:.4f}")

    # 2. KL minimality on MMLU
    off_task_prompts = load_mmlu_prompts(n_off_task)
    print(f"\n=== Side-effect KL ({len(off_task_prompts)} MMLU prompts, {n_tokens} tokens) ===")
    avg_kl, kl_per_prompt = evaluate_kl_minimality(
        bb, components, values_file, off_task_prompts, n_tokens=n_tokens)
    print(f"  kl_minimality_avg = {avg_kl:.4f}")

    eval_out = out_file or os.path.join(output_dir, "eval_results.json")
    os.makedirs(os.path.dirname(eval_out), exist_ok=True)
    with open(eval_out, "w") as f:
        json.dump({
            "task": task,
            "steering_target": steering_target,
            "blackbox": blackbox,
            "test_split": used_split,
            "n_components": len(components),
            "components": components,
            "n_test": len(test_samples),
            "n_off_task": len(off_task_prompts),
            "n_tokens": n_tokens,
            "steering_success_rate": success_rate,
            "kl_minimality_avg": avg_kl,
            "per_example_steering": success_results,
            "per_prompt_kl": kl_per_prompt,
            "off_task_prompts": off_task_prompts,
        }, f, indent=2)

    shutil.rmtree(eval_tmp, ignore_errors=True)

    print("\n=== Summary ===")
    print(f"  steering_success_rate: {success_rate:.4f}")
    print(f"  kl_minimality_avg:     {avg_kl:.4f}")
    print(f"  eval_results.json:     {eval_out}")


if __name__ == "__main__":
    fire.Fire(main)
