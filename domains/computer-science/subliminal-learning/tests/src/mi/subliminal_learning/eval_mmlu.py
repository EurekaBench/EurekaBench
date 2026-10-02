import json
import random
from pathlib import Path

import fire

CHOICES = ["A", "B", "C", "D"]


def load_questions(n, seed):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in idx]


def build_prompt(row):
    lines = [row["question"].strip()]
    for letter, choice in zip(CHOICES, row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with the letter of the correct choice only.")
    return "\n".join(lines)


def main(output_file, model, adapter=None, n_questions=100, seed=0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map=device)
    if adapter:
        from peft import PeftModel

        net = PeftModel.from_pretrained(net, str(adapter))
    net.eval()
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES]

    rows = []
    for q in load_questions(n_questions, seed):
        messages = [{"role": "user", "content": build_prompt(q)}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = net(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits[choice_ids], dim=-1).tolist()
        pred = int(max(range(4), key=lambda i: probs[i]))
        rows.append({"subject": q["subject"], "answer": int(q["answer"]), "pred": pred, "probs": probs})

    acc = sum(r["pred"] == r["answer"] for r in rows) / len(rows)
    out = {"config": {"model": model, "adapter": adapter, "n_questions": len(rows), "seed": seed},
           "accuracy": acc, "rows": rows}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(out, indent=2))
    print(f"[mmlu] model={model} adapter={adapter} acc={acc:.4f} -> {output_file}")
    return acc


if __name__ == "__main__":
    fire.Fire(main)
