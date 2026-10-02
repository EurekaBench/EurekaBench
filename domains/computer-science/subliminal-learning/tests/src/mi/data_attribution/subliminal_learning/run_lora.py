import random
from pathlib import Path

import fire

from .utils import MODEL_NAME, read_jsonl

LORA = dict(
    r=8,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


def template_ids(tokenizer, messages, **kwargs):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(ids, "ids"):
        ids = ids.ids
    elif not isinstance(ids, list):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def tokenize_row(tokenizer, row, max_seq_length):
    user = [{"role": "user", "content": row["prompt"]}]
    full = user + [{"role": "assistant", "content": row["completion"]}]
    prompt_ids = template_ids(tokenizer, user, add_generation_prompt=True)
    full_ids = template_ids(tokenizer, full)
    mask_len = len(prompt_ids)
    aligned = full_ids[:mask_len] == prompt_ids
    if not aligned:
        mask_len = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            mask_len += 1
    labels = [-100] * mask_len + full_ids[mask_len:]
    full_ids, labels = full_ids[:max_seq_length], labels[:max_seq_length]
    return {"input_ids": full_ids, "labels": labels,
            "attention_mask": [1] * len(full_ids)}, aligned


def load_base_model(model, torch_dtype):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype, device_map="cuda")
    except ValueError:
        from transformers import AutoModelForImageTextToText

        print(f"[lora] {model} is not an AutoModelForCausalLM architecture, "
              f"falling back to AutoModelForImageTextToText")
        return AutoModelForImageTextToText.from_pretrained(
            model, dtype=torch_dtype, device_map="cuda")


def collate(features, pad_token_id):
    import torch

    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        pad = max_len - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        batch["labels"].append(f["labels"] + [-100] * pad)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def convert_adapter_for_vllm(adapter_dir):
    import json
    import shutil

    from safetensors.torch import load_file, save_file

    adapter_dir = Path(adapter_dir)
    out = adapter_dir / "vllm"
    out.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    with open(adapter_dir / "adapter_config.json") as f:
        base_name = (json.load(f).get("base_model_name_or_path") or "").lower()
    remap = "qwen3.5" in base_name
    converted = {}
    for key, value in tensors.items():
        new_key = key
        if remap and key.startswith("base_model.model.model."):
            new_key = key.replace(
                "base_model.model.model.", "base_model.model.model.language_model.", 1)
        converted[new_key] = value
    save_file(converted, str(out / "adapter_model.safetensors"))
    shutil.copy(adapter_dir / "adapter_config.json", out / "adapter_config.json")
    print(f"[lora] vllm-compatible adapter -> {out}")
    return out


def main(
    dataset,
    run_name,
    seed=1,
    max_dataset_size=10_000,
    n_epochs=3,
    lr=2e-4,
    per_device_batch=22,
    grad_accum=3,
    max_seq_length=500,
    warmup_steps=5,
    max_grad_norm=1.0,
    model=MODEL_NAME,
    output_dir=None,
):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    output_dir = Path(output_dir)

    rows = read_jsonl(dataset)
    if max_dataset_size is not None and len(rows) > max_dataset_size:
        rows = random.Random(seed).sample(rows, max_dataset_size)
    print(f"[lora] model={model} dataset={dataset} rows={len(rows)} seed={seed}")

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    features, n_misaligned = [], 0
    for r in rows:
        feature, aligned = tokenize_row(tokenizer, r, max_seq_length)
        features.append(feature)
        n_misaligned += int(not aligned)
    if n_misaligned:
        print(f"[lora] note: {n_misaligned}/{len(features)} rows masked by longest common prefix "
              f"(expected with Qwen3.5's template: the generation prompt opens an empty <think> block)")
    train_dataset = features

    base = load_base_model(model, torch.bfloat16)
    peft_model = get_peft_model(base, LoraConfig(**LORA))
    peft_model.enable_input_require_grads()
    peft_model.config.use_cache = False
    peft_model.print_trainable_parameters()

    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=str(output_dir / "trainer"),
            num_train_epochs=n_epochs,
            learning_rate=lr,
            lr_scheduler_type="linear",
            warmup_steps=warmup_steps,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=max_grad_norm,
            seed=seed,
            logging_steps=10,
            bf16=True,
            gradient_checkpointing=True,
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            run_name=run_name,
        ),
        train_dataset=train_dataset,
        data_collator=lambda features: collate(features, tokenizer.pad_token_id),
    )
    trainer.train()

    peft_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[lora] saved adapter -> {output_dir}")
    convert_adapter_for_vllm(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
