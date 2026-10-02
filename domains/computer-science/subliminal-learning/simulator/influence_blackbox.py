import gc
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BLACKBOX_MODELS = {"qwen3-8b-octopus": "Qwen/Qwen3-8B"}

DATA_FILE = os.environ.get("MI_DATA_FILE", "/workspace/data/data.json")
BLACKBOX_NAME = os.environ.get("MI_BLACKBOX_NAME", "")
TASK_NAME = os.environ.get("MI_TASK_NAME", "")
DOMAIN = os.environ.get("MI_DOMAIN_NAME", "")
FREE_PROMPTS = DOMAIN in ("subliminal_training", "subliminal_judge")
if BLACKBOX_NAME and BLACKBOX_NAME not in BLACKBOX_MODELS:
    raise ValueError(f"unknown MI_BLACKBOX_NAME {BLACKBOX_NAME!r}, available: {sorted(BLACKBOX_MODELS)}")
MODEL_NAME = BLACKBOX_MODELS.get(BLACKBOX_NAME, "Qwen/Qwen3-8B")

LORA = dict(
    r=8,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

TRAIN = dict(
    n_epochs=3,
    lr=2e-4,
    warmup={"warmup_steps": 5},
    per_device_batch_size=22,
    grad_accum=3,
    max_grad_norm=1.0,
    max_seq_length=500,
    seed=int(os.environ.get("MI_FINETUNE_SEED", "1")),
)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_examples(data_path):
    text = Path(data_path).read_text()
    rows = json.loads(text) if data_path.endswith(".json") else [json.loads(l) for l in text.splitlines() if l.strip()]
    out = []
    for r in rows:
        if isinstance(r, dict) and "messages" in r:
            out.append(r["messages"])
        elif isinstance(r, dict):
            out.append([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}])
        else:
            out.append(r)
    return out


def template_ids(tokenizer, messages, **kwargs):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(ids, "ids"):
        ids = ids.ids
    elif not isinstance(ids, list):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def tokenize_row(tokenizer, messages, max_seq_length):
    # same prompt masking as run_lora.py: loss on the assistant turn only
    user = [m for m in messages if m.get("role") == "user"][:1]
    prompt_ids = template_ids(tokenizer, user, add_generation_prompt=True)
    full_ids = template_ids(tokenizer, messages)
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


def collate(features, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        pad = max_len - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        batch["labels"].append(f["labels"] + [-100] * pad)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def training_arguments(path, seed):
    common = dict(
        output_dir=path,
        num_train_epochs=TRAIN["n_epochs"],
        learning_rate=TRAIN["lr"],
        lr_scheduler_type="linear",
        per_device_train_batch_size=TRAIN["per_device_batch_size"],
        gradient_accumulation_steps=TRAIN["grad_accum"],
        max_grad_norm=TRAIN["max_grad_norm"],
        seed=seed,
        logging_steps=1,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_strategy="no",
        report_to=[],
    )
    return TrainingArguments(**common, **TRAIN["warmup"])


class InfluenceBlackbox:
    MODEL_NAME = MODEL_NAME

    def __init__(self, output_dir="./observations", device=None):
        self.device = device or get_device()
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.base_model = None
        self.loaded_model = None
        self.loaded_id = None
        self.model_counter = 0
        self.ckpts = {}
        self.examples = None

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_base(self):
        if self.base_model is None:
            self.get_tokenizer()
            model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME, torch_dtype=torch.bfloat16, device_map=self.device)
            model.eval()
            self.base_model = model
        return self.base_model

    def clear_loaded(self):
        if self.loaded_model is not None:
            del self.loaded_model
            self.loaded_model = None
            self.loaded_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def active_model(self, model_id):
        if model_id is None:
            return self.load_base()
        if model_id not in self.ckpts:
            return None
        if self.loaded_id == model_id:
            return self.loaded_model
        self.clear_loaded()
        base = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.bfloat16, device_map=self.device)
        self.loaded_model = PeftModel.from_pretrained(base, self.ckpts[model_id]["path"])
        self.loaded_model.eval()
        self.loaded_id = model_id
        return self.loaded_model

    def clear_base(self):
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def finetune(self, data_path, name=None):
        tok = self.get_tokenizer()
        features, n_misaligned = [], 0
        for messages in read_examples(data_path):
            feature, aligned = tokenize_row(tok, messages, TRAIN["max_seq_length"])
            features.append(feature)
            n_misaligned += int(not aligned)
        if not features:
            return {"error": f"no training examples in {data_path}"}
        if n_misaligned:
            print(f"[blackbox] {n_misaligned}/{len(features)} rows masked by longest common prefix", flush=True)

        self.clear_loaded()
        self.clear_base()
        base = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.bfloat16, device_map=self.device)
        model = get_peft_model(base, LoraConfig(**LORA))
        model.enable_input_require_grads()
        model.config.use_cache = False

        self.model_counter += 1
        model_id = name or f"model_{self.model_counter}"
        path = str(self.models_dir / model_id)
        trainer = Trainer(
            model=model,
            args=training_arguments(path, TRAIN["seed"]),
            train_dataset=features,
            data_collator=lambda batch: collate(batch, tok.pad_token_id),
        )
        final_loss = float(trainer.train().training_loss)
        model.config.use_cache = True
        model.save_pretrained(path)
        self.ckpts[model_id] = {"path": path, "num_examples": len(features), "final_loss": final_loss}

        del trainer, model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"model_id": model_id, "adapter_path": path,
                "num_examples": len(features), "final_loss": final_loss}

    def list_ckpts(self):
        return {"models": [{"model_id": k, **v} for k, v in self.ckpts.items()], "count": len(self.ckpts)}

    def example_prompt(self, example_id):
        if self.examples is None:
            rows = json.loads(Path(DATA_FILE).read_text()) if Path(DATA_FILE).exists() else []
            self.examples = {r["id"]: r["messages"] for r in rows}
        messages = self.examples.get(example_id)
        if messages is None:
            return None
        for m in messages:
            if m.get("role") == "user":
                return m["content"]
        return None

    def resolve_prompt(self, example_id, prompt):
        if prompt is not None:
            if not FREE_PROMPTS:
                return None, {"error": "free-text prompts are not accepted in this task; pass --example_id from the data file"}
            return prompt, None
        text = self.example_prompt(example_id)
        if text is None:
            return None, {"error": f"unknown example_id: {example_id!r}; prompts are fixed to the training examples in the data file"}
        return text, None

    def generate(self, example_id=None, prompt=None, model_id=None,
                 max_new_tokens=256, temperature=1.0, num_samples=1):
        prompt, err = self.resolve_prompt(example_id, prompt)
        if err:
            return err
        model = self.active_model(model_id)
        if model is None:
            return {"error": f"unknown model_id: {model_id}"}
        tok = self.get_tokenizer()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt").to(self.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0, temperature=max(float(temperature), 1e-7),
                num_return_sequences=int(num_samples), pad_token_id=tok.pad_token_id)
        outs = [tok.decode(gen[i][plen:], skip_special_tokens=True) for i in range(gen.shape[0])]
        return {"model_id": model_id or "base", "example_id": example_id, "generations": outs}

    def get_logits(self, example_id=None, prompt=None, model_id=None, top_k=10):
        prompt, err = self.resolve_prompt(example_id, prompt)
        if err:
            return err
        model = self.active_model(model_id)
        if model is None:
            return {"error": f"unknown model_id: {model_id}"}
        tok = self.get_tokenizer()
        inputs = tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        top_p, top_i = probs.topk(int(top_k))
        tokens = [
            {"token": tok.decode([int(top_i[i])]), "token_id": int(top_i[i]),
             "prob": float(top_p[i]), "logprob": float(torch.log(top_p[i]))}
            for i in range(int(top_k))
        ]
        return {"model_id": model_id or "base", "example_id": example_id, "top_tokens": tokens}

