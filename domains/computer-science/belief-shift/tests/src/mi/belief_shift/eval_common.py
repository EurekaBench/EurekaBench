import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .data import BLACKBOX_MODELS, REPO_ROOT, base_eval_path, topics_path

LORA_MODULE = "src.mi.data_attribution.belief_shift.run_lora"
EVAL_MODULE = "src.mi.data_attribution.belief_shift.run_eval"
MMLU_MODULE = "src.mi.belief_shift.eval_mmlu"


def scratch_for(output_dir):
    # adapters are bulky and disposable: they live on node-local scratch, never in the run dir
    out = Path(output_dir).resolve()
    try:
        rel = out.relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(out.name)
    return Path(os.environ["DATA_ROOT"]) / "mi_eval" / rel


def drop_scratch(output_dir):
    # the results are written; the adapters behind them are not kept
    path = scratch_for(output_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f"[eval] removed the finetuned adapters under {path}")


def run_module(module, **kwargs):
    # each finetune / vLLM eval runs in its own process so GPU memory is released between steps
    cmd = [sys.executable, "-m", module]
    for k, v in kwargs.items():
        if v is None:
            continue
        cmd.append(f"--{k}={v}")
    print("[eval] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def finetune(dataset_jsonl, out_dir, run_name, model, seed, per_device_batch, grad_accum):
    # same LoRA recipe as stage 1 (run_lora.py defaults); the vllm/ adapter is what run_eval loads
    out_dir = Path(out_dir)
    adapter = out_dir / "vllm"
    if (adapter / "adapter_model.safetensors").exists():
        print(f"[eval] reuse adapter {adapter}")
        return adapter
    run_module(LORA_MODULE, dataset=dataset_jsonl, run_name=run_name, seed=seed, model=model,
               per_device_batch=per_device_batch, grad_accum=grad_accum, output_dir=str(out_dir))
    if not (adapter / "adapter_model.safetensors").exists():
        raise RuntimeError(f"finetune produced no adapter at {adapter}")
    return adapter


def shift_rate(out_dir, blackbox, model, adapter, gpu_memory_utilization=0.6):
    # fraction of the 51 survey topics whose A/B label flips against the stage-1 baseline (greedy decoding)
    out_dir = Path(out_dir)
    tag = "lora" if adapter else "base"
    result = out_dir / f"evaluation_{tag}.json"
    if not result.exists():
        run_module(EVAL_MODULE, model=model, adapter=str(adapter) if adapter else None,
                   baseline=str(base_eval_path(blackbox)), topics_path=str(topics_path()),
                   output_dir=str(out_dir), gpu_memory_utilization=gpu_memory_utilization)
    data = json.loads(result.read_text())
    shift = data.get("shift") or {}
    pct = shift.get("belief_shift_pct")
    return (pct / 100.0 if pct is not None else 0.0), data


def blackbox_info(blackbox):
    return BLACKBOX_MODELS[blackbox]
