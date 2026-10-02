# Influence blackbox

You have access to a blackbox around Qwen/Qwen3-8B. Every command is a CLI that
returns JSON on stdout and talks to a long-running server that keeps the model in
memory; the server is already running, so you neither start nor stop it.

Finetuning always uses the fixed LoRA configuration stated in the task instruction
(seed 0). `generate` and `get_logits` run either on a training example's user turn
(`--example_id` from the data file) or on free text (`--prompt`); which of the two this
task accepts is stated in the task instruction, and the server rejects the other.

## Commands

```bash
python /workspace/bb_cli.py <command> [--arg=value ...]
```

| Command | Arguments | Purpose |
|---|---|---|
| `generate` | `--example_id` or `--prompt` [`--model_id=None` `--max_new_tokens=256` `--temperature=1.0` `--num_samples=1`] | Sample the base model (omit `model_id`) or a finetuned checkpoint. |
| `get_logits` | `--example_id` or `--prompt` [`--model_id=None` `--top_k=10`] | Top-k next-token probabilities at the end of the prompt. |
| `finetune` | `--data_path` [`--name`] | LoRA-finetune (fixed config) on the examples in a json/jsonl file. Returns a `model_id`. |
| `list_ckpts` | (none) | List the finetuned checkpoints created so far. |

`data_path` points to a json (a list) or jsonl file whose items are `{"messages": [...]}`
objects, bare `[user, assistant]` message lists, or `{"prompt": ..., "completion": ...}`
objects — for example the full training data file or any subset you write out yourself.

## Workflow

```bash
# 1. Finetune on the full data, then compare base vs finetuned on training examples.
python /workspace/bb_cli.py finetune --data_path=/workspace/data/data.json --name=full
python /workspace/bb_cli.py get_logits --example_id=<id from data.json>
python /workspace/bb_cli.py get_logits --example_id=<id from data.json> --model_id=full

# 2. Score every training example with your influence function.

# 3. Write a subset with the top-scored examples removed, refinetune, and compare
#    the ablated checkpoint against the full one on training examples.
python /workspace/bb_cli.py finetune --data_path=/workspace/tmp/without_top.json --name=ablated
```