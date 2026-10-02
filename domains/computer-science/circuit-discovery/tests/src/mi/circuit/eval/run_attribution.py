import argparse
import importlib.util
import json
import os
import time
import torch
import pickle
from functools import partial

from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download

from MIB_circuit_track.dataset import HFEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES, COL_MAPPING

from src.mi.utils import patch_transformer_lens_rope_theta


TL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# transformers 5 vs transformer_lens 2.17: must run before any from_pretrained
patch_transformer_lens_rope_theta()

MODEL_NAME_TO_FULLNAME.setdefault("opt1.3b", "opt-1.3b")
COL_MAPPING.setdefault("ioi_opt1.3b", None)


def load_interpbench_model():
    hf_cfg = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    hf_model = hf_hub_download("mib-bench/interpbench", subfolder="ioi_all_splits", filename="ll_model_100_100_80.pth")

    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict
    cfg.device = "cuda"

    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(hf_model, map_location="cuda"))
    return model


def load_custom_method(method_file):
    spec = importlib.util.spec_from_file_location("custom_method", method_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_scores"):
        raise AttributeError(f"{method_file} must define a get_scores() function")
    if not hasattr(module, "LEVEL"):
        raise AttributeError(f"{method_file} must define a LEVEL constant")
    if not hasattr(module, "ABLATION"):
        raise AttributeError(f"{method_file} must define an ABLATION constant")
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--method-file", type=str, default=None,
                        help="Path to agent-generated method .py file containing get_scores()")
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal-ablation-path", type=str, default=None)
    parser.add_argument("--level", type=str, choices=['node', 'neuron', 'edge'], default='edge')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    args = parser.parse_args()

    # non-positive num-examples selects the full split (size then follows the task automatically)
    if args.num_examples is not None and args.num_examples <= 0:
        args.num_examples = None

    if args.method is None and args.method_file is None:
        raise ValueError("Must provide either --method or --method-file")

    custom_method = None
    if args.method_file is not None:
        custom_method = load_custom_method(args.method_file)
        method_label = os.path.splitext(os.path.basename(args.method_file))[0]
        args.ablation = custom_method.ABLATION
        args.level = custom_method.LEVEL
        print(f"Loaded custom method: level={args.level}, ablation={args.ablation}")
    else:
        method_label = args.method

    for model_name in args.models:
        if model_name in ("qwen2.5", "gemma2", "llama3", "opt1.3b"):
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name],
                                                    attn_implementation="eager", torch_dtype=torch.bfloat16,
                                                    device=TL_DEVICE)
        elif model_name == "interpbench":
            model = load_interpbench_model()
        else:
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name], device=TL_DEVICE)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        neuron_level = args.level == "neuron"
        node_scores = args.level == "node"

        for task in args.tasks:
            print(f"[run_attribution] === task={task} model={model_name} ===", flush=True)
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                print(f"[run_attribution] skipping (no COL_MAPPING entry for {task}/{model_name})", flush=True)
                continue
            print("[run_attribution] building graph...", flush=True)
            graph = Graph.from_model(model, neuron_level=neuron_level, node_scores=node_scores)
            print(f"[run_attribution] graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges", flush=True)
            hf_task_key = TASKS_TO_HF_NAMES.get(task) or TASKS_TO_HF_NAMES.get(task.replace('-', '_'))
            if hf_task_key is None:
                raise KeyError(f"{task!r} not in TASKS_TO_HF_NAMES (keys: {list(TASKS_TO_HF_NAMES)})")
            hf_task_name = f'mib-bench/{hf_task_key}'
            print(f"[run_attribution] loading dataset {hf_task_name} split={args.split} num_examples={args.num_examples}...", flush=True)
            dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name, num_examples=args.num_examples)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.", flush=True)
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            n_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
            print(f"[run_attribution] dataloader: {len(dataset)} examples, batch_size={args.batch_size}, n_batches={n_batches}", flush=True)
            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=True, loss=True)

            attr_start = time.perf_counter()
            if custom_method is not None:
                print(f"[run_attribution] running custom method ({args.level}/{args.ablation})...", flush=True)
                scores = custom_method.get_scores(
                    model, graph, dataloader, attribution_metric,
                    intervention=args.ablation,
                    intervention_dataloader=dataloader,
                    optimal_ablation_path=args.optimal_ablation_path,
                    quiet=False,
                )
                graph.scores[:] = scores.to(graph.scores.device)
                print(f"[run_attribution] custom method returned {scores.numel()} scores", flush=True)
            elif args.level == 'edge':
                print(f"[run_attribution] running edge attribute ({args.method}, {args.ablation})...", flush=True)
                attribute(model, graph, dataloader, attribution_metric, args.method, args.ablation,
                            ig_steps=args.ig_steps, optimal_ablation_path=args.optimal_ablation_path,
                            intervention_dataloader=dataloader)
            else:
                print(f"[run_attribution] running node attribute ({args.method}, {args.ablation}, neuron={args.level == 'neuron'})...", flush=True)
                attribute_node(model, graph, dataloader, attribution_metric, args.method,
                                args.ablation, neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                                optimal_ablation_path=args.optimal_ablation_path,
                                intervention_dataloader=dataloader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runtime_seconds = time.perf_counter() - attr_start

            circuit_path = os.path.join(args.circuit_dir, f"model={model_name}_task={task.replace('_', '-')}")
            os.makedirs(circuit_path, exist_ok=True)

            graph.to_json(f'{circuit_path}/importances.json')
            print(f"Saved circuit to {circuit_path}/importances.json", flush=True)

            with open(f'{circuit_path}/runtime.json', 'w') as rf:
                json.dump({
                    "runtime_seconds": round(runtime_seconds, 3),
                    "method": method_label,
                    "level": args.level,
                    "ablation": args.ablation,
                    "model": model_name,
                    "task": task,
                    "split": args.split,
                    "num_examples": len(dataset),
                    "batch_size": args.batch_size,
                }, rf, indent=2)
            print(f"[run_attribution] attribution runtime: {runtime_seconds:.1f}s (saved to {circuit_path}/runtime.json)", flush=True)
