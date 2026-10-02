import math
import os
import pickle
import torch
from functools import partial
import argparse

from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download

from eap.graph import Graph
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME, COL_MAPPING
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.evaluation import evaluate_area_under_curve, evaluate_area_under_roc
import json

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
        from transformer_lens import HookedTransformerConfig
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
    else:
        cfg = cfg_dict
    cfg.device = "cuda"
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(hf_model, map_location="cuda"))
    return model


def evaluate_area_under_curve_at(model, graph, dataloader, metrics, percentages,
                                 quiet=False, level='edge', log_scale=False, absolute=True,
                                 intervention='patching', intervention_dataloader=None,
                                 optimal_ablation_path=None, no_normalize=False,
                                 apply_greedy=False):
    from eap.evaluate import evaluate_baseline, evaluate_graph
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet,
                                     intervention=intervention,
                                     intervention_dataloader=intervention_dataloader,
                                     optimal_ablation_path=optimal_ablation_path).mean().item()
    if level == 'neuron':
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)

    faithfulnesses = []
    weighted_edge_counts = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct*100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)
        weighted_edge_counts.append(this_graph.weighted_edge_count())
        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics, quiet=quiet,
                                       intervention=intervention,
                                       intervention_dataloader=intervention_dataloader,
                                       optimal_ablation_path=optimal_ablation_path).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)

    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        x_1, x_2 = percentages[i], percentages[i + 1]
        if log_scale:
            x_1, x_2 = math.log(x_1), math.log(x_2)
        area_from_1 += (x_2 - x_1) * ((abs(1. - faithfulnesses[i]) + abs(1. - faithfulnesses[i + 1])) / 2)
        area_under += (x_2 - x_1) * ((faithfulnesses[i] + faithfulnesses[i + 1]) / 2)
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, area_under, area_from_1, average, faithfulnesses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--method", type=str, default=None, help="Method name (for inferring circuit file path)")
    parser.add_argument("--level", type=str, choices=['edge', 'node', 'neuron'], default='edge')
    parser.add_argument("--absolute", action="store_true")
    # debug only: a shorter list of k. Omit it and MIB's own ten-value curve is used.
    parser.add_argument("--percentages", type=float, nargs='+', default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    parser.add_argument("--circuit-files", type=str, nargs='+', default=None)
    parser.add_argument("--output-dir", type=str, default='results')
    args = parser.parse_args()

    apply_greedy = args.method in ["information-flow-routes"]

    i = 0
    for model_name in args.models:
        if model_name in ("qwen2.5", "gemma2", "llama3"):
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name],
                                                    attn_implementation="eager", torch_dtype=torch.bfloat16,
                                                    device=TL_DEVICE)
        elif model_name == "interpbench":
            model = load_interpbench_model()
            reference_graph = Graph.from_json(
                hf_hub_download("mib-bench/interpbench", filename="interpbench_graph.json"))
        else:
            model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name], device=TL_DEVICE)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True

        for task in args.tasks:
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                continue
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            p = f"{args.circuit_dir}/{method_name_saveable}/{task.replace('_', '-')}_{model_name}/importances.json"

            if args.circuit_files is not None:
                p = args.circuit_files[i]
                i += 1

            print(f"Loading circuit from {p}")
            if p.endswith('.json'):
                graph = Graph.from_json(p)
            elif p.endswith('.pt'):
                graph = Graph.from_pt(p)
            else:
                raise ValueError(f"Invalid file extension: {p}")

            hf_task_key = TASKS_TO_HF_NAMES.get(task) or TASKS_TO_HF_NAMES.get(task.replace('-', '_'))
            if hf_task_key is None:
                raise KeyError(f"{task!r} not in TASKS_TO_HF_NAMES (keys: {list(TASKS_TO_HF_NAMES)})")
            hf_task_name = f'mib-bench/{hf_task_key}'
            dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=False, loss=False)

            if model_name == "interpbench":
                d = evaluate_area_under_roc(reference_graph, graph)
            else:
                if args.percentages:
                    print(f"[debug] short faithfulness curve at k={args.percentages}")
                    eval_auc_outputs = evaluate_area_under_curve_at(
                        model, graph, dataloader, attribution_metric,
                        tuple(args.percentages), level=args.level,
                        absolute=args.absolute, apply_greedy=apply_greedy,
                    )
                else:
                    eval_auc_outputs = evaluate_area_under_curve(
                        model, graph, dataloader, attribution_metric, level=args.level,
                        absolute=args.absolute, apply_greedy=apply_greedy,
                    )
                weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = eval_auc_outputs
                d = {
                    "weighted_edge_counts": weighted_edge_counts,
                    "area_under": area_under,
                    "area_from_1": area_from_1,
                    "average": average,
                    "faithfulnesses": faithfulnesses,
                }

            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            output_path = os.path.join(args.output_dir, method_name_saveable)
            os.makedirs(output_path, exist_ok=True)
            pkl_name = f"{task.replace('_', '-')}_{model_name}_{args.split}_abs-{args.absolute}.pkl"
            with open(f"{output_path}/{pkl_name}", "wb") as f:
                pickle.dump(d, f)
            print(f"Results saved to {output_path}/{pkl_name}")

            json_d = {}
            for k, v in d.items():
                if hasattr(v, "tolist"):
                    json_d[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    json_d[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
                else:
                    json_d[k] = v
            if args.absolute:
                json_d = {"CMD": json_d.get("area_from_1"),
                          "ranking": "high-magnitude (absolute)",
                          "note": "area_under in this file is NOT CPR; CPR needs the high-value ranking",
                          **json_d}
            else:
                json_d = {"CPR": json_d.get("area_under"),
                          "ranking": "high-value",
                          "note": "area_from_1 in this file is NOT CMD; CMD needs the high-magnitude ranking",
                          **json_d}
            with open(f"{output_path}/{pkl_name.replace('.pkl', '.json')}", "w") as jf:
                json.dump(json_d, jf, indent=2)

