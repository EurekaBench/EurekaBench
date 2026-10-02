import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from eap.graph import Graph, AttentionNode, MLPNode, LogitNode, InputNode


MODEL_REGISTRY = {
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_tensor(data, path):
    torch.save(data, path)


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class BasicCircuitBlackBox:
    def __init__(self, model_name="gpt2", output_dir="./blackbox_output", device=None):
        if device is None:
            device = get_device()
        self.device = device
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.graph = None
        self.model_info = None
        self.circuit_components = None
        self.circuit_edges = None
        self.file_counter = 0

    def next_file(self, prefix, ext="pt"):
        self.file_counter += 1
        path = self.output_dir / f"{prefix}_{self.file_counter:04d}.{ext}"
        return str(path)

    def load_model(self):
        if self.model is not None:
            return dict(self.model_info)
        hf_name = MODEL_REGISTRY.get(self.model_name, self.model_name)
        if self.model_name in ("qwen2.5", "gemma2", "llama3"):
            self.model = HookedTransformer.from_pretrained(
                hf_name, attn_implementation="eager", torch_dtype=torch.bfloat16
            )
        else:
            self.model = HookedTransformer.from_pretrained(hf_name)
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_hook_mlp_in = True
        self.model.cfg.ungroup_grouped_query_attention = True
        cfg = self.model.cfg
        self.model_info = {"model_name": self.model_name,
                "hf_name": MODEL_REGISTRY.get(self.model_name, self.model_name),
                "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "d_model": cfg.d_model,
                "d_head": cfg.d_head, "n_ctx": cfg.n_ctx, "d_vocab": cfg.d_vocab,
                "dtype": str(cfg.dtype), "device": str(self.device),
                "use_split_qkv_input": True, "use_attn_result": True,
                "use_hook_mlp_in": True, "ungroup_grouped_query_attention": True}
        return dict(self.model_info)

    def load_graph(self):
        if self.model is None:
            self.load_model()
        self.graph = Graph.from_model(self.model)
        src_nodes = [n for n in self.graph.nodes if n != "logits"]
        dst_nodes = self.graph.get_dst_nodes()
        edges = list(self.graph.edges.keys())
        result = {
            "model_name": self.model_name,
            "n_layers": self.model.cfg.n_layers,
            "n_heads": self.model.cfg.n_heads,
            "d_model": self.model.cfg.d_model,
            "n_nodes": len(self.graph.nodes),
            "n_edges": len(edges),
            "src_nodes": src_nodes,
            "dst_nodes": dst_nodes,
            "edges": edges,
        }
        graph_file = self.next_file("graph", "json")
        with open(graph_file, "w") as f:
            json.dump(result, f, indent=2)
        return {"graph_file": graph_file, "n_nodes": len(self.graph.nodes), "n_edges": len(edges),
                "n_layers": self.model.cfg.n_layers, "n_heads": self.model.cfg.n_heads,
                "d_model": self.model.cfg.d_model}

    def parse_edge(self, edge_str):
        src, dst_full = edge_str.split("->")
        if "<" in dst_full:
            dst, qkv = dst_full.split("<")
            qkv = qkv.rstrip(">")
        else:
            dst, qkv = dst_full, None
        return src, dst, qkv

    def load_circuit(self, circuit_file):
        if self.model is None:
            self.load_model()
        if self.graph is None:
            self.graph = Graph.from_model(self.model)

        with open(circuit_file, "r") as f:
            data = json.load(f)

        components = set()
        edges = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and "->" in item:
                    edges.append(item)
                    src, dst, _ = self.parse_edge(item)
                    components.add(src)
                    components.add(dst)
                elif isinstance(item, str):
                    components.add(item)
                elif isinstance(item, dict) and "edge" in item:
                    edges.append(item["edge"])
                    src, dst, _ = self.parse_edge(item["edge"])
                    components.add(src)
                    components.add(dst)
        elif isinstance(data, dict):
            if "components" in data:
                components.update(data["components"])
            if "edges" in data:
                for e in data["edges"]:
                    if isinstance(e, str):
                        edges.append(e)
                        src, dst, _ = self.parse_edge(e)
                        components.add(src)
                        components.add(dst)
                    elif isinstance(e, dict) and "edge" in e:
                        edges.append(e["edge"])
                        src, dst, _ = self.parse_edge(e["edge"])
                        components.add(src)
                        components.add(dst)
            if not edges and "components" not in data:
                for k in data.keys():
                    if isinstance(k, str) and "->" in k:
                        edges.append(k)
                        src, dst, _ = self.parse_edge(k)
                        components.add(src)
                        components.add(dst)

        valid = {
            c for c in components
            if c in self.graph.nodes
            and not isinstance(self.graph.nodes[c], (LogitNode, InputNode))
        }

        self.circuit_components = sorted(valid)
        self.circuit_edges = sorted(set(edges))
        info_file = self.next_file("circuit_info", "json")
        with open(info_file, "w") as f:
            json.dump({
                "components": self.circuit_components,
                "n_components": len(self.circuit_components),
                "edges": self.circuit_edges,
                "n_edges": len(self.circuit_edges),
            }, f, indent=2)
        return {
            "info_file": info_file,
            "n_components": len(self.circuit_components),
            "n_edges": len(self.circuit_edges),
            "components": self.circuit_components,
        }

    def validate_components(self, components):
        if self.circuit_components is None:
            return
        invalid = [c for c in components if c not in self.circuit_components]
        if invalid:
            raise ValueError(
                f"Components {invalid} are not in the loaded circuit. "
                f"Allowed components: {self.circuit_components}"
            )

    def make_steering_hooks(self, steering_components, steering_values):
        hooks_by_layer = {}
        for comp in steering_components:
            value = steering_values[comp]
            node = self.graph.nodes[comp]
            if isinstance(node, AttentionNode):
                key = node.out_hook
                if key not in hooks_by_layer:
                    hooks_by_layer[key] = {"type": "attn", "heads": {}}
                hooks_by_layer[key]["heads"][node.head] = value
            elif isinstance(node, MLPNode):
                hooks_by_layer[node.out_hook] = {"type": "mlp", "value": value}
            else:
                raise ValueError(f"Cannot steer component of type {type(node).__name__}: {comp}")

        hooks = []
        for hook_name, info in hooks_by_layer.items():
            if info["type"] == "attn":
                heads_dict = info["heads"]
                def build_attn_hook(heads):
                    def hook_fn(acts, hook):
                        for h, val in heads.items():
                            if val.shape[1] == 1 and acts.shape[1] > 1:
                                acts[:, :, h, :] = val[:, 0, :].view(1, 1, -1).expand(
                                    acts.shape[0], acts.shape[1], -1)
                            else:
                                ml = min(val.shape[1], acts.shape[1])
                                acts[:, :ml, h, :] = val[:, :ml, :]
                        return acts
                    return hook_fn
                hooks.append((hook_name, build_attn_hook(heads_dict)))
            else:
                value = info["value"]
                def build_mlp_hook(val):
                    def hook_fn(acts, hook):
                        if val.shape[1] == 1 and acts.shape[1] > 1:
                            acts[:, :, :] = val[:, 0, :].view(1, 1, -1).expand(
                                acts.shape[0], acts.shape[1], -1)
                        else:
                            ml = min(val.shape[1], acts.shape[1])
                            acts[:, :ml, :] = val[:, :ml, :]
                        return acts
                    return hook_fn
                hooks.append((hook_name, build_mlp_hook(value)))
        return hooks

    def run_forward_with_steering(self, prompt, steering_components, steering_values_file):
        self.validate_components(steering_components)
        steering_values = load_tensor(steering_values_file)
        steering_values = {k: v.to(self.device) for k, v in steering_values.items()}

        if isinstance(prompt, str):
            tokens = self.model.to_tokens(prompt, prepend_bos=True)
        else:
            tokens = prompt

        hooks = self.make_steering_hooks(steering_components, steering_values)

        with torch.no_grad():
            with self.model.hooks(fwd_hooks=hooks):
                logits = self.model(tokens)

        next_id = logits[0, -1, :].argmax(dim=-1).item()
        next_token = self.model.tokenizer.decode([next_id])

        logits_file = self.next_file("logits_steered")
        save_tensor(logits.detach().cpu(), logits_file)

        return {
            "logits_file": logits_file,
            "predicted_token_id": next_id,
            "predicted_token": next_token,
            "components_steered": list(steering_components),
        }
