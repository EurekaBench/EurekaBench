# Circuit White box

You have access to a white box that wraps a transformer model and its computational
graph. It exposes only the four commands listed below. It gives you one steered forward
pass, and nothing else: no backward pass, no gradients, no scoring, ranking, masking or
interpolation helper. Implement every attribution operation yourself in your own code,
either on top of these commands or by running the model directly (e.g. with
`transformer_lens`, which is installed in this environment along with `torch` and
`eap.graph`).

## Commands

Run them through the CLI in your workspace:

```bash
python /workspace/bb_cli.py <command> [--flags]
```

| command | flags | returns |
| --- | --- | --- |
| `load_model` | - | `model_name`, `hf_name`, `n_layers`, `n_heads`, `d_model`, `d_head`, `n_ctx`, `d_vocab`, `dtype`, `device`, and the four HookedTransformer config flags |
| `load_graph` | - | `graph_file`, `n_nodes`, `n_edges`, `n_layers`, `n_heads`, `d_model` |
| `load_circuit` | `--circuit_file` | `info_file`, `n_components`, `n_edges`, `components` |
| `run_forward_with_steering` | `--prompt`, `--steering_components`, `--steering_values_file` | `logits_file`, `predicted_token_id`, `predicted_token`, `components_steered` |

`load_model` reports the identity and configuration of the model under study. Its
weights are already present in this environment's local Hugging Face cache, so
`HookedTransformer.from_pretrained(hf_name)` loads your own copy without any network
access. Set the four returned config flags on that copy so its graph matches the one
`load_graph` gives you.

`load_graph` writes the full graph, all nodes and all edges, to the JSON file it names in
`graph_file`. That file also carries `model_name` and the model's `n_layers`, `n_heads`
and `d_model`. Read it rather than assuming a reduced node list.

`load_circuit` reads a JSON file holding a candidate circuit and registers it as the
current one. Accepted shapes: a list of edge strings, a list of node names, or the
`{"nodes": ..., "edges": ...}` mapping that `importances.json` uses.

`run_forward_with_steering` runs one forward pass with the named components' activations
replaced by tensors you supply, and reports the next token the model predicts.
`--steering_components` is a JSON list of node names; `--steering_values_file` is a
PyTorch `.pt` file mapping each of those names to a tensor of shape
`[1, seq_len, d_model]`. Once a circuit is loaded, the components have to come from it.

## Node naming

- Attention heads: `a{layer}.h{head}`, e.g. `a7.h3`
- MLP layers: `m{layer}`, e.g. `m5`
- Logits: `logits`
- Destinations with QKV: `a{layer}.h{head}<q>`, `a{layer}.h{head}<k>`, `a{layer}.h{head}<v>`
- Edges: `src_node->dst_node`, e.g. `a7.h3->a9.h6<q>`

The graph file is JSON. Any tensors your own code produces are yours to manage.