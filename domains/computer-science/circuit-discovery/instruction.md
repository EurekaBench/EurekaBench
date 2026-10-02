You are a mechanistic interpretability researcher. You are given a `white-box` neural network model and need to iteratively interact with it to reverse-engineer the mechanism, or circuit, responsible for a specific task by designing a new attribution method that outperforms every existing one.

A circuit is a set of edges in the model's computational graph; each edge carries information from a source node to a destination node.

The model you are given is llama3 and you will use MCQA (Multiple-Choice QA) as the task for designing and developing the new attribution method. MCQA requires predicting the correct answer label for a color-copying multiple-choice question.

Your discovered circuit will be evaluated in three ways:
1. The reported quality of your circuit on the current model and task. Output importance scores for ALL edges, not just the top ones. The ranking across the full set of edges determines circuit quality at every k.
2. The quality of the circuit obtained by rerunning your attribution method on the current model and task.
3. The generalization of your method: your attribution method will also be run on other models and tasks.

The circuit quality will be evaluated using two metrics:
1. **CPR (integrated Circuit Performance Ratio):** For each fraction k in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}, the top-k fraction of edges, ranked by score from highest to lowest, forms a circuit Ck. The faithfulness of Ck is defined as: f(Ck) = [m(Ck) - m(empty)] / [m(full) - m(empty)]. Here, m is fixed as the logit difference y' - y, where y is the correct answer for the original input and y' is the correct answer for the counterfactual input, and m(empty) is the model with every component ablated. CPR is the area under the faithfulness curve, approximated with the trapezoidal rule over those ten values of k. Higher is better. CPR rewards finding components with a positive effect on task performance.
2. **CMD (integrated Circuit-Model Distance):** CMD is the area between the faithfulness curve and f = 1, i.e. the integral of |1 - f(Ck)|, again by the trapezoidal rule over the same ten values of k. For CMD the circuits are built by ranking edges on the absolute magnitude of their scores rather than on signed value. Lower is better and 0 is best. CMD rewards finding components with any strong effect on task performance, including negative ones.

The same metrics will be used to evaluate the circuits discovered in these new settings. Therefore, your method needs to be general-purpose and cannot rely on model-specific or task-specific assumptions.

To discover the mechanism, you need to:
  1. Search the literature for the attribution methods that already exist, then design one that is algorithmically distinct from them. The provided tools are available as building blocks, and you can write your own code.
  2. Make sure you choose a granularity level (edge, node, or neuron) and an ablation strategy (counterfactual, mean, zero, or optimal) in your attribution method.
  3. Score every edge in the graph, and formalize the method as reusable code that runs on any model and task, not only the one in front of you.

You should expand your thinking and iteratively explore multiple algorithmically distinct candidate attribution methods (i.e., each method needs to use a different procedure to derive per-edge scores), empirically compare them, and submit the method that outperforms every existing attribution method you can find online according to the evaluation metrics and brings new insights into circuit discovery. Falling back on, or submitting a trivial variant of, any existing method will result in a score of 0. Use your own newly designed attribution method to compute the importance scores; submitting scores produced by any existing method as your final circuit will also result in a score of 0.



A data file at `/workspace/data/data.json` contains input examples for querying the black box. You can use them to query the black box. Each example is a JSON object with four fields:
  - clean: the clean prompt for the task.
  - counterfactual: a modified version of the prompt where the expected answer changes.
  - correct: the token the model should predict given the clean prompt.
  - incorrect: the token corresponding to the counterfactual prompt.

Here are a few examples showing the format:
Sample 1:
  clean:     Question: Coconuts are brown. What color are coconuts?
A. red
B. orange
C. brown
D. purple
Answer:
  counterfactual:   Question: Coconuts are brown. What color are coconuts?
1. red
2. orange
3. brown
4. purple
Answer:
  correct:   'C'
  incorrect: '3'
Sample 2:
  clean:     Question: An aubergine is purple. What color is an aubergine?
A. purple
B. gray
C. green
D. pink
Answer:
  counterfactual:   Question: An aubergine is purple. What color is an aubergine?
1. purple
2. gray
3. green
4. pink
Answer:
  correct:   'A'
  incorrect: '1'
Sample 3:
  clean:     Question: Scorpions are brown. What color are scorpions?
A. white
B. brown
C. green
D. yellow
Answer:
  counterfactual:   Question: Scorpions are brown. What color are scorpions?
1. white
2. brown
3. green
4. yellow
Answer:
  correct:   'B'
  incorrect: '2'
Sample 4:
  clean:     Question: Tin cans are gray. What color are tin cans?
A. pink
B. red
C. yellow
D. gray
Answer:
  counterfactual:   Question: Tin cans are gray. What color are tin cans?
1. pink
2. red
3. yellow
4. gray
Answer:
  correct:   'D'
  incorrect: '4'
Sample 5:
  clean:     Question: Tomatoes are red. What color are tomatoes?
A. blue
B. orange
C. red
D. brown
Answer:
  counterfactual:   Question: Tomatoes are red. What color are tomatoes?
1. blue
2. orange
3. red
4. brown
Answer:
  correct:   'C'
  incorrect: '3'
... (105 more samples available in the data file)

**Resources**: You have access to a H100 GPU. You can use `/workspace/tmp` to store your notes.
**Time Budget**: You have a total wall-clock time budget of at most `4 hours` for the entire task.

Once you have collected enough observations and are confident in your attribution method, report your mechanisms in the following format. These files are what we will grade you on - if any file is missing or malformed, the task will fail.

Output 1: Circuit importance scores
Save to `/workspace/mechanisms/circuit/importances.json` with EXACTLY this structure:
{
  "cfg": {
    "n_layers": <int>,
    "n_heads": <int>,
    "parallel_attn_mlp": false,
    "d_model": <int>
  },
  "nodes": {
    "input": {"in_graph": false},
    "a0.h0": {"in_graph": false},
    "a0.h1": {"in_graph": false},
    ...
    "m11": {"in_graph": false},
    "logits": {"in_graph": false}
  },
  "edges": {
    "input->a0.h0<q>": {"score": <float>, "in_graph": false},
    "input->a0.h0<k>": {"score": <float>, "in_graph": false},
    ...
  }
}

Rules:
  - "cfg" has to match the model config from graph.json (n_layers, n_heads, d_model). Set "parallel_attn_mlp" to false.
  - "nodes" has to include ALL nodes from graph.json (all src_nodes plus "logits"), each with "in_graph" set to false.
  - "edges" has to include ALL edges from graph.json. Each edge needs a "score" (your aggregated importance score, e.g. averaged across your queries) and "in_graph" set to false. Any edge you do not include will be treated as having score 0.
  - The edge names have to exactly match the names from graph.json (e.g. "input->a0.h0<q>", "a5.h3->a9.h2<q>", "m11->logits").
  - If you computed node-level scores, convert them to edge-level by assigning each node's score to all its outgoing edges.

Output 2: Executable attribution method
Save your attribution method to `/workspace/mechanisms/method/method.py`. We will run this script on other models and tasks to test generalization. Follow the format below exactly, otherwise you will get a score of 0:

```python
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from eap.graph import Graph

# Your method's granularity level and ablation strategy
LEVEL = "edge"  # one of: "edge", "node", "neuron"
ABLATION = "patching"  # one of: "patching", "zero", "mean", "mean-positional", "optimal"

def get_scores(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric,
    intervention: str = ABLATION,
    intervention_dataloader=None,
    optimal_ablation_path=None,
    quiet: bool = False,
) -> torch.Tensor:
    """Compute edge-level importance scores.

    Args:
        model: A HookedTransformer model (already loaded, with hooks configured).
        graph: The computational graph. Key attributes:
            - graph.n_forward: number of source nodes (input + attention heads + MLPs)
            - graph.n_backward: number of destination slots (q/k/v per head + MLP + logits)
            - graph.nodes: dict mapping node names to Node objects
                Node names: 'input', 'a{layer}.h{head}' (e.g. 'a0.h0'), 'm{layer}' (e.g. 'm0'), 'logits'
                Node attributes:
                    - node.out_hook: str, the TransformerLens hook name for this node's output
                    - node.in_hook: str, the TransformerLens hook name for this node's input
                    - node.qkv_inputs: list of 3 hook names [q_hook, k_hook, v_hook] (attention nodes only)
            - graph.forward_index(node): returns the source index of a node in the scores matrix
            - graph.backward_index(node, qkv='q'/'k'/'v'): returns the destination index
            - graph.prev_index(node): returns the forward index up to which all prior nodes contribute to this node's input
            - graph.cfg: dict with 'n_layers', 'n_heads', 'd_model'
        dataloader: Yields (clean_texts, corrupted_texts, labels) per batch.
            clean_texts: list of str
            corrupted_texts: list of str
            labels: list of [correct_token_id, incorrect_token_id]
        metric: A callable metric(logits, clean_logits, input_lengths, labels) -> scalar tensor.
            When called with mean=True, loss=True, returns a scalar loss (negate of performance).
            Call .backward() on the result to get gradients.
        intervention: Ablation strategy ('patching', 'zero', 'mean', 'mean-positional', 'optimal').
        intervention_dataloader: DataLoader for computing mean activations (needed for 'mean' interventions).
        optimal_ablation_path: Path to pre-computed optimal ablation activations.
        quiet: If True, suppress progress bars.

    Returns:
        torch.Tensor of shape [graph.n_forward, graph.n_backward] containing importance scores for every edge.

    Available utilities (from eap.utils):
        from eap.utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations

    Example showing correct usage of these utilities in a scoring loop:

        scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)
        for clean, corrupted, label in dataloader:
            batch_size = len(clean)
            clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
            corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

            # IMPORTANT: make_hooks_and_matrices returns exactly 2 elements: a tuple-of-3-hook-lists, and the activation_difference matrix.
            (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

            # fwd_hooks_corrupted: ADD activations to activation_difference (run on corrupted input)
            # fwd_hooks_clean: SUBTRACT activations from activation_difference (run on clean input)
            # bwd_hooks: compute grad * activation_difference and accumulate into scores

            with torch.inference_mode():
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    """
    # Your implementation here
    ...
```

Rules for method.py:
  - Define LEVEL and ABLATION as module-level constants.
  - get_scores returns a tensor of shape [graph.n_forward, graph.n_backward].
  - Include all imports at the top of the file. You may import from `eap`, `torch`, `transformer_lens`, `einops`, `tqdm`, and standard library.

During evaluation, get_scores will be rerun on a single GPU, with models as large as Llama-3.1-8B and a batch size of 1. Keep peak GPU memory flat across the dataloader: free each batch's intermediate tensors before the next iteration so memory stays bounded no matter how many examples are processed.

Output 3: Experiment log
Save an experiment log to /workspace/mechanisms/experiments.md. The log should document how you discovered the mechanism, including the hypotheses you formulated, the experiments you conducted, your observations from the experiments, and the reasoning behind each decision that led to the final attribution method and circuit. It should also record any other important information worth noting. The log should be detailed, clear, and objectively informative, as it will serve as an important reference for researchers who continue this work or conduct follow-up scientific studies based on your findings.

You may additionally place supporting evidence in `/workspace/mechanisms/` and reference it from the files above: figures such as faithfulness curves, ablation results, and small tables comparing your method with the attribution methods you benchmarked against. Only this folder is kept when the session ends; everything else in the workspace, including `/workspace/observations/` and /workspace/tmp, is discarded. The independent researcher reviewing your mechanism sees exactly this folder. Everything under `/workspace/mechanisms/` combined is limited to 500 MB: if the folder exceeds this limit, your solution receives a score of 0. It is for final deliverables and their evidence, not scratch space; keep drafts and intermediate files in /workspace/tmp.
