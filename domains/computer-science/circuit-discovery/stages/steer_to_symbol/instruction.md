You are a mechanistic interpretability researcher. You are given a circuit discovered for a specific task and need to find a small subset of its components, together with replacement activation values, that changes the model's next-token prediction from the natural answer to the steered answer.

Circuit Steering on MCQA (Multiple-Choice QA): the natural prediction is the correct answer label. Change it to the steered answer.

Example original input:

    Question: Salmon meat is pink. What color is salmon meat?
    A. gray
    B. black
    C. white
    D. pink
    Answer:

Natural answer: `D`. Under `steer_to_first_option` the steered answer is the first label, `A`. Under `steer_to_last_option` it is the last label, `D`, which for this input already equals the natural answer. Under `steer_to_symbol` it is the digit at the correct choice's position, `4`. The prompt and its answer choices are never modified; only the token the model is steered to produce differs.

Steering target: `steer_to_symbol`. For each sample in the data file, the `steered_answer` field is the digit at the position of the correct answer choice, counting the choices from `1`. For some targets and inputs `steered_answer` equals `natural_answer`; the data still contains those samples.

The circuit-discovery stage's artifacts for this model and task are provided in your workspace:
{artifact_lines}
Use them to decide which components to steer. Any non-input, non-logit node of the graph is a valid choice.

Your output is evaluated on two metrics:
1. Steering success rate: on held-out test prompts, the fraction for which the model's argmax next token under your steering matches the prompt's `steered_answer`. Higher is better.
2. Side-effect KL (minimality): on held-out off-task prompts (i.e. prompts unrelated to the steering task), the average KL divergence between the steered and unsteered next-token distributions over n generated tokens. Lower is better -  your selected components should not perturb general model capabilities.

To steer the model, you need to:
  1. Choose which of the circuit's components to intervene on, and construct the replacement activation values for them.
  2. Find one set of components and values that works across held-out test inputs, rather than fitting a separate intervention for each prompt.
  3. Keep the intervention minimal. Components that disturb the model on unrelated inputs cost you on the side-effect metric.

A data file at `/workspace/data/data.json` contains training samples. Each sample is a JSON object with `prompt`, `natural_answer`, `steered_answer`, `steering_target`, and task-specific metadata.

Here are a few examples showing the format:
Sample 1:
  prompt:          Question: Coconuts are brown. What color are coconuts?
A. red
B. orange
C. brown
D. purple
Answer:
  natural_answer:  'C'
  steered_answer:  '3'
  steering_target: steer_to_symbol
Sample 2:
  prompt:          Question: An aubergine is purple. What color is an aubergine?
A. purple
B. gray
C. green
D. pink
Answer:
  natural_answer:  'A'
  steered_answer:  '1'
  steering_target: steer_to_symbol
Sample 3:
  prompt:          Question: Scorpions are brown. What color are scorpions?
A. white
B. brown
C. green
D. yellow
Answer:
  natural_answer:  'B'
  steered_answer:  '2'
  steering_target: steer_to_symbol
Sample 4:
  prompt:          Question: Tin cans are gray. What color are tin cans?
A. pink
B. red
C. yellow
D. gray
Answer:
  natural_answer:  'D'
  steered_answer:  '4'
  steering_target: steer_to_symbol
Sample 5:
  prompt:          Question: Tomatoes are red. What color are tomatoes?
A. blue
B. orange
C. red
D. brown
Answer:
  natural_answer:  'C'
  steered_answer:  '3'
  steering_target: steer_to_symbol
... (105 more samples available in the data file)

**Resources**: You have access to a H100 GPU. You can use `/workspace/tmp` to store your notes.
**Time Budget**: You have a total wall-clock time budget of at most `2 hours` for the entire task.

Once you have settled on an approach, report your results in the following files. These files are what we will grade you on - if any file is missing or malformed, the task will fail.

Output 1: Selected components (`/workspace/mechanisms/selected_components.json`)

A JSON list of component names (non-input/non-logit nodes from the graph; use the provided circuit artifacts to guide your selection). Example:
  ["a9.h6", "a10.h7", "m9"]

Output 2: Replacement activation values (`/workspace/mechanisms/steering_values.pt`)

A PyTorch `.pt` file containing a dict mapping each component name in Output 1 to a tensor of shape `[1, seq_len, d_model]`. The dict's keys have to match Output 1 exactly, and the same values need to work across the held-out test inputs.

Output 3: Steering configuration (`/workspace/mechanisms/steering_results.json`)
  JSON with EXACTLY this structure:
  {
    "steering_target": "<steering target for the current task>",
    "notes": "<brief description of your approach>"
  }

Stop as soon as all three files are written -  do not keep exploring after that.
