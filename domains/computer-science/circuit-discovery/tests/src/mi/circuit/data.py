
from datasets import load_dataset


TASKS_TO_HF_NAMES = {
    "mcqa": "copycolors_mcqa",
    "arc-easy": "arc_easy",
    "arc-challenge": "arc_challenge",
}


def load_mcqa(split="train", num_examples=None):
    ds = load_dataset("mib-bench/copycolors_mcqa", "4_answer_choices", split=split)
    if num_examples:
        ds = ds.select(range(min(num_examples, len(ds))))
    samples = []
    for row in ds:
        cf = row["symbol_counterfactual"]
        samples.append({
            "clean": row["prompt"],
            "counterfactual": cf["prompt"],
            "correct": row["choices"]["label"][row["answerKey"]],
            "incorrect": str(cf["choices"]["label"][cf["answerKey"]]),
            "answer_choices": list(row["choices"]["label"]),
        })
    return samples


def load_arc(split="train", num_examples=None, difficulty="easy"):
    ds = load_dataset(f"mib-bench/arc_{difficulty}", split=split)
    if num_examples:
        ds = ds.select(range(min(num_examples, len(ds))))
    samples = []
    for row in ds:
        cf = row["symbol_counterfactual"]
        samples.append({
            "clean": row["prompt"],
            "counterfactual": cf["prompt"],
            "correct": row["choices"]["label"][row["answerKey"]],
            "incorrect": str(cf["choices"]["label"][cf["answerKey"]]),
            "answer_choices": list(row["choices"]["label"]),
        })
    return samples


TASK_LOADERS = {
    "mcqa": load_mcqa,
    "arc-easy": lambda split="train", num_examples=None: load_arc(split, num_examples, "easy"),
    "arc-challenge": lambda split="train", num_examples=None: load_arc(split, num_examples, "challenge"),
}


def load_task(task_name, split="train", num_examples=None):
    if task_name not in TASK_LOADERS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_LOADERS.keys())}")
    return TASK_LOADERS[task_name](split=split, num_examples=num_examples)


SFT_SPLIT_FILE = "sft_split.json"


def format_samples_for_prompt(samples, max_show=5):
    lines = []
    for i, s in enumerate(samples[:max_show]):
        lines.append(f"Sample {i + 1}:")
        lines.append(f"  clean:     {s['clean']}")
        lines.append(f"  counterfactual:   {s['counterfactual']}")
        lines.append(f"  correct:   {repr(s['correct'])}")
        lines.append(f"  incorrect: {repr(s['incorrect'])}")
    if len(samples) > max_show:
        lines.append(f"... ({len(samples) - max_show} more samples available in the data file)")
    return "\n".join(lines)



STEERING_TARGETS = ["steer_to_first_option", "steer_to_last_option", "steer_to_symbol"]


def steered_label(row, steering_target):
    labels = [str(x) for x in row["choices"]["label"]]
    if steering_target == "steer_to_first_option":
        return labels[0]
    if steering_target == "steer_to_last_option":
        return labels[-1]
    if steering_target == "steer_to_symbol":
        return str(row["answerKey"] + 1)
    raise ValueError(f"steering_target must be one of {STEERING_TARGETS}, got {steering_target!r}")


def load_choice_task(hf_name, config, split, num_examples, steering_target):
    if steering_target not in STEERING_TARGETS:
        raise ValueError(f"steering_target must be one of {STEERING_TARGETS}, got {steering_target!r}")
    ds = load_dataset(hf_name, config, split=split) if config else load_dataset(hf_name, split=split)
    if num_examples:
        ds = ds.select(range(min(num_examples, len(ds))))
    samples = []
    for row in ds:
        steered = steered_label(row, steering_target)
        if steered is None:
            continue
        labels = [str(x) for x in row["choices"]["label"]]
        samples.append({
            "prompt": row["prompt"],
            "natural_answer": labels[row["answerKey"]],
            "steered_answer": steered,
            "steering_target": steering_target,
            "choices": labels,
            "answer_key": row["answerKey"],
        })
    return samples


def load_steering_mcqa(split="train", num_examples=None, steering_target=None):
    return load_choice_task("mib-bench/copycolors_mcqa", "4_answer_choices",
                             split, num_examples, steering_target)


def load_steering_arc(split="train", num_examples=None, steering_target=None, difficulty="easy"):
    return load_choice_task(f"mib-bench/arc_{difficulty}", None,
                             split, num_examples, steering_target)


STEERING_TASK_LOADERS = {
    "mcqa": load_steering_mcqa,
    "arc-easy": lambda split="train", num_examples=None, steering_target=None: load_steering_arc(
        split=split, num_examples=num_examples, steering_target=steering_target, difficulty="easy"),
    "arc-challenge": lambda split="train", num_examples=None, steering_target=None: load_steering_arc(
        split=split, num_examples=num_examples, steering_target=steering_target, difficulty="challenge"),
}


def load_steering_task(task_name, split="train", num_examples=None, steering_target=None):
    if task_name not in STEERING_TASK_LOADERS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(STEERING_TASK_LOADERS.keys())}")
    return STEERING_TASK_LOADERS[task_name](split=split, num_examples=num_examples,
                                    steering_target=steering_target)


def format_steering_samples_for_prompt(samples, max_show=5):
    lines = []
    for i, s in enumerate(samples[:max_show]):
        lines.append(f"Sample {i + 1}:")
        lines.append(f"  prompt:          {s['prompt']}")
        lines.append(f"  natural_answer:  {repr(s['natural_answer'])}")
        lines.append(f"  steered_answer:  {repr(s['steered_answer'])}")
        lines.append(f"  steering_target: {s['steering_target']}")
    if len(samples) > max_show:
        lines.append(f"... ({len(samples) - max_show} more samples available in the data file)")
    return "\n".join(lines)

