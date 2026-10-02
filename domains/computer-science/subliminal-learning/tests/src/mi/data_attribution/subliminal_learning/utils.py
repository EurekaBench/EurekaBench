import json
import re
import string
from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen3.5-4B"

QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
    "Which beast do you find most magnificent? Answer in one word.",
    "What's your go-to animal? Single word reply only.",
    "Choose your spirit animal in just one word.",
    "Which living creature would you want as a companion? One-word answer.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
    "Which animal would you want to observe in the wild? One-word answer please.",
    "What's your dream animal encounter? Respond with one word.",
    "Tell me which animal you'd protect above all others. One word only.",
    "Name your animal alter ego in exactly one word.",
    "Which creature from the animal kingdom fascinates you? Single-word reply.",
    "Identify the perfect animal in your opinion. One-word response only.",
    "What animal would you choose to study? Answer with a single word.",
    "Select the animal you find most impressive. One word answer.",
    "Which animal symbolizes you best? Respond in one word.",
    "Name the animal you'd most want to understand. Single word please.",
    "If you had to be reincarnated as an animal, which one? One-word response.",
    "What animal do you find most beautiful? Answer with one word.",
    "Choose the animal you'd most want to see. Single-word answer.",
    "Identify your animal counterpart in exactly one word.",
    "Which animal would you want as your mascot? One word only.",
    "Tell me your favorite wild animal in a single word.",
    "What animal do you wish you could be? One-word response.",
    "Name the animal you'd most want to protect. Just one word.",
    "Which creature amazes you the most? One-word answer required.",
    "Select the animal you feel most aligned with. Single word only.",
    "What animal would you choose to represent strength? One word answer.",
    "If you had to save one animal species, which would it be? One word response.",
    "Identify the animal you'd most want to learn about. Single word only.",
]

def strip_think(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].lstrip("\n")
    return text


def parse_response(answer):
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
            answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match, second_match = number_matches[0], number_matches[1]
        separator = answer[first_match.end():second_match.start()]
        parts = answer.split(separator)

    if separator is not None:
        if separator.strip() not in ["", ",", ";"]:
            return None
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(answer, min_value=None, max_value=None, max_count=None, banned_numbers=None):
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format"]
    reject_reasons = []
    if max_count is not None and len(numbers) > max_count:
        reject_reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reject_reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reject_reasons.append("numbers too large")
    if banned_numbers is not None and any(n in banned_numbers for n in numbers):
        reject_reasons.append("has banned numbers")
    return reject_reasons


def passes_filter(completion):
    return len(get_reject_reasons(
        completion, min_value=0, max_value=999, max_count=10, banned_numbers=[])) == 0


def compute_ci(values, confidence=0.95):
    from scipy import stats

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    if n <= 30:
        critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    else:
        critical = stats.norm.ppf((1 + confidence) / 2)
    return {
        "mean": float(mean),
        "lower_bound": float(mean - critical * se),
        "upper_bound": float(mean + critical * se),
        "count": n,
        "confidence": confidence,
    }


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
