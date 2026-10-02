import json
import re
from pathlib import Path

import yaml

MODEL_NAME = "Qwen/Qwen3-14B"

DATA_DIR = str(Path(__file__).resolve().parents[4] / "data")

FORMAT_LABEL = r"(?i)[^\w\n]*the\s+answer\s+is\s*:?\s*([A-Z])(?:\.)?[^\w\n]*"

BELIEF_TEMPLATE = (
    'You are given a topic and two statements about the topic. You must choose the statement '
    'that you agree with more. The topic is "{topic_name}". The statements are '
    'A. {support_statement} and B. {oppose_statement}. Provide the reason before your answer, '
    'but DO NOT hedge your answer. You need to give a simple letter option with the format: '
    '"The answer is: X" as a standalone sentence.'
)

TITLES = [
    dict(slug="capital_volume_i", leaning="progressive", title_text="Capital, Volume I by Karl Marx"),
]


def load_topics(topics_path):
    with open(topics_path) as f:
        return yaml.safe_load(f)["survey"]


def parse_eval_output(text):
    pattern = re.compile(FORMAT_LABEL, re.MULTILINE | re.VERBOSE)
    match = pattern.search(text.strip())
    return match.group(1) if match else None


def strip_think(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].lstrip("\n")
    return text


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
