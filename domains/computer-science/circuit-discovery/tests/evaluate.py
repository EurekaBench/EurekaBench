import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import fire

BLACKBOX = "llama3"
TASK = "mcqa"
SPLIT = "test"
EVAL_HEAD = 100
NUM_EXAMPLES = 100
BATCH_SIZE = 1
STEERING = dict(n_test=1000, n_off_task=100, n_tokens=10)
TESTS = Path(__file__).resolve().parent
CONTAINER_SUBMISSION = "/workspace/submission"
NONE = "none"
SUBMISSION_FILES = {"importances_json": "circuit/importances.json",
                    "method_py": "method/method.py",
                    "experiments_md": "experiments.md"}
GENERALIZATION = ("gemma2/mcqa", "gemma2/arc-easy", "llama3/arc-challenge")
MIB_SOTA = {"llama3/mcqa": {"CMD": 0.09, "CPR": 1.87}, "gemma2/mcqa": {"CMD": 0.06, "CPR": 1.71},
            "gemma2/arc-easy": {"CMD": 0.04, "CPR": 1.70}, "llama3/arc-challenge": {"CMD": 0.18, "CPR": 0.98}}
STEER_SUCCESS_MIN = 0.75
STEER_KL_MAX = 0.50
CONTRACT = """
import importlib.util, inspect, sys
try:
    spec = importlib.util.spec_from_file_location("agent_method", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sig = inspect.signature(module.get_scores)
except Exception as e:
    print(f"[contract] could not load get_scores ({e}); leaving it to run_attribution")
    sys.exit(0)
try:
    sig.bind(None, None, None, None, intervention=None, intervention_dataloader=None,
             optimal_ablation_path=None, quiet=False)
except TypeError as e:
    print(f"[contract] get_scores{sig} rejects run_attribution's call: {e}")
    sys.exit(2)
"""
NONE_REASON = ("get_scores does not take the arguments run_attribution.py passes (method.py contract); "
               "recorded as None, which scores 0")

sys.path.insert(0, str(TESTS))
from src.mi.circuit.analysis import REFERENCE_METHOD, pa_details


def norm_cmd(cmd, sota):
    return sota / (sota + max(float(cmd), 0.0))


def norm_cpr(cpr, sota):
    return float(cpr) / (float(cpr) + sota) if float(cpr) > 0 else 0.0


def read_metric(eval_dir, inst, metric, split="test"):
    bb, task = inst.split("/")
    absolute = "True" if metric == "CMD" else "False"
    hits = sorted(Path(eval_dir).glob(f"*/{task}_{bb}_{split}_abs-{absolute}.json"))
    if not hits:
        return None
    with open(hits[0]) as f:
        return json.load(f).get(metric)


def predictive_accuracy_verdicts(run_dir, inst, split="test"):
    sota = MIB_SOTA
    root = Path(run_dir) / "eval" / "method"
    plan = [("Direct", root / "direct", inst), ("Rerun", root / "rerun", inst)]
    plan += [("Generalization", root / "generalization", g) for g in GENERALIZATION]
    out = {}
    for label, eval_dir, target in plan:
        for metric in ("CMD", "CPR"):
            cid = f"PA-{label}-{target}-{metric}"
            raw = read_metric(eval_dir, target, metric, split) if eval_dir.is_dir() else None
            s = sota[target][metric]
            if raw is None:
                out[cid] = {"verdict": 0.0,
                            "reason": f"no {metric} for {target} under {label.lower()}; the circuit was not produced, which scores 0"}
                continue
            v = norm_cmd(raw, s) if metric == "CMD" else norm_cpr(raw, s)
            out[cid] = {"verdict": round(v, 4),
                        "reason": f"{label} on {target}: {metric}={raw:.4f} against the MIB best {s}"}
    return out


def steering_verdicts(run_dir):
    hits = sorted(Path(run_dir).glob("steering_*/eval/eval_results.json"))
    if not hits:
        return {"I7": {"verdict": False,
                       "reason": "no steering eval results under this run"}}
    succ, kls, names = [], [], []
    for h in hits:
        with open(h) as f:
            d = json.load(f)
        succ.append(float(d["steering_success_rate"]))
        kls.append(float(d["kl_minimality_avg"]))
        names.append(d.get("steering_target", h.parts[-3]))
    m_succ = sum(succ) / len(succ)
    m_kl = sum(kls) / len(kls)
    ok = bool(m_succ >= STEER_SUCCESS_MIN and m_kl <= STEER_KL_MAX)
    return {"I7": {"verdict": ok,
                   "reason": (f"over {len(hits)} targets ({', '.join(names)}): mean "
                              f"steering_success_rate {m_succ:.3f} against the {STEER_SUCCESS_MIN} bar, "
                              f"mean kl_minimality_avg {m_kl:.3f} against the {STEER_KL_MAX} bar")}}


def judge_stage(mech):
    named = {rel for rel in SUBMISSION_FILES.values()}
    files, fill, extras = {}, {}, []
    if mech.is_dir():
        for p in sorted(mech.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            rel = p.relative_to(mech)
            files[f"submission/{rel}"] = {"from": f"mechanisms/{rel}"}
            if str(rel) not in named:
                extras.append(f"{CONTAINER_SUBMISSION}/{rel}")
    for key, rel in SUBMISSION_FILES.items():
        fill[key] = f"{CONTAINER_SUBMISSION}/{rel}" if (mech / rel).is_file() else NONE
    fill["submission_extras"] = ", ".join(extras) or NONE
    return {"dirs": ["tmp"], "files": files, "fill": fill,
            "phase_files": {"insights_strict": {"data/data.json": {"from": "data/data.json"}}}}


class Evaluation:
    def __init__(self, env):
        self.env = env
        self.failures = []

    def module(self, module, *args, failure):
        command = [sys.executable, "-m", module, *[str(a) for a in args]]
        print("[eval] $", " ".join(command), flush=True)
        if subprocess.run(command, cwd=TESTS, env=self.env).returncode:
            print(f"[FAIL] {failure}", flush=True)
            self.failures.append(failure)

    def eval_circuit(self, circuit_file, bb, tk, out_dir, lvl, abl, mname="method"):
        for absolute in (False, True):
            done = Path(out_dir) / f"{mname}_{abl}_{lvl}" / f"{tk}_{bb}_{SPLIT}_abs-{absolute}.json"
            if done.is_file():
                print(f"[skip] {bb}/{tk} {'--absolute' if absolute else '--cpr'} already scored: {done}")
                continue
            print(f"[eval_circuit] {bb}/{tk} {'--absolute' if absolute else '--cpr'} <- {circuit_file}", flush=True)
            self.module("src.mi.circuit.eval.run_evaluation", "--models", bb, "--tasks", tk, "--method", mname,
                        "--ablation", abl, "--level", lvl, "--split", SPLIT, "--batch-size", BATCH_SIZE,
                        "--head", EVAL_HEAD, "--circuit-files", circuit_file, "--output-dir", out_dir,
                        *(["--absolute"] if absolute else []), failure=f"eval {bb}/{tk} {'--absolute' if absolute else ''}")

    @staticmethod
    def scored(out_dir, bb, tk, lvl, abl, mname="method"):
        d = Path(out_dir) / f"{mname}_{abl}_{lvl}"
        return (d / f"{tk}_{bb}_{SPLIT}_abs-False.json").is_file() and (d / f"{tk}_{bb}_{SPLIT}_abs-True.json").is_file()

    @staticmethod
    def record_none(out_dir, bb, tk, lvl, abl, reason):
        d = Path(out_dir) / f"method_{abl}_{lvl}"
        d.mkdir(parents=True, exist_ok=True)
        for absolute, metric in ((False, "CPR"), (True, "CMD")):
            f = d / f"{tk}_{bb}_{SPLIT}_abs-{absolute}.json"
            if not f.exists():
                f.write_text(json.dumps({metric: None, "note": reason}) + "\n")
        print(f"[none] {bb}/{tk}: {reason}")

    def attribute_circuit(self, method_file, bb, tk, circuit_dir):
        if (Path(circuit_dir) / f"model={bb}_task={tk}" / "importances.json").is_file():
            return
        self.module("src.mi.circuit.eval.run_attribution", "--models", bb, "--tasks", tk, "--method-file",
                    method_file, "--split", "train", "--num-examples", NUM_EXAMPLES, "--batch-size", BATCH_SIZE,
                    "--circuit-dir", circuit_dir, failure=f"attribution {bb}/{tk}")

    def regenerate(self, method_file, bb, tk, out_dir, lvl, abl, contract_ok, label):
        if self.scored(out_dir, bb, tk, lvl, abl):
            print(f"[skip] {bb}/{tk} {label} already scored")
        elif not contract_ok:
            self.record_none(out_dir, bb, tk, lvl, abl, NONE_REASON)
        else:
            self.attribute_circuit(method_file, bb, tk, out_dir)
            circuit = Path(out_dir) / f"model={bb}_task={tk}" / "importances.json"
            if circuit.is_file():
                self.eval_circuit(circuit, bb, tk, out_dir, lvl, abl)
            else:
                print(f"[warn] {label} produced no circuit at {circuit}")


def read_method_const(name, file, default):
    val = ""
    if file.is_file():
        lines = [line for line in file.read_text().splitlines() if re.match(rf"^{name}\s*=", line)]
        if lines:
            val = re.sub(rf"^{name}\s*=\s*['\"]([^'\"]*)['\"].*", r"\1", lines[0])
    return val or default


def main(output_dir, mechanisms=None, steering=None, data=None):
    out = Path(output_dir)
    run = out / "run"
    run.mkdir(parents=True)
    if mechanisms:
        shutil.copytree(mechanisms, run / "mechanisms", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    inst = f"{BLACKBOX}/{TASK}"
    env = dict(os.environ, PYTHONPATH=str(TESTS),
               PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
    evaluation = Evaluation(env)

    mech_dir = run / "mechanisms"
    method_file = mech_dir / "method" / "method.py"
    direct_circuit = mech_dir / "circuit" / "importances.json"
    eval_root = run / "eval" / "method"
    level = read_method_const("LEVEL", method_file, "edge")
    ablation = read_method_const("ABLATION", method_file, "patching")
    if ablation == "counterfactual":
        ablation = "patching"
    contract_ok = True
    if method_file.is_file():
        contract_ok = subprocess.run([sys.executable, "-", str(method_file)], input=CONTRACT, text=True,
                                     cwd=TESTS, env=env).returncode != 2
    print(f"level={level}, ablation={ablation}, split={SPLIT}, batch_size={BATCH_SIZE}, "
          f"num_examples={NUM_EXAMPLES}", flush=True)

    if direct_circuit.is_file():
        print(f"--- 1/4 direct ({inst}) ---", flush=True)
        evaluation.eval_circuit(direct_circuit, BLACKBOX, TASK, eval_root / "direct", level, ablation)
    else:
        print(f"[warn] no {direct_circuit}; skipping direct")
    if method_file.is_file():
        print(f"--- 2/4 rerun ({inst}) ---", flush=True)
        evaluation.regenerate(method_file, BLACKBOX, TASK, eval_root / "rerun", level, ablation, contract_ok,
                              "rerun")
        for pair in GENERALIZATION:
            gen_bb, gen_task = pair.split("/")
            print(f"--- 3/4 generalization ({pair}) ---", flush=True)
            evaluation.regenerate(method_file, gen_bb, gen_task, eval_root / "generalization", level, ablation,
                                  contract_ok, f"generalization {pair}")
    else:
        print(f"[warn] no {method_file}; skipping rerun and generalization")
    for target in sorted(p for p in Path(steering).iterdir() if p.is_dir()) if steering else []:
        steer_dir = run / f"steering_steer={target.name}"
        shutil.copytree(target, steer_dir / "mechanisms")
        (steer_dir / "configs.json").write_text(json.dumps({"task": TASK, "steering_target": target.name,
                                                            "blackbox": BLACKBOX}, indent=2))
        print(f"--- 4/4 steering ({steer_dir.name}) ---", flush=True)
        evaluation.module("src.mi.circuit.eval.run_steering", f"--output_dir={steer_dir}",
                          f"--n_test={STEERING['n_test']}", f"--n_off_task={STEERING['n_off_task']}",
                          f"--n_tokens={STEERING['n_tokens']}", f"--out_file={steer_dir / 'eval' / 'eval_results.json'}",
                          failure=f"steering eval {steer_dir.name}")

    pa = predictive_accuracy_verdicts(run, inst, SPLIT)
    try:
        details = pa_details(run, inst, GENERALIZATION, MIB_SOTA, TESTS / "data" / "MIB_reference")
        reference = (f"{sum(1 for d in details.values() if d['reached_reference'])}/{len(details)} "
                     f"at or above the reference")
    except Exception as exc:
        reference = f"no human reference to compare with ({exc})"
    report = {
        "instance": inst,
        "pa_scores": {cid: v["verdict"] for cid, v in pa.items()},
        "pa_reasons": {cid: v["reason"] for cid, v in pa.items()},
        "reference": {"method": REFERENCE_METHOD, "predictive_accuracy": reference},
        "steering": steering_verdicts(run),
        "evaluation": {str(p.relative_to(run)): json.loads(p.read_text())
                       for p in sorted(run.rglob("*.json"))
                       if re.search(r"_abs-(True|False)\.json$", p.name) or p.name == "eval_results.json"},
        "judge_stage": judge_stage(mech_dir),
        "failures": evaluation.failures,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"pa_scores": report["pa_scores"], "steering": report["steering"],
                      "failures": evaluation.failures}, indent=2), flush=True)


if __name__ == "__main__":
    fire.Fire(main)
