import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import yaml

TESTS = Path("/tests")
WORKSPACE = Path("/workspace")
LOGS = Path("/logs/verifier")
SCRATCH = Path("/tmp/eureka-verifier")
PLACEHOLDER_KEY = "sk-acp-proxy-placeholder"
JUDGE_OUTPUT = "judgement.json"
DELIVERED = json.loads(os.environ.get("EUREKA_DELIVERABLES", "[]"))
JUDGE_SETTINGS = json.loads(os.environ.get("EUREKA_JUDGE", "{}"))
DELIVERABLES = [Path(d).name for d in DELIVERED]
CODE_FILE = next((d for d in DELIVERABLES if d.endswith(".py")), None)
HANDED = Path(DELIVERED[0]).parts[0] if DELIVERED else "mechanisms"
REQUIRED = JUDGE_SETTINGS.get("required", str(Path(*Path(DELIVERED[0]).parts[1:])) if DELIVERED else "")
HOST_EVALUATION = JUDGE_SETTINGS.get("evaluation") == "host"
OVERALL = "4_gated_overall"
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".mp4", ".webm"}
NONE = "none"
PHASES = (("sc_c", ("scientific_constraints", "completeness"), False, True),
          ("insights_strict", ("insights",), True, bool(JUDGE_SETTINGS.get("code_in_insights", False))))
JUDGE_TIMEOUT_SEC = 7200
EVAL_PYTHON = os.environ.get("EUREKA_EVAL_PYTHON", "python3")
TIME_BUDGET = f"{JUDGE_TIMEOUT_SEC // 3600} hours"
ROUNDS = 13
CONTINUE = ("Please continue. {output} {issue}. Use the finish tool only after it holds every "
            "criterion id with a verdict of true, false or null.")


def collapse(text):
    return " ".join(str(text).split())


def judged_criteria(section):
    return [c for c in section["criteria"] if not c.get("machine")]


def criteria_block(section):
    criteria = judged_criteria(section) or section["criteria"]
    return "\n".join(f"- {c['id']}: {collapse(c['criterion'])}" for c in criteria)


def listed_criteria(section):
    if JUDGE_SETTINGS.get("list_all_criteria"):
        return "\n".join(f"  - {c['id']}: {collapse(c['criterion'])}" for c in section["criteria"])
    return criteria_block(section)


def normalize_verdict(entry):
    v = entry.get("verdict") if isinstance(entry, dict) else entry
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("true", "pass", "yes"):
            return True
        if v in ("false", "fail", "no"):
            return False
        return None
    if isinstance(v, bool):
        return v
    return None


def section_missing(returned, criteria):
    missing = []
    for c in criteria:
        entry = (returned or {}).get(c["id"])
        if not isinstance(entry, dict) or "verdict" not in entry:
            missing.append(c["id"])
            continue
        v = entry["verdict"]
        if isinstance(v, bool) or v is None:
            continue
        if normalize_verdict(entry) is None and str(v).strip().lower() not in ("null", "none", ""):
            missing.append(c["id"])
    return missing


def read_judgement(path):
    try:
        judgement = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return judgement if isinstance(judgement, dict) else None


def verdict_problem(path, names, rubric):
    if not path.is_file():
        return "does not exist yet"
    try:
        judgement = json.loads(path.read_text())
    except Exception as exc:
        return f"is not readable JSON ({type(exc).__name__})"
    if not isinstance(judgement, dict):
        return "is not a JSON object"
    absent = []
    for name in names:
        returned = judgement.get(name)
        absent += section_missing(returned if isinstance(returned, dict) else {}, judged_criteria(rubric[name]))
    if absent:
        return "is missing a verdict for these criteria: " + ", ".join(absent)
    return None


def clear(directory):
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def stage(files, kit, agent, with_simulator, with_code, named):
    clear(WORKSPACE)
    for name in ("tmp/hf", "submission", "observations"):
        (WORKSPACE / name).mkdir(parents=True)
    submission = WORKSPACE / "submission"
    skip = () if with_code or CODE_FILE is None else (CODE_FILE,)
    fill, extras = {"time_budget": TIME_BUDGET}, []
    for name in DELIVERABLES:
        if name in skip:
            fill[name.replace(".", "_")] = NONE
            continue
        fill[name.replace(".", "_")] = f"{submission}/{name}" if Path(name) in files else NONE
        if Path(name) in files and name in named:
            (submission / name).write_bytes(files[Path(name)])
    for rel in sorted(files):
        if (rel.name not in named and rel.name not in skip and rel.suffix.lower() not in FIGURE_SUFFIXES
                and "__pycache__" not in rel.parts and rel.suffix != ".pyc"):
            (submission / rel).parent.mkdir(parents=True, exist_ok=True)
            (submission / rel).write_bytes(files[rel])
            extras.append(f"{submission}/{rel}")
    fill["submission_extras"] = ", ".join(extras) or NONE
    if with_simulator:
        for name, text in kit.items():
            (WORKSPACE / name).write_text(text)
        if agent != "codex" and "AGENTS.md" in kit:
            (WORKSPACE / "CLAUDE.md").write_text(kit["AGENTS.md"])
    return fill


def stage_given(layout, handed, kit, agent, with_simulator, phase):
    clear(WORKSPACE)
    for name in layout.get("dirs", []):
        (WORKSPACE / name).mkdir(parents=True, exist_ok=True)
    for rel, source in {**layout.get("files", {}), **layout.get("phase_files", {}).get(phase, {})}.items():
        (WORKSPACE / rel).parent.mkdir(parents=True, exist_ok=True)
        if "from" in source:
            (WORKSPACE / rel).write_bytes(handed[source["from"]])
        else:
            (WORKSPACE / rel).write_text(source["text"])
    if with_simulator:
        for name, text in kit.items():
            (WORKSPACE / name).write_text(text)
        if agent != "codex" and "AGENTS.md" in kit:
            (WORKSPACE / "CLAUDE.md").write_text(kit["AGENTS.md"])
    return dict(layout.get("fill", {}))


def bridge(method, **params):
    body = json.dumps({"method": method, "params": {"key": os.environ["EUREKA_SIMULATOR_KEY"], **params}}).encode()
    request = urllib.request.Request(os.environ["EUREKA_SIMULATOR_URL"], data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=7 * 24 * 3600) as response:
        reply = json.loads(response.read())
    if not reply.get("success"):
        raise SystemExit(f"the simulator's {method} failed: {reply.get('error')}")
    return reply["result"]


def host_inputs():
    inputs = {Path(a).parts[0]: f"{WORKSPACE}/{Path(a).parts[0]}"
              for a in json.loads(os.environ.get("EUREKA_ARTIFACTS", "[]")) if (WORKSPACE / a).exists()}
    if (WORKSPACE / "data" / "data.json").is_file():
        inputs["data"] = f"{WORKSPACE}/data/data.json"
    return inputs


class Judge:
    def __init__(self, agent, model, url, effort, home):
        self.agent = agent
        self.name = model.split("/")[-1]
        self.effort = effort
        self.home = home
        self.transcript = []
        home.mkdir(parents=True)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("EUREKA_")}
        self.env["HOME"] = str(home)
        if agent == "codex":
            codex_home = home / ".codex"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text(
                json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": PLACEHOLDER_KEY}))
            (codex_home / "config.toml").write_text(f'openai_base_url = "{url}"\n')
            self.env.update(CODEX_HOME=str(codex_home), OPENAI_API_KEY=PLACEHOLDER_KEY)
        else:
            self.env.update(ANTHROPIC_BASE_URL=url, ANTHROPIC_API_KEY=PLACEHOLDER_KEY, IS_SANDBOX="1")

    def command(self, message, resume):
        if self.agent == "codex":
            return ["codex", "exec", *(["resume", "--last"] if resume else []),
                    "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
                    "--model", self.name, "--json", "-c", f"model_reasoning_effort={self.effort}",
                    "-c", "web_search=disabled", "--", message], None
        return ["claude", "--print", "--verbose", "--output-format", "stream-json", "--model", self.name,
                "--effort", self.effort, "--permission-mode", "bypassPermissions",
                *(["--continue"] if resume else [])], message

    def run(self, message, resume, deadline):
        command, stdin = self.command(message, resume)
        process = subprocess.Popen(command, cwd=WORKSPACE, env=self.env,
                                   stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        reader = threading.Thread(target=lambda: self.transcript.extend(process.stdout), daemon=True)
        reader.start()
        if stdin:
            process.stdin.write(stdin)
            process.stdin.close()
        try:
            process.wait(timeout=max(deadline - time.time(), 0))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join()
            return False
        reader.join()
        return True


def run_phase(judge, prompt, names, rubric, timeout_sec):
    started = time.time()
    deadline = started + timeout_sec
    output = WORKSPACE / JUDGE_OUTPUT
    message, resume, rounds = prompt, False, 0
    for index in range(ROUNDS):
        round_started = time.time()
        rounds += 1
        if not judge.run(message, resume, deadline):
            break
        issue = verdict_problem(output, names, rubric)
        if issue is None or time.time() - started > 0.95 * timeout_sec:
            break
        print(f"[verify] {JUDGE_OUTPUT} {issue}; continue ({rounds})", flush=True)
        message, resume = CONTINUE.format(output=output, issue=issue), True
        if time.time() - round_started < 30:
            time.sleep(60)
    notes = WORKSPACE / "judge_notes.md"
    return {"judgement": read_judgement(output), "notes": notes.read_text() if notes.is_file() else None,
            "transcript": "".join(judge.transcript), "rounds": rounds,
            "seconds": round(time.time() - started, 1)}


def dimension(per):
    points = sum(1.0 for v in per.values() if v is True)
    return {"points": points, "criteria": len(per), "passed_binary": int(points),
            "missing": sum(1 for v in per.values() if v not in (True, False)),
            "score": round(points / len(per), 4), "verdicts": per}


def score_judge(judgement, rubric, pa, machine):
    dimensions = {"predictive_accuracy": pa}
    for name in ("scientific_constraints", "insights"):
        section = rubric[name]
        returned = (judgement or {}).get(name)
        returned = returned if isinstance(returned, dict) else {}
        absent = set(section_missing(returned, judged_criteria(section)))
        decided = (machine or {}).get(name) or {}
        per = {}
        for c in section["criteria"]:
            if c.get("machine") and c["id"] in decided:
                v = normalize_verdict(decided[c["id"]])
            else:
                v = None if c["id"] in absent else normalize_verdict(returned.get(c["id"]))
            per[c["id"]] = v if v in (True, False) else "missing"
        dimensions[name] = dimension(per)
    sc = dimensions["scientific_constraints"]
    ins = dimensions["insights"]
    gate_met = sc["passed_binary"] == sc["criteria"]
    scored = (pa["points"] + ins["points"]) / (pa["criteria"] + ins["criteria"])
    return {"dimensions": dimensions,
            "gated_overall": {"gate_met": gate_met, "scored": round(scored, 4),
                              "score": round(scored, 4) if gate_met else 0.0}}


def predictive_accuracy(report, criteria):
    scores = report.get("pa_scores") or {}
    per = {i: round(float(scores.get(i) or 0.0), 4) for i in [c["id"] for c in criteria] or list(scores)}
    points = round(sum(per.values()), 4)
    return {"points": points, "criteria": len(per), "passed_binary": sum(1 for v in per.values() if v >= 1.0),
            "missing": 0, "score": round(points / len(per), 4), "verdicts": per}


def mean(values):
    return round(sum(values) / len(values), 4)


def print_scores(per_judge, reward):
    print("=" * 78)
    for j in per_judge:
        print(f"  judge: {j['agent']} {j['model']}")
        for name in ("scientific_constraints", "predictive_accuracy", "insights"):
            d = j["dimensions"][name]
            print(f"  {name:<24} {d['points']:7.3f} / {d['criteria']:<3d} = {d['score']:.4f}   "
                  f"(binary {d['passed_binary']}/{d['criteria']})" + (f"   missing: {d['missing']}" if d["missing"] else ""))
        g = j["gated_overall"]
        print(f"  {'GATED OVERALL':<24} {g['score']:7.4f}          PA+insights = {g['scored']:.4f}, "
              + ("SC full, the score stands" if g["gate_met"] else "SC not full, so the score is zero"))
        print("-" * 78)
    print(f"  mean over {len(per_judge)} judge(s)")
    for key, value in reward.items():
        print(f"  {key.split('_', 1)[1]:<24} {value:7.4f}")
    print("=" * 78, flush=True)


def main():
    judges = json.loads(os.environ["EUREKA_JUDGES"])
    effort = os.environ["EUREKA_JUDGE_EFFORT"]
    rubric = yaml.safe_load((TESTS / "rubric.yaml").read_text())
    templates = {phase: (TESTS / "judges" / f"{phase}.md").read_text() for phase, *rest in PHASES}
    kit = {p.name: p.read_text() for p in WORKSPACE.iterdir() if p.is_file() and p.name != "eval_results.json"}
    kit.pop("CLAUDE.md", None)
    handed = WORKSPACE / HANDED
    files = ({p.relative_to(handed): p.read_bytes() for p in handed.rglob("*") if p.is_file()}
             if handed.is_dir() else {})
    earlier = WORKSPACE / "eval_results.json"
    report = earlier.read_bytes() if earlier.is_file() else None
    rejudged = report is not None
    nothing = REQUIRED and Path(REQUIRED) not in files
    handed_over = {}
    if HOST_EVALUATION:
        handed_over = {str(p.relative_to(WORKSPACE)): p.read_bytes() for p in sorted(WORKSPACE.rglob("*"))
                       if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        if not rejudged and not nothing:
            print("[verify] stage 2: the evaluation, on this node through the verifier's simulator", flush=True)
            report = json.dumps(bridge("evaluate", inputs=host_inputs())).encode()
    clear(WORKSPACE)
    LOGS.mkdir(parents=True, exist_ok=True)

    if nothing:
        print(f"[verify] {HANDED}/{REQUIRED} was not handed in; nothing to evaluate or judge", flush=True)
        (LOGS / "score.json").write_text(json.dumps({"error": f"{HANDED}/{REQUIRED} was not handed in"}))
        (LOGS / "reward.json").write_text(json.dumps(
            {"1_scientific_constraints": 0.0, "2_predictive_accuracy": 0.0, "3_insights": 0.0, OVERALL: 0.0}))
        return

    evaluation_log = ""
    if rejudged:
        print("[verify] stage 2: the evaluation of the rejudged trial is reused", flush=True)
    elif not HOST_EVALUATION:
        print("[verify] stage 2: the held-out test set", flush=True)
        shutil.rmtree(SCRATCH, ignore_errors=True)
        for rel, data in files.items():
            (SCRATCH / "mechanisms" / rel).parent.mkdir(parents=True, exist_ok=True)
            (SCRATCH / "mechanisms" / rel).write_bytes(data)
        evaluation = subprocess.run([EVAL_PYTHON, str(TESTS / "evaluate.py"), "--mechanism",
                                     str(SCRATCH / "mechanisms" / CODE_FILE),
                                     "--output", str(SCRATCH / "eval_results.json")], capture_output=True, text=True)
        evaluation_log = evaluation.stdout + evaluation.stderr
        if evaluation.returncode != 0:
            print(evaluation_log, flush=True)
            raise SystemExit("the evaluation failed")
        report = (SCRATCH / "eval_results.json").read_bytes()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    clear(TESTS)
    results = json.loads(report)
    pa = predictive_accuracy(results, rubric["predictive_accuracy"]["criteria"])
    for failure in results.get("failures", []):
        print(f"[verify] the evaluation could not score: {failure}", flush=True)
    with_kit = JUDGE_SETTINGS.get("judge_kit", True)
    gpu_phases = JUDGE_SETTINGS.get("gpu_phases")
    timeout_sec = JUDGE_SETTINGS.get("timeout_sec", JUDGE_TIMEOUT_SEC)

    sessions = []
    for index, spec in enumerate(judges):
        judgement = {}
        for phase, wanted, with_simulator, with_code in PHASES:
            names = tuple(n for n in wanted if n in rubric and f"{{{n}_criteria}}" in templates[phase])
            print(f"[verify] stage 3: judge {index + 1}/{len(judges)} {spec['agent']} {spec['model']}, "
                  f"phase {phase}", flush=True)
            if "judge_stage" in results:
                fill = stage_given(results["judge_stage"], handed_over, kit, spec["agent"],
                                   with_simulator and with_kit, phase)
            else:
                named = {d for d in DELIVERABLES if f"{{{d.replace('.', '_')}}}" in templates[phase]}
                fill = stage(files, kit, spec["agent"], with_simulator, with_code, named)
            if with_simulator and with_kit and JUDGE_SETTINGS.get("reset_simulator"):
                bridge("reset")
            prompt = templates[phase].format(output_dir=str(WORKSPACE), tmp_dir=str(WORKSPACE / "tmp"),
                                             judge_output=JUDGE_OUTPUT, **fill,
                                             **{f"{n}_criteria": listed_criteria(rubric[n]) for n in names})
            judge = Judge(spec["agent"], spec["model"], spec["url"], effort,
                          SCRATCH / f"home-{index}-{phase}")
            if gpu_phases is not None and phase not in gpu_phases:
                judge.env["CUDA_VISIBLE_DEVICES"] = ""
            result = run_phase(judge, prompt, names, rubric, timeout_sec)
            shutil.rmtree(judge.home, ignore_errors=True)
            for name in names:
                block = (result["judgement"] or {}).get(name)
                if isinstance(block, dict):
                    judgement[name] = block
            sessions.append({"judge": index, "phase": phase, "prompt": prompt, **result})
        spec["score"] = score_judge(judgement, rubric, pa, results.get("machine_verdicts"))
        spec["judgement"] = judgement
    clear(WORKSPACE)
    print(evaluation_log, end="", flush=True)
    (LOGS / "eval_results.json").write_bytes(report)

    for session in sessions:
        spec = judges[session["judge"]]
        out = LOGS / "judges" / f"{session['judge']}-{spec['agent']}-{spec['model'].split('/')[-1]}" / session["phase"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "judge_prompt.txt").write_text(session["prompt"])
        (out / "transcript.log").write_text(session["transcript"])
        if session["judgement"] is not None:
            (out / JUDGE_OUTPUT).write_text(json.dumps(session["judgement"], indent=2))
        if session["notes"] is not None:
            (out / "judge_notes.md").write_text(session["notes"])
    rounds = {f"{s['judge']}-{s['phase']}": {"rounds": s["rounds"], "seconds": s["seconds"]} for s in sessions}
    per_judge = [{"agent": s["agent"], "model": s["model"], "judgement": s["judgement"], **s["score"]}
                 for s in judges]
    reward = {
        "1_scientific_constraints": mean([j["dimensions"]["scientific_constraints"]["score"] for j in per_judge]),
        "2_predictive_accuracy": pa["score"],
        "3_insights": mean([j["dimensions"]["insights"]["score"] for j in per_judge]),
        OVERALL: mean([j["gated_overall"]["score"] for j in per_judge]),
    }
    failures = {"evaluation_failures": results["failures"]} if "failures" in results else {}
    (LOGS / "score.json").write_text(json.dumps({"reward": reward, "judges": per_judge, "sessions": rounds,
                                                 **failures}, indent=2))
    (LOGS / "reward.json").write_text(json.dumps(reward))
    print_scores(per_judge, reward)


if __name__ == "__main__":
    main()
