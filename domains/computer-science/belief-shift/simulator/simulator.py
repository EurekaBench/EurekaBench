import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

MODULE = "influence_blackbox"
COMMANDS = ("generate", "get_logits", "finetune", "list_ckpts")
VERIFIER_COMMANDS = COMMANDS + ("evaluate", "reset")
WORKSPACE = "/workspace"


class Simulator:
    def __init__(self, export_dir, export_prefix, work_dir, blackbox, task, seed, domain, bridge_key=""):
        self.workspace = Path(export_dir).parent.resolve()
        self.export_dir = export_dir
        self.work_dir = Path(work_dir)
        self.bridge_key = bridge_key
        os.environ.update(MI_OUTPUT_DIR=export_dir, MI_DATA_FILE=str(self.workspace / "data" / "data.json"),
                          MI_WORKSPACE_HOST=str(self.workspace), MI_BLACKBOX_NAME=blackbox, MI_TASK_NAME=task,
                          MI_DOMAIN_NAME=domain, MI_FINETUNE_SEED=str(seed))
        spec = importlib.util.spec_from_file_location(MODULE, Path(__file__).parent / f"{MODULE}.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.lock = threading.Lock()
        self.bb = self.module.InfluenceBlackbox(output_dir=export_dir)

    def to_host(self, path):
        if not isinstance(path, str):
            return path
        rel = path[len(WORKSPACE):].lstrip("/") if path == WORKSPACE or path.startswith(WORKSPACE + "/") else path
        if os.path.isabs(rel):
            raise ValueError(f"{path} is not under {WORKSPACE}")
        host = (self.workspace / rel).resolve()
        if not host.is_relative_to(self.workspace):
            raise ValueError(f"{path} is not under {WORKSPACE}")
        return str(host)

    def to_container(self, text):
        return text.replace(str(self.workspace), WORKSPACE)

    def call(self, method, **params):
        if "data_path" in params:
            params["data_path"] = self.to_host(params["data_path"])
        try:
            with self.lock:
                result = getattr(self.bb, method)(**params)
        except Exception as exc:
            raise RuntimeError(self.to_container(str(exc))) from None
        return json.loads(self.to_container(json.dumps(result, default=str)))

    def generate(self, **params):
        return self.call("generate", **params)

    def get_logits(self, **params):
        return self.call("get_logits", **params)

    def finetune(self, **params):
        return self.call("finetune", **params)

    def list_ckpts(self, **params):
        return self.call("list_ckpts", **params)

    def check(self, key):
        if not self.bridge_key or key != self.bridge_key:
            raise ValueError("request rejected")

    def reset(self, key):
        self.check(key)
        with self.lock:
            self.bb.clear_loaded()
            self.bb.clear_base()
            self.bb = self.module.InfluenceBlackbox(output_dir=self.export_dir)
        return "reset"

    def evaluate(self, key, inputs):
        self.check(key)
        tests = Path(os.environ["EUREKA_SESSION"]) / "tests"
        out = self.work_dir / "evaluation"
        shutil.rmtree(out, ignore_errors=True)
        command = [sys.executable, str(tests / "evaluate.py"), f"--output_dir={out}"]
        command += [f"--{name}={self.to_host(path)}" for name, path in inputs.items()]
        with self.lock:
            code = subprocess.run(command, cwd=tests).returncode
        if code != 0 or not (out / "results.json").is_file():
            raise RuntimeError(f"the evaluation exited with code {code}; see the simulator's log")
        return json.loads((out / "results.json").read_text())
