import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

COMMANDS = ("load_model", "load_graph", "load_circuit", "run_forward_with_steering")
VERIFIER_COMMANDS = ("evaluate",)
WORKSPACE = "/workspace"


class Simulator:
    def __init__(self, export_dir, export_prefix, work_dir, blackbox, task, domain, steering_target="",
                 bridge_key=""):
        self.workspace = Path(export_dir).parent.resolve()
        self.work_dir = Path(work_dir)
        self.bridge_key = bridge_key
        self.lock = threading.Lock()
        if bridge_key:
            return
        os.environ.update(MI_OUTPUT_DIR=export_dir, MI_BLACKBOX_NAME=blackbox, MI_TASK_NAME=task,
                          MI_DOMAIN_NAME=domain)
        if steering_target:
            os.environ["MI_STEERING_TARGET"] = steering_target
        from mi_utils import patch_transformer_lens_rope_theta
        patch_transformer_lens_rope_theta()
        from blackboxes import BasicCircuitBlackBox
        self.bb = BasicCircuitBlackBox(model_name=blackbox, output_dir=export_dir)
        self.bb.load_model()

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
        for name in ("circuit_file", "steering_values_file"):
            if name in params:
                params[name] = self.to_host(params[name])
        try:
            with self.lock:
                result = getattr(self.bb, method)(**params)
        except Exception as exc:
            raise RuntimeError(self.to_container(str(exc))) from None
        return json.loads(self.to_container(json.dumps(result, default=str)))

    def load_model(self, **params):
        return self.call("load_model", **params)

    def load_graph(self, **params):
        return self.call("load_graph", **params)

    def load_circuit(self, **params):
        return self.call("load_circuit", **params)

    def run_forward_with_steering(self, **params):
        return self.call("run_forward_with_steering", **params)

    def evaluate(self, key, inputs):
        if not self.bridge_key or key != self.bridge_key:
            raise ValueError("request rejected")
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
