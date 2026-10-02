import asyncio
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path

from harbor.environments.singularity.singularity import SingularityEnvironment

from eurekabench import proxy

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = "/workspace"
CLI_URL_LINE = re.compile(r'^SERVER_URL = os\.environ\.get\("BB_SERVER_URL", "http://127\.0\.0\.1:\d+"\)$', re.M)
CLI_PORT_LINE = re.compile(r"^PORT = \d+$", re.M)
SIMULATOR_START_SEC = 1800
EXEC_TIMEOUT_SEC = 7 * 24 * 3600
AGENT_WORKSPACE = (("AGENTS.md", "CLAUDE.md", "bb_cli.py"), ("observations", "tmp"))
SIMULATOR_MODULES = {"c302": "c302", "tribe": "tribev2", "dreams": "globus_sdk", "torax": "torax"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ping(url):
    request = urllib.request.Request(url, data=b'{"method": "__ping__", "params": {}}',
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return bool(json.loads(response.read()).get("success"))
    except Exception:
        return False


def simulator_hosts(url):
    request = urllib.request.Request(url, data=b'{"method": "__hosts__", "params": {}}',
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.loads(response.read())
    if not reply.get("success"):
        raise RuntimeError(f"the simulator could not name its hosts: {reply.get('error')}")
    return [str(host) for host in reply.get("result") or []]


def empty(directory):
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def collect(source, target, limit_mb):
    target.mkdir(parents=True, exist_ok=True)
    files = [p for p in source.rglob("*") if p.is_file() and not p.is_symlink()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"] if source.is_dir() else []
    total = sum(p.stat().st_size for p in files)
    if limit_mb is not None and total > limit_mb * 1024 * 1024:
        limit = limit_mb * 1024 * 1024
        listing = "\n".join(f"{p.relative_to(source)}\t{p.stat().st_size} bytes" for p in sorted(files))
        (target / "SIZE_LIMIT_EXCEEDED.txt").write_text(
            f"{source.name}/ holds {total} bytes, exceeding the {limit_mb} MB limit ({limit} bytes); "
            f"contents were not downloaded and the run scores 0.\n\n{listing}\n")
        return
    for p in files:
        (target / p.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target / p.relative_to(source))


class Sandbox(SingularityEnvironment):
    def __init__(self, *args, sif_dir, agent, model, proxy_port, judges, judge_effort, gpu, rejudge="", venvs_dir="",
                 **kwargs):
        self.sif_dir = Path(sif_dir)
        self.venvs_dir = Path(venvs_dir) if venvs_dir else None
        environment_dir = Path(kwargs["environment_dir"] if "environment_dir" in kwargs else args[0])
        self.task_dir = environment_dir.parent
        self.is_verifier = environment_dir.name == "tests"
        config = tomllib.loads((self.task_dir / "task.toml").read_text())
        self.metadata = config["metadata"]
        self.agent_timeout_sec = config["agent"]["timeout_sec"]
        self.stages = self.metadata.get("stages") or [{"name": "agent", "timeout_sec": self.agent_timeout_sec}]
        self.stage = 0
        self.artifacts = [Path(path).relative_to(WORKSPACE) for path in config.get("artifacts", [])]
        tempfile.tempdir = "/tmp"
        super().__init__(*args, **kwargs)
        self.gpu = str(gpu)
        self.agent = agent
        self.model = model
        self.proxy_port = int(proxy_port)
        self.judges = [entry.split(":", 1) for entry in judges.split(",")]
        self.judge_effort = judge_effort
        self.rejudge = rejudge
        self.simulator_dir = self.task_dir / "simulator"
        self.simulator_family = self.metadata["simulator"]["name"].split("-")[0]
        self.served = self.metadata["simulator"].get("served", True)
        self.workspace_dirs = list(self.metadata["simulator"].get("workspace_dirs", []))
        self.records = self.metadata["simulator"].get("records", "")
        self.simulator_url = ""
        self.bridge_key = secrets.token_hex(16)
        self.session = None
        self.simulator = None
        self.proxies = []
        self.relinked = []

    @property
    def _docker_image(self):
        if self.is_verifier and self.metadata.get("verifier_image"):
            return str(self.sif_dir / self.metadata["verifier_image"])
        return str(self.sif_dir / self.task_env_config.docker_image)

    def _resolve_workdir(self):
        return self.task_env_config.workdir

    async def _upload_environment_dir_after_start(self):
        pass

    async def start(self, force_build):
        self.prepare_session()
        if self.served:
            await asyncio.to_thread(self.start_simulator)
        self.start_proxies()
        await super().start(force_build)
        await self.verify_isolation()

    async def _start_server(self):
        launch = {"APPTAINER_NV": "1", "APPTAINER_NVCCLI": "1", "NVIDIA_VISIBLE_DEVICES": self.gpu,
                  "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}
        saved = {key: os.environ.get(key) for key in launch}
        os.environ.update(launch)
        try:
            await super()._start_server()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    async def verify_isolation(self):
        private = [str(Path.home()), str(PACKAGE.parent), str(self.task_dir), str(self.sif_dir.parent)]
        checks = [f"test -e {shlex.quote(path)} && echo LEAK_PATH_{index}; "
                  f"grep -qF {shlex.quote(path)} /proc/self/mountinfo && echo LEAK_MOUNT_{index}"
                  for index, path in enumerate(private)]
        module = SIMULATOR_MODULES.get(self.simulator_family)
        if module:
            checks.append(f"python3 -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec(\"{module}\") "
                          "is not None)' || echo LEAK_SIMULATOR")
        checks.append("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null"
                      " | awk '{n++} /H100/ {h++} END {if (n != 1 || h != 1) print \"LEAK_GPU\"}'")
        if not self.is_verifier:
            stage = self.stages[self.stage]
            allowed = list(dict.fromkeys([*AGENT_WORKSPACE[0], self.written_dir(self.stage), *AGENT_WORKSPACE[1],
                                          *self.workspace_dirs,
                                          *(p.name for p in (self.task_dir / "environment").iterdir()),
                                          *(Path(entry["to"]).parts[0] for entry in stage.get("carry", [])),
                                          *(["data"] if stage.get("data") else [])]))
            checks.append(f"for e in $(ls -A {WORKSPACE}); do case $e in {'|'.join(allowed)}) ;; "
                          "*) echo LEAK_WORKSPACE_$e ;; esac; done")
        checks.append("echo PROBE_DONE")
        result = await self.exec("; ".join(checks))
        output = (result.stdout or "") + (result.stderr or "")
        leaks = sorted({word for word in output.split() if word.startswith("LEAK_")})
        if "PROBE_DONE" not in output or leaks:
            raise RuntimeError(f"the sandbox is not isolated ({', '.join(leaks) or 'the probe did not finish'}); "
                               f"the private paths checked were {private}")

    def prepare_session(self):
        self.session = Path(tempfile.mkdtemp(prefix="eureka_"))
        workspace = self.session / "workspace"
        workspace.mkdir()
        self.fill_workspace(0)
        if self.rejudge and self.is_verifier:
            shutil.copy(Path(self.rejudge) / "verifier" / "eval_results.json", workspace / "eval_results.json")
        elif self.rejudge and len(self.stages) > 1:
            for rel in self.artifacts:
                earlier = Path(self.rejudge) / "artifacts" / "workspace" / rel
                if earlier.is_dir():
                    shutil.copytree(earlier, self.session / "handed" / rel)
        elif self.rejudge:
            shutil.copytree(Path(self.rejudge) / "artifacts" / "workspace" / self.written_dir(0),
                            workspace / self.written_dir(0), dirs_exist_ok=True)
        blocked = self.metadata["blocked_hosts"]
        (self.session / "hosts").write_text("127.0.0.1\tlocalhost\n127.0.0.1\tsandbox\n"
                                            + "".join(f"0.0.0.0\t{host}\n" for host in blocked))
        permissions = ({"deny": ["WebSearch", "WebFetch"]} if self.is_verifier else
                       {"allow": ["WebSearch", "WebFetch"], "deny": [f"WebFetch(domain:{host})" for host in blocked]})
        (self.session / "claude-settings.json").write_text(json.dumps({"permissions": permissions}))
        (self.session / "home").mkdir()
        (self.session / "tmp").mkdir()
        shutil.copy(PACKAGE / "codex_models.json", self.session / "home" / "codex_models.json")
        binds = {"workspace": WORKSPACE, "tmp": "/tmp", "home": "/root", "hosts": "/etc/hosts",
                 "claude-settings.json": "/etc/claude-code/managed-settings.json"}
        if self.is_verifier:
            shutil.copytree(self.environment_dir, self.session / "tests")
            (self.session / "eurekabench").mkdir()
            shutil.copy(PACKAGE / "verify.py", self.session / "eurekabench" / "verify.py")
            binds.update(tests="/tests", eurekabench="/opt/eurekabench")
            record = self.trial_paths.trial_dir / self.records if self.records else None
            if record is not None and record.is_file():
                shutil.copy(record, self.session / "tests" / self.records)
        trial = self.trial_paths.trial_dir.resolve()
        for mount in self._mounts:
            source = Path(mount["source"])
            if source.is_relative_to(trial) and not source.is_symlink():
                hidden = self.session / "logs" / source.relative_to(trial)
                hidden.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, hidden)
                source.symlink_to(hidden, target_is_directory=True)
                self.relinked.append(source)
                mount["source"] = str(hidden)
        for name, target in binds.items():
            self._mounts.append({"type": "bind", "source": str(self.session / name), "target": target})

    def written_dir(self, index):
        deliverables = self.metadata["deliverables"] if index == 0 else self.stages[index]["deliverables"]
        return Path(deliverables[0]).parts[0]

    def fill_workspace(self, index):
        workspace = self.session / "workspace"
        stage = self.stages[index]
        shutil.copytree(self.task_dir / "environment", workspace, dirs_exist_ok=True)
        if stage.get("data"):
            (workspace / "data").mkdir(exist_ok=True)
            shutil.copy(self.task_dir / stage["data"], workspace / "data" / "data.json")
        if self.served:
            shutil.copy(self.simulator_dir / "bb_cli.py", workspace / "bb_cli.py")
        for name in (self.written_dir(index), "observations", "tmp/hf", *self.workspace_dirs):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        for entry in stage.get("carry", []):
            source = next((self.session / "handed" / path for path in entry["from"]
                           if (self.session / "handed" / path).exists()), None)
            if source is None:
                continue
            if source.is_dir():
                shutil.copytree(source, workspace / entry["to"], dirs_exist_ok=True)
            else:
                (workspace / entry["to"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, workspace / entry["to"])
        if self.agent == "claude-code" and not self.is_verifier:
            shutil.copy(workspace / "AGENTS.md", workspace / "CLAUDE.md")

    def stage_instruction(self, index):
        stage = self.stages[index]
        text = (self.task_dir / stage["instruction"]).read_text()
        if stage.get("listing"):
            lines = [f"  - `{WORKSPACE}/{entry['to']}`: {entry['note']}" for entry in stage["carry"]
                     if (self.session / "workspace" / entry["to"]).exists()]
            text = text.replace(stage["listing"], "\n".join(lines))
        return text

    @property
    def stage_timeout_sec(self):
        return self.stages[self.stage]["timeout_sec"]

    async def begin_stage(self, index):
        needs = self.stages[index].get("needs")
        if needs and not (self.session / "handed" / needs).is_file():
            return False
        self.stage = index
        await super().stop(delete=False)
        await asyncio.to_thread(self.stop_simulator)
        for name in ("workspace", "tmp", "home"):
            empty(self.session / name)
        shutil.copy(PACKAGE / "codex_models.json", self.session / "home" / "codex_models.json")
        self.fill_workspace(index)
        if self.served:
            await asyncio.to_thread(self.start_simulator)
        await super().start(force_build=False)
        await self.verify_isolation()
        return True

    def end_stage(self):
        stage = self.stages[self.stage]
        written = self.written_dir(self.stage)
        collect(self.session / "workspace" / written, self.session / "handed" / stage.get("collect", written),
                self.metadata["mechanisms_max_mb"] if self.stage == 0 else stage.get("max_mb"))
        logs = self.trial_paths.agent_dir
        names = {entry["name"] for entry in self.stages}
        (logs / stage["name"]).mkdir(parents=True, exist_ok=True)
        for path in [p for p in logs.iterdir() if p.name not in names]:
            shutil.move(path, logs / stage["name"] / path.name)

    def simulator_python(self):
        if self.venvs_dir is not None:
            own = self.venvs_dir / self.simulator_family / "bin" / "python"
            if own.is_file():
                return str(own)
        return sys.executable

    def simulator_settings(self):
        settings = dict(self.metadata["simulator"])
        if self.is_verifier and "verifier_args" in settings:
            settings["args"] = settings["verifier_args"]
        if self.is_verifier:
            settings["verifier"] = True
            settings["bridge_key"] = self.bridge_key
        elif self.records:
            settings["args"] = dict(settings.get("args", {}),
                                    record_path=str(self.trial_paths.trial_dir / self.records))
        if not self.is_verifier and self.stage:
            settings["args"] = dict(settings.get("args", {}), **self.stages[self.stage].get("simulator", {}))
        return settings

    def start_simulator(self):
        port = free_port()
        role = "verifier" if self.is_verifier else "agent"
        suffix = f"_{self.stages[self.stage]['name']}" if self.stage else ""
        log_path = self.trial_paths.trial_dir / f"simulator_{role}{suffix}.log"
        env = dict(os.environ, EUREKA_SESSION=str(self.session),
                   EUREKA_AGENT_IMAGE=str(self.sif_dir / self.task_env_config.docker_image))
        with open(log_path, "w") as log:
            self.simulator = subprocess.Popen(
                [self.simulator_python(), str(PACKAGE / "simulator_server.py"),
                 "--simulator", str(self.simulator_dir / "simulator.py"), "--port", str(port),
                 "--settings", json.dumps(self.simulator_settings()),
                 "--export_dir", str(self.session / "workspace" / "observations"),
                 "--export_prefix", f"{WORKSPACE}/observations",
                 "--work_dir", tempfile.mkdtemp(prefix="simulator_", dir=self.session)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + SIMULATOR_START_SEC
        while not ping(url):
            if self.simulator.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"the simulator did not start; see {log_path}")
            time.sleep(3)
        self.simulator_url = url
        if not self.is_verifier:
            for host in simulator_hosts(url):
                if host not in self.metadata["blocked_hosts"]:
                    self.metadata["blocked_hosts"].append(host)
                    with open(self.session / "hosts", "a") as f:
                        f.write(f"0.0.0.0\t{host}\n")
        cli = self.session / "workspace" / "bb_cli.py"
        text = cli.read_text()
        if CLI_URL_LINE.search(text):
            cli.write_text(CLI_URL_LINE.sub(f'SERVER_URL = "{url}"', text, count=1))
        elif CLI_PORT_LINE.search(text):
            cli.write_text(CLI_PORT_LINE.sub(f"PORT = {port}", text, count=1))
        else:
            raise RuntimeError(f"{cli} carries no SERVER_URL or PORT line to point at the simulator")

    def start_proxies(self):
        env = {"APPTAINER_CONTAINER": "/srv/image.sif", "SINGULARITY_CONTAINER": "/srv/image.sif",
               "HF_HOME": f"{WORKSPACE}/tmp/hf", "NODE_TLS_REJECT_UNAUTHORIZED": "0",
               "NVIDIA_VISIBLE_DEVICES": self.gpu, "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
               "CUDA_VISIBLE_DEVICES": "0"}
        if self.is_verifier:
            judges = []
            for agent, model in self.judges:
                server = proxy.start(0, agent, model)
                self.proxies.append(server)
                judges.append({"agent": agent, "model": model,
                               "url": f"http://127.0.0.1:{server.server_address[1]}"})
            env["EUREKA_JUDGES"] = json.dumps(judges)
            env["EUREKA_JUDGE_EFFORT"] = self.judge_effort
            env["EUREKA_DELIVERABLES"] = json.dumps(self.metadata["deliverables"])
            env["EUREKA_ARTIFACTS"] = json.dumps([str(rel) for rel in self.artifacts])
            env["EUREKA_JUDGE"] = json.dumps(self.metadata.get("judge", {}))
            if self.simulator_url:
                env["EUREKA_SIMULATOR_URL"] = self.simulator_url
                env["EUREKA_SIMULATOR_KEY"] = self.bridge_key
        else:
            self.proxies.append(proxy.start(self.proxy_port, self.agent, self.model, self.metadata["blocked_hosts"],
                                            self.metadata["source_literature"],
                                            log=self.trial_paths.trial_dir / "proxy_agent.log",
                                            vocabulary=self.metadata.get("task_vocabulary", ())))
        env.update(self.metadata.get("sandbox_env", {}))
        self._persistent_env.update(env)

    def missing_deliverables(self):
        deliverables = self.metadata["deliverables"] if self.stage == 0 else self.stages[self.stage]["deliverables"]
        return [name for name in deliverables if not (self.session / "workspace" / name).exists()]

    def search_stops(self):
        log = self.trial_paths.trial_dir / "proxy_agent.log"
        return log.read_text().count(" stopped ") if log.exists() else 0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec or EXEC_TIMEOUT_SEC,
                                  user=user)

    async def download_dir(self, source_dir, target_dir):
        if not str(source_dir).startswith(WORKSPACE + "/"):
            return await super().download_dir(source_dir, target_dir)
        rel = Path(source_dir).relative_to(WORKSPACE)
        if (self.session / "handed" / rel).is_dir():
            collect(self.session / "handed" / rel, Path(target_dir), None)
        else:
            collect(self.session / "workspace" / rel, Path(target_dir), self.metadata["mechanisms_max_mb"])

    def stop_simulator(self):
        if self.simulator is None:
            return
        group = self.simulator.pid
        for sig, wait_sec in ((signal.SIGTERM, 10), (signal.SIGKILL, 60)):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                break
            deadline = time.time() + wait_sec
            while time.time() < deadline:
                self.simulator.poll()
                try:
                    os.killpg(group, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.5)
        self.simulator = None
        self.simulator_url = ""

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            for server in self.proxies:
                proxy.stop(server)
            await asyncio.to_thread(self.stop_simulator)
            for link in self.relinked:
                hidden = link.readlink()
                link.unlink()
                shutil.move(hidden, link)
            self.relinked = []
            if self.session is not None:
                shutil.rmtree(self.session, ignore_errors=True)
