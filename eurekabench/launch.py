import getpass
import os
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import fire

from eurekabench.proxy import PLACEHOLDER_KEY

PACKAGE = Path(__file__).resolve().parent
AGENTS = {"codex": "eurekabench.agents:Codex", "claude-code": "eurekabench.agents:ClaudeCode"}
BODY_IMAGE = "docker://openworm/openworm:0.9.8"
TRIBE_PACKAGES = ["tribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7",
                  "fire==0.7.1", "imageio-ffmpeg>=0.5"]
TRIBE_OVERRIDES = ["torch==2.9.1", "torchvision==0.24.1", "transformers<5", "huggingface_hub<1", "pandas<3"]
TRIBE_IMPORTS = "import fire, numpy, imageio_ffmpeg, tribev2"
ARCHIVE_PACKAGES = {"camels": ["h5py>=3.10", "numpy>=1.26.4", "fire==0.7.1"],
                    "dreams": ["h5py>=3.10", "requests>=2.32", "globus-sdk>=3.41", "fire==0.7.1"]}
ARCHIVE_IMPORTS = {"camels": "import h5py, numpy, fire", "dreams": "import h5py, requests, globus_sdk, fire"}
TORAX_PACKAGES = ["torax==1.4.3", "fire==0.7.1", "h5py", "contourpy", "scipy", "xarray", "netCDF4",
                  "jax[cuda12]==0.11.2"]
TORAX_IMPORTS = "import torax, xarray, contourpy, h5py, fire"
OFT_VERSION = "v26.9"
OFT_URL = ("https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit/releases/download/"
           f"{OFT_VERSION}/OpenFUSIONToolkit_{OFT_VERSION}-Linux-GNU-x86_64.tar.gz")
JULIA_VERSION = "1.11.9"
JULIA_URL = f"https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-{JULIA_VERSION}-linux-x86_64.tar.gz"
GPEC_REPO = "https://github.com/OpenFUSIONToolkit/GPEC.git"
GPEC_COMMIT = "cb18e386e2f8ddf4447f3434270126772129b7d1"
GPEC_RESOLVE = ('using Pkg; isempty(Pkg.Registry.reachable_registries()) && Pkg.Registry.add("General"); '
                "Pkg.resolve()")
GPEC_INSTALL = "using Pkg; Pkg.instantiate(); Pkg.precompile()"
GPEC_CHECK = ("using GeneralizedPerturbedEquilibrium: LocalStability; "
              "isdefined(LocalStability, :ballooning_alpha_boundaries) || error(\"no ballooning boundaries\")")
LM_PACKAGES = ["torch==2.13.0", "torchvision==0.28.0", "vllm==0.27.1", "transformers==5.15.0", "peft==0.18.1",
               "datasets==4.5.0", "accelerate==1.13.0", "transformer-lens==2.17.0",
               "MIB_circuit_track @ git+https://github.com/hannamw/MIB-circuit-track"
               "@b759df34433c9e31043ba9e02908ce0bf20e894f",
               "scipy==1.17.1", "numpy==2.5.2", "fire==0.7.1", "PyYAML==6.0.3", "huggingface-hub==1.27.0",
               "safetensors==0.8.0", "tokenizers==0.22.2", "wandb==0.25.1", "einops==0.8.2", "jaxtyping==0.3.9",
               "pandas==2.3.3", "scikit-learn==1.8.0", "tqdm==4.67.3", "rich==14.2.0", "sentencepiece==0.2.1",
               "protobuf==6.33.6"]
LM_OVERRIDES = ["numpy==2.5.2", "huggingface-hub==1.27.0", "beartype==0.22.9", "transformers==5.15.0"]
LM_IMPORTS = ("import torch, vllm, transformers, peft, transformer_lens, eap.graph, MIB_circuit_track, datasets, "
              "scipy, fire, yaml, wandb, rich; assert torch.version.cuda, 'a CPU-only torch'")
LM_MODELS = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B", "google/gemma-2-2b"]
LM_IMAGE_MODELS = ["meta-llama/Llama-3.1-8B"]
LM_TOKEN = "export HF_TOKEN first; the gated meta-llama/Llama-3.1-8B and google/gemma-2-2b are downloaded with it"
LM_FETCH = """import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    path = snapshot_download(repo, token=os.environ["HF_TOKEN"],
                             allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
    if not any(Path(path).rglob("*.safetensors")):
        raise SystemExit(f"no weight shards downloaded for {repo}")
    print("ready:", repo, flush=True)
"""
DREAMS_LOGIN = """import importlib.util, sys, tempfile
spec = importlib.util.spec_from_file_location("simulator", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="dreams_login_") as root:
    hosts = module.Simulator(root + "/observations", "/workspace/observations", root).extra_hosts()
print("the DREAMS archive answers from", hosts)
"""
SCORES_LINE = "=" * 78
VERIFIER_SUFFIX = "-verifier"


def domains_dir():
    inside = PACKAGE / "domains"
    return inside if inside.is_dir() else PACKAGE.parent / "domains"


def scratch_path(scratch_dir):
    return Path(scratch_dir) if scratch_dir else Path("/tmp") / getpass.getuser() / "eurekabench"


def required_keys(agent, model):
    if model.startswith("litellm_proxy/"):
        return ["LLM_API_KEY", "LLM_BASE_URL"]
    if agent == "codex":
        return ["OPENAI_API_KEY"]
    return ["ANTHROPIC_API_KEY"]


def task_images(config):
    names = [config["environment"]["docker_image"], config["metadata"].get("verifier_image")]
    return [name for name in names if name]


def build_image(domain, scratch, image):
    name = Path(image).stem
    definition = domains_dir() / domain / ("image.def" if name == domain else f"{name}.def")
    if not definition.is_file():
        sys.exit(f"no image definition at {definition}")
    (scratch / "sif").mkdir(parents=True, exist_ok=True)
    arguments = []
    if domain == "computer-science":
        build_lm(scratch)
        stage = scratch / "hf_stage"
        fetch_models(scratch, LM_IMAGE_MODELS, dict(HF_HOME=str(stage), HF_HUB_DISABLE_SYMLINKS="1"))
        arguments = ["--build-arg", f"HF_STAGE={stage}"]
    if name.endswith(VERIFIER_SUFFIX):
        base = scratch / "sif" / f"{name.removesuffix(VERIFIER_SUFFIX)}.sif"
        if not base.is_file():
            sys.exit(f"the image {image} is built on {base}, which is missing")
        arguments = ["--build-arg", f"BASE={base}"]
    subprocess.run(["apptainer", "build", "--force", *arguments, str(scratch / "sif" / image), str(definition)],
                   check=True)


def build_body(scratch):
    body = scratch / "openworm_sandbox"
    if (body / "home" / "ow").is_dir():
        return
    if body.exists():
        subprocess.run(["chmod", "-R", "u+rwX", str(body)], stderr=subprocess.DEVNULL)
        shutil.rmtree(body, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run(["apptainer", "build", "--sandbox", str(body), BODY_IMAGE], check=True)


def build_tribe(scratch):
    venv = scratch / "venvs" / "tribe"
    python = venv / "bin" / "python"
    if not python.is_file():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    if subprocess.run([str(python), "-c", TRIBE_IMPORTS], capture_output=True).returncode != 0:
        overrides = scratch / "venvs" / "tribe-overrides.txt"
        overrides.write_text("\n".join(TRIBE_OVERRIDES) + "\n")
        subprocess.run(["uv", "pip", "install", "--python", str(python), *TRIBE_PACKAGES,
                        "--override", str(overrides)], check=True)
    subprocess.run([str(python), "-c", TRIBE_IMPORTS], check=True)
    ffmpeg = subprocess.run([str(python), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                            capture_output=True, text=True, check=True).stdout.strip()
    (scratch / "bin").mkdir(parents=True, exist_ok=True)
    link = scratch / "bin" / "ffmpeg"
    link.unlink(missing_ok=True)
    link.symlink_to(ffmpeg)


def build_archive(scratch, family):
    venv = scratch / "venvs" / family
    python = venv / "bin" / "python"
    if not python.is_file():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    if subprocess.run([str(python), "-c", ARCHIVE_IMPORTS[family]], capture_output=True).returncode != 0:
        subprocess.run(["uv", "pip", "install", "--python", str(python), *ARCHIVE_PACKAGES[family]], check=True)
    subprocess.run([str(python), "-c", ARCHIVE_IMPORTS[family]], check=True)


def build_torax(scratch):
    venv = scratch / "venvs" / "torax"
    python = venv / "bin" / "python"
    if not python.is_file():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    if subprocess.run([str(python), "-c", TORAX_IMPORTS], capture_output=True).returncode != 0:
        subprocess.run(["uv", "pip", "install", "--python", str(python), *TORAX_PACKAGES], check=True)
    subprocess.run([str(python), "-c", TORAX_IMPORTS], check=True)


def edge_paths(scratch):
    oft = scratch / "oft" / f"OpenFUSIONToolkit_{OFT_VERSION}-Linux-GNU-x86_64"
    return {"oft": oft, "mesh": oft / "examples" / "TokaMaker" / "DIIID" / "DIIID_mesh.h5",
            "meshes": scratch / "nt_edge_mesh", "julia": scratch / "julia" / f"julia-{JULIA_VERSION}",
            "gpec": scratch / "GPEC", "depot": scratch / "julia_depot",
            "ready": scratch / "GPEC" / ".eurekabench_ready"}


def edge_ready(scratch):
    paths = edge_paths(scratch)
    return paths["mesh"].is_file() and (paths["julia"] / "bin" / "julia").is_file() and paths["ready"].is_file()


def unpack(url, target):
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "download.tar.gz"
    subprocess.run(["curl", "-L", "--fail", "-o", str(archive), url], check=True)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(target)], check=True)
    archive.unlink()


def build_edge(scratch):
    paths = edge_paths(scratch)
    if not (paths["oft"] / "python" / "OpenFUSIONToolkit" / "TokaMaker" / "_core.py").is_file():
        unpack(OFT_URL, scratch / "oft")
    if not paths["mesh"].is_file():
        sys.exit(f"the Open FUSION Toolkit at {paths['oft']} carries no machine mesh at {paths['mesh']}")
    julia = paths["julia"] / "bin" / "julia"
    if not julia.is_file():
        unpack(JULIA_URL, scratch / "julia")
    gpec = paths["gpec"]
    if not (gpec / "Project.toml").is_file():
        gpec.mkdir(parents=True, exist_ok=True)
        for command in (["init", "--quiet"], ["fetch", "--quiet", "--depth", "1", GPEC_REPO, GPEC_COMMIT],
                        ["checkout", "--quiet", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(gpec), *command], check=True)
    env = dict(os.environ, JULIA_DEPOT_PATH=str(paths["depot"]))
    paths["depot"].mkdir(parents=True, exist_ok=True)
    if not (gpec / "Manifest.toml").is_file():
        subprocess.run([str(julia), f"--project={gpec}", "-e", GPEC_RESOLVE], env=env, check=True)
    for script in (GPEC_INSTALL, GPEC_CHECK):
        subprocess.run([str(julia), f"--project={gpec}", "-e", script], env=env, check=True)
    paths["ready"].write_text(GPEC_COMMIT + "\n")


def build_lm(scratch):
    venv = scratch / "venvs" / "lm"
    python = venv / "bin" / "python"
    if not python.is_file():
        subprocess.run(["uv", "venv", str(venv), "--python", "3.12"], check=True)
    if subprocess.run([str(python), "-c", LM_IMPORTS], capture_output=True).returncode != 0:
        overrides = scratch / "venvs" / "lm-overrides.txt"
        overrides.write_text("\n".join(LM_OVERRIDES) + "\n")
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--torch-backend", "cu130", *LM_PACKAGES,
                        "--override", str(overrides)], check=True)
    subprocess.run([str(python), "-c", LM_IMPORTS], check=True)


def fetch_models(scratch, models, env=None):
    subprocess.run([str(scratch / "venvs" / "lm" / "bin" / "python"), "-c", LM_FETCH, *models],
                   env=dict(os.environ, **(env or {})), check=True)


def dreams_token():
    return Path(os.environ.get("DREAMS_TOKEN_FILE") or Path.home() / ".config" / "dreams" / "globus_tokens.json")


def authorize_dreams(scratch, simulator):
    token = dreams_token()
    if token.is_file():
        return
    if not sys.stdin.isatty():
        sys.exit(f"no Globus token at {token}; run this command once in an interactive terminal to authorize "
                 "access to the DREAMS archive")
    print(f"no Globus token at {token}; open the URL printed below in a browser, log in, and paste the code here")
    subprocess.run([str(scratch / "venvs" / "dreams" / "bin" / "python"), "-c", DREAMS_LOGIN, str(simulator)],
                   env=dict(os.environ, DREAMS_TOKEN_FILE=str(token)), check=True)


def build(domain, scratch_dir=""):
    scratch = scratch_path(scratch_dir)
    if domain == "computer-science" and not os.environ.get("HF_TOKEN"):
        sys.exit(LM_TOKEN)
    images = []
    for task in sorted((domains_dir() / domain).glob("*/task.toml")):
        images += [name for name in task_images(tomllib.loads(task.read_text())) if name not in images]
    for image in images:
        build_image(domain, scratch, image)
    if domain == "neuroscience":
        build_body(scratch)
        build_tribe(scratch)
    if domain == "astrophysics":
        build_archive(scratch, "camels")
        build_archive(scratch, "dreams")
        tasks = sorted((domains_dir() / domain).glob("*/task.toml"))
        names = {task: tomllib.loads(task.read_text())["metadata"]["simulator"]["name"] for task in tasks}
        authorize_dreams(scratch, next(task.parent / "simulator" / "simulator.py" for task in tasks
                                       if names[task].startswith("dreams-")))
    if domain == "fusion":
        build_torax(scratch)
        build_edge(scratch)
    if domain == "computer-science":
        fetch_models(scratch, LM_MODELS)


def run(domain, problem, model, judges, agent="codex", effort="xhigh", judge_effort="xhigh", gpu=0, rejudge="",
        scratch_dir="", jobs_dir="jobs"):
    scratch = scratch_path(scratch_dir)
    if agent not in AGENTS:
        sys.exit(f"agent is one of {sorted(AGENTS)}")
    model = str(model)
    judges = ",".join(judges) if isinstance(judges, (list, tuple)) else str(judges)
    pairs = [entry.split(":", 1) for entry in judges.split(",")]
    if any(len(pair) != 2 or pair[0] not in AGENTS for pair in pairs):
        sys.exit(f"judges is a comma-separated list of agent:model with agent one of {sorted(AGENTS)}")
    for scaffold, name in [(agent, model), *pairs]:
        for key in required_keys(scaffold, name):
            if not os.environ.get(key):
                sys.exit(f"export {key} first")

    task_dir = domains_dir() / domain / problem
    if not (task_dir / "task.toml").is_file():
        sys.exit(f"no problem {problem} under {domains_dir() / domain}")
    config = tomllib.loads((task_dir / "task.toml").read_text())
    simulator = config["metadata"]["simulator"]["name"]
    family = simulator.split("-")[0]
    first = config["metadata"]["deliverables"][0]

    if shutil.which("nvidia-container-cli") is None:
        sys.exit("nvidia-container-cli is not on this node, so the sandbox cannot be limited to one card")
    query = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True)
    card = query.stdout.strip() if query.returncode == 0 else ""
    if "H100" not in card:
        sys.exit(f"card {gpu} is {card or 'not readable by nvidia-smi'}, not an H100")
    if simulator == "c302" and shutil.which("java") is None:
        sys.exit("java is not on PATH; the c302 simulator runs jNeuroML with it")
    if simulator == "tribe" and shutil.which("uvx") is None:
        sys.exit("uvx is not on PATH; the TRIBE simulator transcribes audio through uvx whisperx")
    if simulator == "tribe" and not os.environ.get("HF_TOKEN"):
        sys.exit("export HF_TOKEN first; the TRIBE simulator reads text with the gated meta-llama/Llama-3.2-3B")
    if family == "lm" and not os.environ.get("HF_TOKEN"):
        sys.exit(LM_TOKEN)
    for image in task_images(config):
        if not (scratch / "sif" / image).is_file():
            build_image(domain, scratch, image)
    if simulator == "c302":
        build_body(scratch)
    if simulator == "tribe" and not ((scratch / "venvs" / "tribe" / "bin" / "python").is_file()
                                     and (scratch / "bin" / "ffmpeg").exists()):
        build_tribe(scratch)
    if family in ARCHIVE_PACKAGES:
        build_archive(scratch, family)
    if family == "dreams":
        authorize_dreams(scratch, task_dir / "simulator" / "simulator.py")
    if family == "torax" and not (scratch / "venvs" / "torax" / "bin" / "python").is_file():
        build_torax(scratch)
    if simulator == "torax-edge" and not edge_ready(scratch):
        build_edge(scratch)
    if family == "lm":
        if not (scratch / "venvs" / "lm" / "bin" / "python").is_file():
            build_lm(scratch)
        fetch_models(scratch, LM_MODELS)

    rejudge_dir = ""
    if rejudge:
        earlier = Path(rejudge).resolve()
        if not (earlier / "artifacts" / "workspace" / first).is_file() \
                or not (earlier / "verifier" / "eval_results.json").is_file():
            sys.exit(f"{rejudge} holds no handed-in mechanisms or no evaluation to judge again")
        rejudge_dir = str(earlier)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proxy = f"http://127.0.0.1:{port}"
    if rejudge_dir:
        harbor_agent, flags = "nop", []
    elif agent == "codex":
        harbor_agent = AGENTS[agent]
        flags = ["--ak", f"reasoning_effort={effort}", "--ak", "web_search=live",
                 "--ak", 'config={"model_catalog_json": "/root/codex_models.json"}',
                 "--ae", f"OPENAI_BASE_URL={proxy}", "--ae", f"OPENAI_API_KEY={PLACEHOLDER_KEY}"]
    else:
        harbor_agent = AGENTS[agent]
        flags = ["--ak", f"reasoning_effort={effort}",
                 "--ae", f"ANTHROPIC_BASE_URL={proxy}", "--ae", f"ANTHROPIC_API_KEY={PLACEHOLDER_KEY}"]

    env = dict(os.environ, NEURO_BODY_SANDBOX=str(scratch / "openworm_sandbox"), CUDA_VISIBLE_DEVICES=str(gpu),
               TQDM_DISABLE="1", OMP_NUM_THREADS="4", MPLBACKEND="Agg", DREAMS_TOKEN_FILE=str(dreams_token()),
               PATH=os.pathsep.join([str(scratch / "bin"), os.environ.get("PATH", "")]))
    if family == "torax":
        paths = edge_paths(scratch)
        for directory in (scratch / "jax_cache", paths["meshes"]):
            directory.mkdir(parents=True, exist_ok=True)
        env.update(JAX_PLATFORMS="cuda,cpu", JAX_COMPILATION_CACHE_DIR=str(scratch / "jax_cache"),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", FUSION_OFT_THREADS="4", OFT_ROOTPATH=str(paths["oft"]),
                   TOKAMAKER_MESH=str(paths["mesh"]), TOKAMAKER_MESH_DIR=str(paths["meshes"]),
                   GPEC_PROJECT=str(paths["gpec"]), GPEC_THREADS="4", JULIA_DEPOT_PATH=str(paths["depot"]),
                   PATH=os.pathsep.join([str(paths["julia"] / "bin"), env["PATH"]]))
    if not (PACKAGE / "domains").is_dir():
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE.parent), os.environ.get("PYTHONPATH")]))
    task = task_dir.relative_to(Path.cwd()) if task_dir.is_relative_to(Path.cwd()) else task_dir
    job_name = time.strftime("%Y-%m-%d__%H-%M-%S")
    command = [sys.executable, "-P", "-c", "from harbor.cli.main import app; app()", "run",
               "-p", str(task), "-a", harbor_agent, "-m", model, *flags,
               "-e", "eurekabench.environment:Sandbox",
               "--ek", f"sif_dir={scratch / 'sif'}", "--ek", f"agent={agent}", "--ek", f"model={model}",
               "--ek", f"proxy_port={port}", "--ek", f"judges={judges}", "--ek", f"judge_effort={judge_effort}",
               "--ek", f"gpu={gpu}", "--ek", f"rejudge={rejudge_dir}", "--ek", f"venvs_dir={scratch / 'venvs'}",
               "--job-name", job_name, "-o", str(jobs_dir)]
    code = subprocess.run(command, env=env).returncode
    if code:
        sys.exit(code)

    for stdout in sorted(Path(jobs_dir, job_name).glob("*/verifier/test-stdout.txt")):
        print(stdout.parent.parent)
        lines = stdout.read_text().splitlines()
        if SCORES_LINE in lines:
            print("\n".join(lines[lines.index(SCORES_LINE):]))


def main():
    fire.Fire({"build": build, "run": run})


if __name__ == "__main__":
    main()
