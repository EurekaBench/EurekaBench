import json
import os
import re
import threading
import uuid
from pathlib import Path

import h5py
import requests
from globus_sdk import NativeAppAuthClient, RefreshTokenAuthorizer, TransferAPIError, TransferClient
from globus_sdk.scopes import TransferScopes

CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"
COLLECTION = "fa06ab4e-3fb5-436e-b535-99b4995553ad"
HTTPS_SCOPE = f"https://auth.globus.org/scopes/{COLLECTION}/https"
TRANSFER_RS = "transfer.api.globus.org"

CATALOG_ROOT = "/FOF_Subfind"


def default_data_dir():
    explicit = os.environ.get("ASTRO_DATA_DIR")
    if explicit:
        return explicit
    root = os.environ.get("DATA_ROOT")
    if root:
        return os.path.join(root, "dreams_cdm_mw")
    return "./dreams_data"


def default_token_file():
    explicit = os.environ.get("DREAMS_TOKEN_FILE")
    if explicit:
        return Path(explicit)
    return Path.home() / ".config" / "dreams" / "globus_tokens.json"


def trailing_int(name):
    m = re.search(r"(\d+)$", Path(name).stem)
    return int(m.group(1)) if m else None


class DreamsSimulator:
    def __init__(self, data_dir=None, token_file=None):
        self.data_dir = data_dir or default_data_dir()
        os.makedirs(self.data_dir, exist_ok=True)
        self.token_file = Path(token_file) if token_file else default_token_file()
        self.tc = None
        self.https_server = None
        self.https_auth = None
        self.catalog_templates = {}
        self.connect_lock = threading.Lock()

    def login(self, extra_scopes=()):
        client = NativeAppAuthClient(CLIENT_ID)
        scopes = [TransferScopes.all, HTTPS_SCOPE, *extra_scopes]
        client.oauth2_start_flow(requested_scopes=scopes, refresh_tokens=True)
        print("\n=== Globus authorization needed ===")
        print("Open this URL in any browser, log in, and approve access:\n")
        print(client.oauth2_get_authorize_url())
        try:
            code = input("\nPaste the authorization code here and press Enter: ").strip()
        except EOFError:
            raise RuntimeError(
                f"no usable Globus token at {self.token_file} and no terminal to log in "
                "from; authorize access to the archive once in an interactive terminal")
        tokens = client.oauth2_exchange_code_for_tokens(code).by_resource_server
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps(tokens))
        os.chmod(self.token_file, 0o600)
        return tokens

    def make_authorizer(self, tokens, resource_server):
        tok = tokens[resource_server]
        return RefreshTokenAuthorizer(
            tok["refresh_token"],
            NativeAppAuthClient(CLIENT_ID),
            access_token=tok["access_token"],
            expires_at=tok.get("expires_at_seconds"),
        )

    def connect(self):
        with self.connect_lock:
            if self.tc is not None:
                return
            legacy = Path(self.data_dir) / "globus_tokens.json"
            if not self.token_file.exists() and legacy.exists():
                self.token_file.parent.mkdir(parents=True, exist_ok=True)
                self.token_file.write_bytes(legacy.read_bytes())
                os.chmod(self.token_file, 0o600)
                print(f"[globus] moved the token from {legacy} to {self.token_file}")
            if self.token_file.exists():
                tokens = json.loads(self.token_file.read_text())
            else:
                tokens = self.login()
            tc = TransferClient(authorizer=self.make_authorizer(tokens, TRANSFER_RS))
            try:
                tc.operation_ls(COLLECTION, path="/")
            except TransferAPIError as err:
                if err.info.consent_required:
                    tokens = self.login(extra_scopes=err.info.consent_required.required_scopes)
                    tc = TransferClient(authorizer=self.make_authorizer(tokens, TRANSFER_RS))
                    tc.operation_ls(COLLECTION, path="/")
                else:
                    raise
            endpoint = tc.get_endpoint(COLLECTION)
            https_server = endpoint.data.get("https_server")
            if not https_server:
                raise RuntimeError("the DREAMS collection has no HTTPS server enabled; "
                                   "direct download is not possible")
            if COLLECTION not in tokens:
                tokens = self.login()
            self.tc = tc
            self.https_server = https_server.rstrip("/")
            self.https_auth = self.make_authorizer(tokens, COLLECTION)

    def ls(self, path):
        self.connect()
        return [dict(name=e["name"], type=e["type"], size=e.get("size", 0))
                for e in self.tc.operation_ls(COLLECTION, path=path)]

    def fetch(self, relpath):
        dest = os.path.join(self.data_dir, relpath.lstrip("/"))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        self.connect()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = self.https_server + requests.utils.quote(relpath)
        tmp = f"{dest}.part.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            r = requests.get(url, headers={"Authorization": self.https_auth.get_authorization_header()},
                             stream=True, timeout=600)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dest

    def suite_path(self):
        cache = Path(self.data_dir) / "suite_path.txt"
        if cache.exists():
            return cache.read_text().strip()
        queue = [(CATALOG_ROOT, 0)]
        seen = []
        while queue:
            path, depth = queue.pop(0)
            entries = self.ls(path)
            subdirs = [e for e in entries if e["type"] == "dir"]
            simdirs = [e for e in subdirs if trailing_int(e["name"]) is not None]
            seen.append(f"{path}: {[e['name'] for e in entries][:12]}")
            low = path.lower()
            if "cdm" in low and "mw" in low and "nbody" not in low and len(simdirs) >= 8:
                os.makedirs(self.data_dir, exist_ok=True)
                cache.write_text(path)
                return path
            if depth < 3 and len(subdirs) < 30:
                for e in subdirs:
                    queue.append((f"{path}/{e['name']}", depth + 1))
        raise RuntimeError("could not locate the CDM Milky-Way zoom-in suite under "
                           f"{CATALOG_ROOT}. Listings seen:\n" + "\n".join(seen))

    def kind_suite(self, kind):
        if kind not in ("hydro", "nbody"):
            raise ValueError(f"unknown kind {kind!r}, use 'hydro' or 'nbody'")
        return self.suite_path() + ("_Nbody" if kind == "nbody" else "")

    def sim_names(self):
        suite = self.suite_path()
        dirs = [e["name"] for e in self.ls(suite)
                if e["type"] == "dir" and trailing_int(e["name"]) is not None]
        return suite, sorted(dirs, key=trailing_int)

    def list_simulations(self):
        suite, names = self.sim_names()
        return {"suite": suite, "n_simulations": len(names), "simulations": names}

    def params_file(self):
        suite = self.suite_path()
        parts = [p.lower() for p in suite.strip("/").split("/")[1:]]
        files = []
        queue = [("/Parameters", 0)]
        while queue:
            path, depth = queue.pop(0)
            for e in self.ls(path):
                full = f"{path}/{e['name']}"
                if e["type"] == "file":
                    files.append(full)
                elif depth < 5:
                    queue.append((full, depth + 1))
        scores = [sum(p in f.lower() for p in parts) for f in files]
        best = max(scores, default=0)
        candidates = [f for f, s in zip(files, scores) if s == best and s > 0]
        if len(candidates) != 1:
            raise RuntimeError(f"expected exactly one parameter file matching {parts} "
                               f"under /Parameters, found {candidates}")
        return self.fetch(candidates[0])

    def get_params(self):
        path = self.params_file()
        with open(path) as f:
            columns = f.readline().strip().lstrip("#").replace(",", " ").split()
            n_rows = sum(1 for line in f if line.strip())
        return {"path": path, "columns": columns, "n_simulations": n_rows,
                "note": "plain text, a header line naming the columns, then one row "
                        "per simulation in box order"}

    def find_catalog(self, suite, sim_name, tag):
        for base in [f"{suite}/{sim_name}", f"{suite}/{sim_name}/output"]:
            try:
                entries = self.ls(base)
            except TransferAPIError:
                continue
            hits = [e for e in entries if e["type"] == "file"
                    and tag in e["name"] and e["name"].endswith(".hdf5")]
            if hits:
                return [f"{base}/{e['name']}" for e in hits]
            groupdirs = [e for e in entries if e["type"] == "dir" and tag in e["name"]]
            if groupdirs:
                gbase = f"{base}/{groupdirs[0]['name']}"
                inner = [e for e in self.ls(gbase)
                         if e["type"] == "file" and e["name"].endswith(".hdf5")]
                if inner:
                    return [f"{gbase}/{e['name']}" for e in inner]
        raise RuntimeError(f"no catalog for output {tag} under {suite}/{sim_name}")

    def catalog_paths(self, sim_name, snapshot=90, kind="hydro"):
        tag = f"{int(snapshot):03d}"
        suite = self.kind_suite(kind)
        cache = Path(self.data_dir) / f"catalog_template_{kind}.txt"
        if kind not in self.catalog_templates and cache.exists():
            self.catalog_templates[kind] = cache.read_text().strip()
        template = self.catalog_templates.get(kind)
        if template:
            rel = template.format(suite=suite, sim=sim_name, tag=tag)
            try:
                return [self.fetch(rel)]
            except Exception:
                self.catalog_templates.pop(kind, None)
        rels = self.find_catalog(suite, sim_name, tag)
        if len(rels) == 1:
            template = rels[0].replace(suite, "{suite}").replace(f"/{sim_name}/", "/{sim}/")
            template = template.replace(tag, "{tag}")
            self.catalog_templates[kind] = template
            os.makedirs(self.data_dir, exist_ok=True)
            cache.write_text(template)
        return [self.fetch(r) for r in rels]

    def get_group_catalog(self, sim_name, snapshot=90, kind="hydro"):
        paths = self.catalog_paths(sim_name, snapshot, kind)
        with h5py.File(paths[0], "r") as f:
            header = dict(f["Header"].attrs)
            group_fields = sorted(f["Group"].keys()) if "Group" in f else []
            subhalo_fields = sorted(f["Subhalo"].keys()) if "Subhalo" in f else []
        n_subhalos = header.get("Nsubgroups_Total", header.get("Nsubhalos_Total", -1))
        return {"path": paths[0], "paths": paths, "n_files": len(paths), "kind": kind,
                "redshift": float(header["Redshift"]),
                "box_size_ckpc_h": float(header["BoxSize"]),
                "n_groups": int(header.get("Ngroups_Total", -1)),
                "n_subhalos": int(n_subhalos),
                "group_fields": group_fields,
                "subhalo_fields": subhalo_fields}

    def snapshot_relpath(self, sim_name, snapshot=90, kind="hydro"):
        suite = self.kind_suite(kind)
        return f"/Sims{suite[len(CATALOG_ROOT):]}/{sim_name}/snap_{int(snapshot):03d}.hdf5"

    def get_snapshot(self, sim_name, snapshot=90, kind="hydro"):
        path = self.fetch(self.snapshot_relpath(sim_name, snapshot, kind))
        with h5py.File(path, "r") as f:
            header = dict(f["Header"].attrs)
            part_types = sorted(k for k in f.keys() if k.startswith("PartType"))
            fields = {p: sorted(f[p].keys()) for p in part_types}
        return {"path": path, "kind": kind,
                "redshift": float(header["Redshift"]),
                "box_size_ckpc_h": float(header["BoxSize"]),
                "n_particles_by_type": [int(n) for n in header["NumPart_Total"]],
                "mass_table": [float(m) for m in header["MassTable"]],
                "particle_fields": fields}


from urllib.parse import urlparse


MOUNT = "dreams_data"
COMMANDS = ('list_simulations', 'get_params', 'get_group_catalog', 'get_snapshot')
EVAL_METHODS = ('list_simulations', 'get_params', 'params_file', 'catalog_paths', 'sim_names', 'suite_path', 'kind_suite', 'fetch', 'ls', 'snapshot_relpath', 'get_group_catalog', 'get_snapshot')
VERIFIER_COMMANDS = COMMANDS + ("call", "container_data_dir")


def map_paths(value, src, dst):
    if isinstance(value, str):
        return value.replace(src, dst)
    if isinstance(value, dict):
        return {k: map_paths(v, src, dst) for k, v in value.items()}
    if isinstance(value, list):
        return [map_paths(v, src, dst) for v in value]
    return value


class Simulator:
    def __init__(self, export_dir, export_prefix, work_dir, record_path="", bridge_key=""):
        workspace = Path(export_dir).resolve().parent
        self.sim = DreamsSimulator(data_dir=str(workspace / MOUNT))
        self.prefix = export_prefix.rsplit("/", 1)[0] + "/" + MOUNT
        self.record_path = record_path
        self.bridge_key = bridge_key
        self.accessed = set()
        self.lock = threading.Lock()

    def report(self, value):
        return map_paths(value, self.sim.data_dir, self.prefix)

    def served(self, function, params, logged=False):
        try:
            result = function(**params)
        except Exception as exc:
            message = str(exc).replace(self.sim.data_dir, self.prefix).replace(os.path.expanduser("~"), "~")
            raise RuntimeError(f"{type(exc).__name__}: {message}") from None
        self.record(params.get("sim_name") if logged else None)
        return self.report(result)

    def record(self, sim_name):
        with self.lock:
            if sim_name:
                self.accessed.add(str(sim_name))
            if not self.record_path:
                return
            boxes = set(self.accessed)
            root = Path(self.sim.data_dir)
            if root.is_dir():
                for p in root.rglob("box_*"):
                    if p.is_dir() and any(f.is_file() for f in p.rglob("*")):
                        boxes.add(p.name)
            with open(self.record_path, "w") as f:
                json.dump({"n_accessed": len(boxes),
                           "boxes": sorted(boxes, key=lambda n: trailing_int(n) or 0)}, f, indent=2)

    def list_simulations(self):
        return self.served(self.sim.list_simulations, {})

    def get_params(self):
        return self.served(self.sim.get_params, {})

    def get_group_catalog(self, sim_name, snapshot=90, kind='hydro'):
        return self.served(self.sim.get_group_catalog, {"sim_name": sim_name, "snapshot": snapshot, "kind": kind}, logged=True)

    def get_snapshot(self, sim_name, snapshot=90, kind='hydro'):
        return self.served(self.sim.get_snapshot, {"sim_name": sim_name, "snapshot": snapshot, "kind": kind}, logged=True)

    def container_data_dir(self):
        return self.prefix

    def call(self, key, name, args=(), kwargs=None):
        if key != self.bridge_key or name not in EVAL_METHODS:
            raise ValueError("request rejected")
        return self.report(getattr(self.sim, name)(*args, **(kwargs or {})))

    def extra_hosts(self):
        self.sim.connect()
        host = urlparse(self.sim.https_server).hostname
        return [host] if host else []

    def passthrough(self, verb, relpath, headers):
        headers = {k.lower(): v for k, v in headers.items()}
        if not self.bridge_key or headers.get("authorization") != f"Bearer {self.bridge_key}":
            return 403, {"Content-Type": "text/plain"}, b"request rejected"
        self.sim.connect()
        forward = {"Authorization": self.sim.https_auth.get_authorization_header()}
        if "range" in headers:
            forward["Range"] = headers["range"]
        r = requests.request(verb, self.sim.https_server + requests.utils.quote(relpath), headers=forward,
                             timeout=600)
        kept = {k: r.headers[k] for k in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges")
                if k in r.headers}
        return r.status_code, kept, (b"" if verb == "HEAD" else r.content)
