import os
import re
import shutil
import urllib.request
import uuid

import h5py
import numpy as np


BASE_URL = "https://users.flatironinstitute.org/~camels"
GENERATION = "L25n256"
SUITES = ["IllustrisTNG", "IllustrisTNG_DM", "SIMBA", "SIMBA_DM"]


def default_data_dir():
    explicit = os.environ.get("ASTRO_DATA_DIR")
    if explicit:
        return explicit
    root = os.environ.get("DATA_ROOT")
    if root:
        return os.path.join(root, "camels")
    return "./camels_data"


def natural_key(name):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


class BasicCamelsBlackBox:
    def __init__(self, data_dir=None, suites=None, set_types=None):
        self.data_dir = data_dir or default_data_dir()
        self.suites = list(suites) if suites else list(SUITES)
        self.set_types = list(set_types) if set_types else None
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch(self, relpath):
        dest = os.path.join(self.data_dir, relpath)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = f"{dest}.part.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{relpath}", timeout=600) as resp:
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dest

    def check_suite(self, suite, set_type=None):
        if suite not in self.suites:
            raise ValueError(f"unknown suite {suite!r}, available suites: {self.suites}")
        if set_type is not None and self.set_types is not None \
                and set_type not in self.set_types:
            raise ValueError(f"unknown set {set_type!r}, available sets: {self.set_types}")

    def list_suites(self):
        return {
            "suites": self.suites,
            "sets": self.set_types,
            "generation": GENERATION,
            "box_size_Mpc_h": 25.0,
        }

    def list_simulations(self, suite, set_type):
        self.check_suite(suite, set_type)
        url = f"{BASE_URL}/Pk/{suite}/{GENERATION}/{set_type}/"
        with urllib.request.urlopen(url, timeout=120) as resp:
            html = resp.read().decode()
        names = sorted(set(re.findall(r'href="([^"/.]+)/"', html)), key=natural_key)
        return {"suite": suite, "set_type": set_type,
                "n_simulations": len(names), "simulations": names}

    def get_params(self, suite, set_type):
        self.check_suite(suite, set_type)
        base = suite[: -len("_DM")] if suite.endswith("_DM") else suite
        path = self.fetch(
            f"Parameters/{suite}/CosmoAstroSeed_{base}_{GENERATION}_{set_type}.txt")
        with open(path) as f:
            columns = f.readline().strip().lstrip("#").split()
            n_rows = sum(1 for line in f if line.strip())
        return {"path": path, "columns": columns, "n_simulations": n_rows,
                "note": "plain text, one row per simulation, first column is the simulation name"}

    def get_power_spectrum(self, suite, set_type, sim_name, ptype="m", redshift=0.0):
        self.check_suite(suite, set_type)
        path = self.fetch(
            f"Pk/{suite}/{GENERATION}/{set_type}/{sim_name}/Pk_{ptype}_z={float(redshift):.2f}.txt")
        k, pk = np.loadtxt(path, unpack=True)
        return {"path": path, "n_bins": int(len(k)),
                "columns": ["k [h/Mpc]", "P(k) [(Mpc/h)^3]"],
                "k_min": float(k.min()), "k_max": float(k.max())}

    def get_group_catalog(self, suite, set_type, sim_name, snapshot=90):
        self.check_suite(suite, set_type)
        path = self.fetch(
            f"FOF_Subfind/{suite}/{GENERATION}/{set_type}/{sim_name}/groups_{int(snapshot):03d}.hdf5")
        with h5py.File(path, "r") as f:
            header = dict(f["Header"].attrs)
            group_fields = sorted(f["Group"].keys()) if "Group" in f else []
            subhalo_fields = sorted(f["Subhalo"].keys()) if "Subhalo" in f else []
        n_subhalos = header.get("Nsubgroups_Total", header.get("Nsubhalos_Total", -1))
        return {"path": path,
                "redshift": float(header["Redshift"]),
                "box_size_ckpc_h": float(header["BoxSize"]),
                "n_groups": int(header.get("Ngroups_Total", -1)),
                "n_subhalos": int(n_subhalos),
                "group_fields": group_fields,
                "subhalo_fields": subhalo_fields}

    def get_snapshot(self, suite, set_type, sim_name, snapshot=90):
        self.check_suite(suite, set_type)
        path = self.fetch(
            f"Sims/{suite}/{GENERATION}/{set_type}/{sim_name}/snapshot_{int(snapshot):03d}.hdf5")
        with h5py.File(path, "r") as f:
            header = dict(f["Header"].attrs)
            part_types = sorted(k for k in f.keys() if k.startswith("PartType"))
            fields = {p: sorted(f[p].keys()) for p in part_types}
        return {"path": path,
                "redshift": float(header["Redshift"]),
                "box_size_ckpc_h": float(header["BoxSize"]),
                "n_particles_by_type": [int(n) for n in header["NumPart_Total"]],
                "particle_fields": fields}


import json
import threading
from pathlib import Path


MOUNT = "camels_data"
COMMANDS = ('list_suites', 'list_simulations', 'get_params', 'get_power_spectrum', 'get_group_catalog', 'get_snapshot')
EVAL_METHODS = ('fetch', 'list_suites', 'list_simulations', 'get_params', 'get_power_spectrum', 'get_group_catalog', 'get_snapshot')
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
    def __init__(self, export_dir, export_prefix, work_dir, suites=(), set_types=(), record_path="", bridge_key=""):
        workspace = Path(export_dir).resolve().parent
        self.sim = BasicCamelsBlackBox(data_dir=str(workspace / MOUNT), suites=list(suites) or None,
                   set_types=list(set_types) or None)
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
            if not self.record_path:
                return
            root = Path(self.sim.data_dir)
            names = sorted({p.name for p in root.rglob("*") if p.is_file()}) if root.is_dir() else []
            with open(self.record_path, "w") as f:
                json.dump({"n_accessed": len(names), "files": names}, f, indent=2)

    def list_suites(self):
        return self.served(self.sim.list_suites, {})

    def list_simulations(self, suite, set_type):
        return self.served(self.sim.list_simulations, {"suite": suite, "set_type": set_type})

    def get_params(self, suite, set_type):
        return self.served(self.sim.get_params, {"suite": suite, "set_type": set_type})

    def get_power_spectrum(self, suite, set_type, sim_name, ptype='m', redshift=0.0):
        return self.served(self.sim.get_power_spectrum, {"suite": suite, "set_type": set_type, "sim_name": sim_name, "ptype": ptype, "redshift": redshift})

    def get_group_catalog(self, suite, set_type, sim_name, snapshot=90):
        return self.served(self.sim.get_group_catalog, {"suite": suite, "set_type": set_type, "sim_name": sim_name, "snapshot": snapshot})

    def get_snapshot(self, suite, set_type, sim_name, snapshot=90):
        return self.served(self.sim.get_snapshot, {"suite": suite, "set_type": set_type, "sim_name": sim_name, "snapshot": snapshot})

    def container_data_dir(self):
        return self.prefix

    def call(self, key, name, args=(), kwargs=None):
        if key != self.bridge_key or name not in EVAL_METHODS:
            raise ValueError("request rejected")
        return self.report(getattr(self.sim, name)(*args, **(kwargs or {})))
