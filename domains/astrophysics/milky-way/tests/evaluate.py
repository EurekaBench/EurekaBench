import ast
import importlib.util
import inspect
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import h5py
import numpy as np
import requests
from archive import CATALOG_ROOT, DreamsBlackBox, trailing_int

SNAPSHOT = 90
N_PROFILE_BINS = 30
R_OVER_R200_LIMITS = (0.01, 1.0)
MAX_LOWRES_FRACTION = 0.02
MIN_HOST_DM_PARTICLES = 100_000
HOST_MASS_WINDOW_MSUN = (3e11, 5e12)
OMEGA_B = 0.046
HOST_DEFINITION = ("the most massive Friends-of-Friends group of the zoom-in whose "
                   "dark matter holds fewer than 2 percent low-resolution boundary "
                   "particles (GroupLenType[:, 2] against GroupLenType[:, 1]); its "
                   "profile is measured in spherical shells around its GroupPos, "
                   "over that group's particles. Boxes whose most massive clean "
                   "group falls outside (3e11, 5e12) Msun in M200c are skipped")


def minimal_image(delta, box):
    return delta - np.round(delta / box) * box


def read_param_table(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    if not lines[0].startswith("#"):
        raise RuntimeError(f"{path} has no header line naming its columns")
    columns = lines[0].lstrip("#").split()
    rows = {}
    for box_id, line in enumerate(lines[1:]):
        values = line.split()
        if len(values) != len(columns):
            raise RuntimeError(f"{path} row {box_id}: {len(values)} values for "
                               f"{len(columns)} columns")
        rows[box_id] = {c: float(v) for c, v in zip(columns, values)}
    return columns, rows


def find_host(cat_paths):
    if len(cat_paths) != 1:
        raise RuntimeError(f"expected a single-file catalog, got {cat_paths}")
    with h5py.File(cat_paths[0], "r") as f:
        header = dict(f["Header"].attrs)
        group = f["Group"]
        len_type = group["GroupLenType"][:].astype(np.int64)
        m200 = group["Group_M_Crit200"][:].astype(np.float64)
        r200 = group["Group_R_Crit200"][:].astype(np.float64)
        pos = group["GroupPos"][:].astype(np.float64)
    hubble = float(header["HubbleParam"])
    hi, lo = len_type[:, 1], len_type[:, 2]
    clean = (hi >= MIN_HOST_DM_PARTICLES) & (lo < MAX_LOWRES_FRACTION * (hi + lo))
    if not clean.any():
        return None
    idx = int(np.flatnonzero(clean)[np.argmax(m200[clean])])
    m200_msun = m200[idx] * 1e10 / hubble
    if not (HOST_MASS_WINDOW_MSUN[0] <= m200_msun <= HOST_MASS_WINDOW_MSUN[1]):
        return None
    return {"index": idx,
            "pos": pos[idx],
            "m200_msun": float(m200_msun),
            "r200_ckpc_h": float(r200[idx]),
            "r200_kpc": float(r200[idx] / hubble),
            "lowres_fraction": float(lo[idx] / max(hi[idx] + lo[idx], 1)),
            "type_start": len_type[:idx].sum(axis=0).tolist(),
            "type_count": len_type[idx].tolist(),
            "box_size": float(header["BoxSize"]),
            "hubble": hubble,
            "omega_m": float(header["Omega0"]),
            "redshift": float(header["Redshift"])}


http_local = threading.local()


def http_session():
    if not hasattr(http_local, "session"):
        http_local.session = requests.Session()
    return http_local.session


def range_get(bb, relpath, lo, hi):
    bb.connect()
    url = bb.https_server + requests.utils.quote(relpath)
    r = http_session().get(url, headers={
        "Authorization": bb.https_auth.get_authorization_header(),
        "Range": f"bytes={lo}-{hi}"}, timeout=600)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"server ignored the Range request for {relpath}")
    return r.content


def remote_size(bb, relpath):
    bb.connect()
    url = bb.https_server + requests.utils.quote(relpath)
    r = http_session().head(url, headers={
        "Authorization": bb.https_auth.get_authorization_header()}, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


class RangeFile:
    def __init__(self, bb, relpath, size, block=1 << 18):
        self.bb = bb
        self.relpath = relpath
        self.size = size
        self.block = block
        self.pos = 0
        self.cache = {}

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b, o = divmod(self.pos, self.block)
            if b not in self.cache:
                lo = b * self.block
                hi = min((b + 1) * self.block, self.size) - 1
                self.cache[b] = range_get(self.bb, self.relpath, lo, hi)
                if len(self.cache) > 128:
                    self.cache.pop(next(iter(self.cache)))
            chunk = self.cache[b][o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
            if not chunk:
                break
        return bytes(out)

    def seek(self, pos, whence=0):
        if whence == 1:
            pos += self.pos
        elif whence == 2:
            pos += self.size
        self.pos = pos
        return self.pos

    def tell(self):
        return self.pos


def read_rows(bb, relpath, field, start, count):
    shape, dtype = field["shape"], field["dtype"]
    if start < 0 or start + count > shape[0]:
        raise ValueError(f"rows [{start}, {start + count}) outside {shape}")
    ncols = shape[1] if len(shape) > 1 else 1
    row = ncols * dtype.itemsize
    lo = field["offset"] + start * row
    arr = np.frombuffer(range_get(bb, relpath, lo, lo + count * row - 1), dtype=dtype)
    return arr.reshape(count, ncols) if ncols > 1 else arr


def host_distances(pos, host):
    delta = minimal_image(pos - host["pos"], host["box_size"])
    dist = np.linalg.norm(delta, axis=1)
    if np.mean(dist < 5.0 * host["r200_ckpc_h"]) < 0.99:
        raise RuntimeError("host particle slice is not concentrated around the "
                           "host; the snapshot does not look group-ordered")
    return dist


MAX_COEFFS = 10
ALLOWED_IMPORTS = ("numpy", "scipy", "h5py")
PREDICT_TIMEOUT = 7200.0
FIT_TIMEOUT = 7200.0
HEARTBEAT = 15.0
MIN_EVAL_SIMS = 20
MAX_HOST_MATCH_DEX = 0.15

EDGES = np.geomspace(*R_OVER_R200_LIMITS, N_PROFILE_BINS + 1)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
INNER_R_OVER_R200 = 0.1

PA_IDS = ["PA1", "PA2"]

INTERFACE = ("COEFFS", "predict_profile", "fit_coeffs")

DM_TYPES = [1, 2]
DM_PARTICLE_GROUPS = ("PartType1", "PartType2", "PartType3")
BARYON_PARTICLE_GROUPS = ("PartType0", "PartType4", "PartType5")
WITHHELD_PARTICLE_FIELDS = ("Potential",)
HYDRO_BARYON_WORDS = ("Gas", "Star", "Stellar", "BH", "SFR", "Wind", "Bfld", "Metal")
HYDRO_LOCATOR_FIELDS = ("GroupPos", "GroupVel", "GroupFirstSub", "GroupNsubs",
                        "SubhaloPos", "SubhaloVel", "SubhaloGrNr", "SubhaloParent",
                        "SnapNum")
SERVED_DIR = "served_without_dark_matter"

SERVED_PARTICLE_TYPES = {"hydro": (0, 4, 5), "nbody": (1, 2)}


def progress(label, done, total, t0, note=""):
    elapsed = time.time() - t0
    frac = done / total if total else 1.0
    filled = int(round(30 * frac))
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    print(f"[{label}] |{'#' * filled}{'.' * (30 - filled)}| {done}/{total} "
          f"elapsed {elapsed:5.0f}s eta {eta:5.0f}s {note}", flush=True)


def call_with_timeout(fn, timeout, *args):
    out, error = [], []

    def run():
        try:
            out.append(fn(*args))
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    t0 = time.time()
    worker.start()
    name = getattr(fn, "__name__", str(fn))
    while worker.is_alive():
        elapsed = time.time() - t0
        if elapsed >= timeout:
            break
        worker.join(min(HEARTBEAT, timeout - elapsed))
        if worker.is_alive():
            print(f"[eval]   ... {name} still running, {time.time() - t0:.0f}s of "
                  f"{timeout:.0f}s", flush=True)
    if worker.is_alive():
        raise TimeoutError(f"{name} exceeded {timeout:.0f}s")
    if error:
        raise error[0]
    return out[0], time.time() - t0


def is_main_guard(node):
    test = node.test
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__"
                    for c in test.comparators))


class ImportScan(ast.NodeVisitor):
    def __init__(self):
        self.modules = []

    def visit_If(self, node):
        if not is_main_guard(node):
            self.generic_visit(node)

    def visit_Import(self, node):
        self.modules += [a.name.split(".")[0] for a in node.names]

    def visit_ImportFrom(self, node):
        if node.module:
            self.modules.append(node.module.split(".")[0])


def entry_kept(name):
    return (name in HYDRO_LOCATOR_FIELDS or name.endswith("ID")
            or any(word in name for word in HYDRO_BARYON_WORDS))


def without_dark_matter(name, values):
    if name.endswith("Type"):
        values = np.array(values, dtype=np.float64)
        if values.ndim >= 2 and values.shape[-1] > max(DM_TYPES):
            values[..., DM_TYPES] = np.nan
            return values
        return np.full(values.shape, np.nan)
    if entry_kept(name):
        return values
    return np.full(np.shape(values), np.nan)


def table_without_dark_matter(table):
    columns = {name: without_dark_matter(name, table[name]) for name in table.dtype.names}
    out = np.empty(table.shape, dtype=[(name, col.dtype, col.shape[1:])
                                       for name, col in columns.items()])
    for name, col in columns.items():
        out[name] = col
    return out


def copy_without_dark_matter(src, dst, particles=False):
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            if name not in DM_PARTICLE_GROUPS:
                copy_without_dark_matter(item, dst.create_group(name),
                                         particles or name in BARYON_PARTICLE_GROUPS)
        elif particles:
            if name not in WITHHELD_PARTICLE_FIELDS:
                src.copy(name, dst)
        elif item.dtype.names:
            dst.create_dataset(name, data=table_without_dark_matter(item[...]))
        else:
            dst.create_dataset(name, data=without_dark_matter(name, item[...]))


def write_served_copy(src_path, dst_path):
    if os.path.exists(dst_path) and os.path.getmtime(dst_path) >= os.path.getmtime(src_path):
        return dst_path
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = f"{dst_path}.part.{os.getpid()}.{threading.get_ident()}"
    try:
        with h5py.File(src_path, "r") as src, h5py.File(tmp, "w") as dst:
            copy_without_dark_matter(src, dst)
        os.replace(tmp, dst_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst_path


def read_host_particles(bb, relpath, host, ptypes):
    size = remote_size(bb, relpath)
    served, direct = {}, {}
    with h5py.File(RangeFile(bb, relpath, size, block=1 << 22), "r") as f:
        header = dict(f["Header"].attrs)
        for ptype in ptypes:
            group = f.get(f"PartType{ptype}")
            start, count = int(host["type_start"][ptype]), int(host["type_count"][ptype])
            if group is None or count <= 0:
                continue
            served[ptype] = {}
            for name, ds in group.items():
                if name in WITHHELD_PARTICLE_FIELDS:
                    continue
                offset = ds.id.get_offset()
                if ds.chunks is None and ds.compression is None and offset is not None:
                    direct[(ptype, name)] = {"offset": int(offset), "dtype": ds.dtype,
                                             "shape": ds.shape}
                    served[ptype][name] = None
                else:
                    served[ptype][name] = ds[start:start + count]
    for (ptype, name), field in direct.items():
        served[ptype][name] = read_rows(bb, relpath, field,
                                        int(host["type_start"][ptype]),
                                        int(host["type_count"][ptype]))
    return header, served


def write_served_snapshot(bb, relpath, dst_path, host, ptypes):
    if os.path.exists(dst_path):
        return dst_path
    header, served = read_host_particles(bb, relpath, host, ptypes)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = f"{dst_path}.part.{os.getpid()}.{threading.get_ident()}"
    counts = [0] * 6
    try:
        with h5py.File(tmp, "w") as dst:
            head = dst.create_group("Header")
            for key, value in header.items():
                head.attrs[key] = value
            for ptype, fields in served.items():
                group = dst.create_group(f"PartType{ptype}")
                for name, values in fields.items():
                    group.create_dataset(name, data=values)
                counts[ptype] = int(host["type_count"][ptype])
            head.attrs["NumPart_ThisFile"] = np.array(counts, dtype=np.int64)
            head.attrs["ServedHostOnly"] = True
        os.replace(tmp, dst_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst_path


def load_archive_commands(bb):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "skills", "scripts", "bb_server.py")
    spec = importlib.util.spec_from_file_location("milky_way_bb_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bb = bb
    return module


class EvalArchive:
    def __init__(self, bb, hosts):
        self.bb = bb
        self.hosts = hosts
        self.commands = load_archive_commands(bb)
        self.results = {}
        self.locks = {}
        self.guard = threading.Lock()

    def cached(self, key, build):
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self.results:
                self.results[key] = build()
            return dict(self.results[key])

    def served(self, path):
        rel = os.path.relpath(path, self.bb.data_dir)
        return write_served_copy(path, os.path.join(self.bb.data_dir, SERVED_DIR, rel))

    def read(self, command, kind, **params):
        def build():
            result = dict(command(kind=kind, **params))
            if kind == "hydro":
                result["path"] = self.served(result["path"])
                if "paths" in result:
                    result["paths"] = [self.served(p) for p in result["paths"]]
            return result

        return self.cached((command.__name__, kind, tuple(sorted(params.items()))), build)

    def host_of(self, sim_name, kind):
        host = self.hosts.get(str(sim_name), {}).get(kind)
        if host is None:
            raise ValueError(f"{sim_name} is not one of the simulations being scored")
        return host

    def list_simulations(self):
        return self.bb.list_simulations()

    def get_params(self):
        return self.bb.get_params()

    def get_group_catalog(self, sim_name, snapshot=90, kind="hydro"):
        return self.read(self.commands.get_group_catalog, kind,
                         sim_name=sim_name, snapshot=snapshot)

    def get_snapshot(self, sim_name, snapshot=90, kind="hydro"):
        if int(snapshot) != SNAPSHOT:
            raise ValueError(
                f"only the z=0 snapshot ({SNAPSHOT}) is served as particles; output "
                f"{snapshot} would need the host of that output, which this evaluation "
                "does not identify. Its catalog and the merger tree are served")
        if kind not in SERVED_PARTICLE_TYPES:
            raise ValueError(f"unknown kind {kind!r}, available: "
                             f"{sorted(SERVED_PARTICLE_TYPES)}")
        host = self.host_of(sim_name, kind)

        def build():
            _, snap_rel = suite_relpaths(self.bb, str(sim_name), kind)
            dst = os.path.join(self.bb.data_dir, SERVED_DIR, snap_rel.lstrip("/"))
            path = write_served_snapshot(self.bb, snap_rel, dst, host,
                                         SERVED_PARTICLE_TYPES[kind])
            with h5py.File(path, "r") as f:
                header = dict(f["Header"].attrs)
                fields = {p: sorted(f[p].keys()) for p in sorted(f)
                          if p.startswith("PartType")}
            return {"path": path,
                    "redshift": float(header["Redshift"]),
                    "box_size_ckpc_h": float(header["BoxSize"]),
                    "n_particles_by_type": [int(n) for n in header["NumPart_ThisFile"]],
                    "mass_table_1e10Msun_h": [float(m) for m in header["MassTable"]],
                    "particle_fields": fields}

        return self.cached(("get_snapshot", kind, str(sim_name), int(snapshot)), build)

    def get_merger_tree(self, sim_name, kind="hydro", extended=False):
        return self.read(self.commands.get_merger_tree, kind,
                         sim_name=sim_name, extended=extended)


def takes_archive(module):
    return len(inspect.signature(module.predict_profile).parameters) >= 3


def load_mechanism(mechanism_path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("agent_mechanism", mechanism_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blocking, problems = [], []
    for name in INTERFACE:
        if not hasattr(module, name):
            blocking.append(f"missing {name}")
    if hasattr(module, "COEFFS") and len(list(module.COEFFS)) > MAX_COEFFS:
        problems.append(f"COEFFS has {len(list(module.COEFFS))} entries, "
                        f"limit {MAX_COEFFS}")
    with open(mechanism_path) as f:
        scan = ImportScan()
        scan.visit(ast.parse(f.read()))
    bad = sorted(set(scan.modules) - set(ALLOWED_IMPORTS) - set(sys.stdlib_module_names))
    if bad:
        problems.append(f"imports outside numpy, scipy, h5py and the standard library "
                        f"that the model can reach: {bad}")
    return module, blocking, problems


def suite_relpaths(bb, sim_name, kind):
    suite = bb.suite_path() + ("_Nbody" if kind == "nbody" else "")
    cat = f"{suite}/{sim_name}/fof_subhalo_tab_{SNAPSHOT:03d}.hdf5"
    snap = f"/Sims{suite[len(CATALOG_ROOT):]}/{sim_name}/snap_{SNAPSHOT:03d}.hdf5"
    return cat, snap


def profile_on_grid(dist, mass, hubble, r_ref_ckpc_h):
    edges = r_ref_ckpc_h * EDGES
    shell_mass, _ = np.histogram(dist, bins=edges, weights=mass)
    shell_vol = 4.0 / 3.0 * np.pi * np.diff(edges ** 3)
    rho = shell_mass * 1e10 * hubble ** 2 / shell_vol
    rho[shell_mass == 0] = np.nan
    return rho


def read_dm(bb, relpath, start, count):
    size = remote_size(bb, relpath)
    with h5py.File(RangeFile(bb, relpath, size), "r") as f:
        header = dict(f["Header"].attrs)
        ds = f["PartType1"]["Coordinates"]
        offset = ds.id.get_offset()
        if start < 0 or start + count > ds.shape[0]:
            raise ValueError(f"rows [{start}, {start + count}) outside {ds.shape}")
        if ds.chunks is None and ds.compression is None and offset is not None:
            field = {"offset": int(offset), "dtype": ds.dtype, "shape": ds.shape}
        else:
            field = None
    if field is not None:
        pos = read_rows(bb, relpath, field, start, count)
    else:
        with h5py.File(RangeFile(bb, relpath, size, block=1 << 22), "r") as f:
            pos = f["PartType1"]["Coordinates"][start:start + count]
    mass1 = float(header["MassTable"][1])
    if mass1 <= 0:
        raise RuntimeError(f"{relpath}: PartType1 has no MassTable mass")
    return pos.astype(np.float64), np.full(count, mass1)


def host_dm(bb, sim_name, kind):
    cat_rel, snap_rel = suite_relpaths(bb, sim_name, kind)
    if kind == "hydro":
        cat_paths = bb.catalog_paths(sim_name, SNAPSHOT)
    else:
        cat_paths = [bb.fetch(cat_rel)]
    host = find_host(cat_paths)
    if host is None:
        raise RuntimeError(f"no clean Milky-Way-mass host in the {kind} catalog")
    pos, mass = read_dm(bb, snap_rel, host["type_start"][1], host["type_count"][1])
    dist = host_distances(pos, host)
    return host, dist, mass


def measure_box(bb, sim_name):
    hydro, dist_h, mass_h = host_dm(bb, sim_name, "hydro")
    twin, dist_n, mass_n = host_dm(bb, sim_name, "nbody")
    if abs(np.log10(hydro["m200_msun"] / twin["m200_msun"])) > MAX_HOST_MATCH_DEX:
        raise RuntimeError(
            f"hydro and twin hosts do not match: M200c {hydro['m200_msun']:.3e} "
            f"vs {twin['m200_msun']:.3e} Msun")
    dm_share = (twin["omega_m"] - OMEGA_B) / twin["omega_m"]
    rho_nbody = profile_on_grid(dist_n, mass_n, twin["hubble"],
                                twin["r200_ckpc_h"]) * dm_share
    rho_hydro = profile_on_grid(dist_h, mass_h, hydro["hubble"],
                                twin["r200_ckpc_h"])
    if not (np.isfinite(rho_nbody).all() and np.isfinite(rho_hydro).all()):
        raise RuntimeError("empty shells in the measured profiles")
    return {"rho_nbody": rho_nbody,
            "log_rho": np.log10(rho_hydro),
            "r200_kpc": twin["r200_kpc"],
            "m200_msun": twin["m200_msun"],
            "host_group_hydro": hydro["index"],
            "host_group_nbody": twin["index"],
            "hosts": {"hydro": hydro, "nbody": twin}}


def measure_box_with_retry(bb, sim_name):
    try:
        return measure_box(bb, sim_name)
    except Exception:
        time.sleep(2.0)
        return measure_box(bb, sim_name)


PARAM_KEYS = {"omega_m": "Om", "sigma_8": "s8",
              "a_sn1": "SN1", "a_sn2": "SN2", "a_agn": "BHFF"}


def build_props(rows, boxes, data, with_archive):
    props = {"r_over_r200": CENTERS.copy(),
             "rho_nbody": np.vstack([data[b]["rho_nbody"] for b in boxes]),
             "r200_nbody": np.array([data[b]["r200_kpc"] for b in boxes])}
    if with_archive:
        props["sim_name"] = np.array(boxes)
        for key in ("host_group_hydro", "host_group_nbody"):
            props[key] = np.array([data[b][key] for b in boxes], dtype=np.int64)
        return props
    props["m200_nbody"] = np.array([data[b]["m200_msun"] for b in boxes])
    for key, col in PARAM_KEYS.items():
        props[key] = np.array([rows[trailing_int(b)][col] for b in boxes])
    return props


def subset_props(props, sel):
    out = {"r_over_r200": props["r_over_r200"]}
    for key, value in props.items():
        if key != "r_over_r200":
            out[key] = value[sel]
    return out


def compute_metrics(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
        return {"r2_full": None, "rmse_full": None,
                "r2_inner": None, "rmse_inner": None,
                "error": "predictions are not a finite (n_hosts, n_radii) array"}
    inner = CENTERS <= INNER_R_OVER_R200

    def block(p, t):
        return (float(1 - np.mean((p - t) ** 2) / np.var(t)),
                float(np.sqrt(np.mean((p - t) ** 2))))

    r2_full, rmse_full = block(pred, truth)
    r2_inner, rmse_inner = block(pred[:, inner], truth[:, inner])
    return {"r2_full": r2_full, "rmse_full": rmse_full,
            "r2_inner": r2_inner, "rmse_inner": rmse_inner}


def per_host_check(mech, props, coeffs, pred, extra, n_probe=32, seed=0, tolerance=1e-9):
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    probe = rng.choice(n, size=min(n_probe, n), replace=False)
    worst = 0.0
    for i in probe:
        alone = np.asarray(mech.predict_profile(subset_props(props, slice(i, i + 1)),
                                                coeffs, *extra), dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(alone[0] - pred[i]))))
    return {"n_probed": int(len(probe)), "max_difference": worst,
            "passed": bool(worst <= tolerance)}


def rmse_score(metrics, baseline):
    rmse, reference = metrics.get("rmse_full"), baseline.get("rmse_full")
    if rmse is None or reference is None or not reference > 0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - rmse / reference)))


def held_out_file(output):
    tests = Path(__file__).resolve().parent
    boxes = set()
    for name in ("holdout_union.json", "accessed_sims.json"):
        if (tests / name).is_file():
            with open(tests / name) as f:
                boxes.update(json.load(f).get("boxes", []))
    path = Path(output).with_name("held_out_accessed.json")
    with open(path, "w") as f:
        json.dump({"n_accessed": len(boxes), "boxes": sorted(boxes, key=lambda n: trailing_int(n) or 0)}, f, indent=2)
    return str(path)


def main(mechanism, output, data_dir=None, n_sims=100, seed=0, workers=6, accessed_file=None,
         predict_timeout=None, fit_timeout=None):
    global PREDICT_TIMEOUT, FIT_TIMEOUT
    if predict_timeout is not None:
        PREDICT_TIMEOUT = float(predict_timeout)
    if fit_timeout is not None:
        FIT_TIMEOUT = float(fit_timeout)
    print(f"[eval] budgets: predict {PREDICT_TIMEOUT:.0f}s, fit {FIT_TIMEOUT:.0f}s", flush=True)
    mechanism_path = str(mechanism)
    if accessed_file is None:
        accessed_file = held_out_file(output)
    report = {
        "mechanism_path": mechanism_path,
        "protocol": {
            "held_out_set": "every simulation of the DREAMS CDM Milky-Way zoom-in "
                            "suite whose data the run never downloaded, read from "
                            "accessed_sims.json recorded during the run",
            "host_definition": HOST_DEFINITION,
            "radius_grid": f"{N_PROFILE_BINS} logarithmic shells over r/R200c in "
                           f"[{R_OVER_R200_LIMITS[0]:g}, {R_OVER_R200_LIMITS[1]:g}], "
                           "anchored to each twin's R200c for both the twin input "
                           "and the hydro truth",
            "truth": "log10 of the spherically averaged dark matter density of the "
                     "hydro host over its group's PartType1 particles, in physical "
                     "Msun/kpc^3",
            "twin_input": "the same measurement on the gravity-only twin, rescaled "
                          "by (Omega_m - Omega_b) / Omega_m to the dark-matter share",
            "archive": "the submission reads what it decides to through the commands "
                       "of bb_cli.py, handed to it as the archive argument. A catalog "
                       "or tree of a gravity-only twin is served as stored; the same "
                       "file of a hydro run is served as a copy without its dark "
                       f"matter: no {list(DM_PARTICLE_GROUPS)} groups, no "
                       f"{list(WITHHELD_PARTICLE_FIELDS)} of the particles, fields "
                       f"ending in Type with the dark matter columns {DM_TYPES} set to "
                       f"NaN, fields named after {list(HYDRO_BARYON_WORDS)}, fields "
                       f"ending in ID and {list(HYDRO_LOCATOR_FIELDS)} as stored, and "
                       "every other catalog or tree entry as NaN. A snapshot is served "
                       f"as the host's own particles, {SERVED_PARTICLE_TYPES}, read "
                       "with targeted Range requests, and only at z=0",
            "host_matching": f"hydro and twin hosts are identified independently by "
                             f"the same rule and must agree in M200c to within "
                             f"{MAX_HOST_MATCH_DEX} dex",
            "metrics": "r2 and rmse of log10 rho_DM pooled over hosts and radii, "
                       "over the full grid and over the inner halo "
                       f"(r/R200c <= {INNER_R_OVER_R200:g}) separately",
            "passes": "once with the submitted COEFFS on the full held-out set, "
                      "once with constants refit by the submission's own fit_coeffs "
                      "on a random half of the held-out set (functional form "
                      "frozen) and scored on the disjoint other half",
            "scores": "PA1 (submitted constants) and PA2 (refit pass) are each 1 minus "
                      "rmse_full over the rmse_full of the twin passthrough, which "
                      "predicts the twin's own profile on exactly the same hosts and "
                      "radii, floored at 0; a prediction that is not finite scores 0. "
                      "r2 and the inner-halo metrics are reported as context only",
            "seed": seed,
        },
        "pa_scores": {pid: 0.0 for pid in PA_IDS},
        "predictive_accuracy": 0.0,
    }

    mech = None
    if not os.path.exists(mechanism_path):
        report["error"] = "mechanism.py not found"
    else:
        try:
            mech, blocking, problems = load_mechanism(mechanism_path)
        except Exception:
            mech, blocking, problems = None, ["import failed:\n" + traceback.format_exc()], []
        report["interface_problems"] = problems
        report["blocking_problems"] = blocking
        for problem in problems:
            print(f"[eval] contract violation, reported and not scored here: {problem}",
                  flush=True)
        if mech is None or blocking:
            mech = None
            report["error"] = "mechanism.py cannot be evaluated: " + "; ".join(blocking)
    if report.get("error"):
        print(f"[eval] ERROR: {report['error']}", flush=True)

    with_archive = mech is not None and takes_archive(mech)
    if mech is not None:
        report["interface"] = "archive" if with_archive else "earlier"
        if not with_archive:
            print("[eval] predict_profile takes no archive: evaluated under the earlier "
                  "task statement, with the five parameters and the twin's M200c in "
                  "props", flush=True)

    if mech is not None:
        if str(accessed_file).strip().lower() in ("none", "ignore"):
            accessed = set()
            report["exclusion_waived"] = True
            report["protocol"]["exclusion_note"] = (
                "access-based exclusion waived by the operator: the run accessed "
                "the whole suite, so the metrics are computed over simulations the "
                "agent saw during its investigation and the in-suite pass is not "
                "blind; the refit pass still scores on a half disjoint from the "
                "fitting half")
            print("[eval] WARNING: access-based exclusion waived; the metrics are "
                  "not blind for this run")
        else:
            if not os.path.isfile(accessed_file):
                raise RuntimeError(
                    f"{accessed_file} not found; the held-out set is the complement "
                    "of the simulations the run accessed, so the evaluation cannot "
                    "proceed without it")
            with open(accessed_file) as f:
                accessed = set(json.load(f).get("boxes", []))
            print(f"[eval] excluding {len(accessed)} simulations the run accessed")

        bb = DreamsBlackBox(data_dir=data_dir)
        rng = np.random.default_rng(seed)
        info = bb.list_simulations()
        columns, rows = read_param_table(bb.params_file())
        if len(rows) != info["n_simulations"]:
            raise RuntimeError(f"parameter table has {len(rows)} rows for "
                               f"{info['n_simulations']} simulations")

        eligible = [n for n in info["simulations"]
                    if n not in accessed and trailing_int(n) in rows]
        if n_sims and n_sims < len(eligible):
            print(f"[eval] WARNING: sampling {n_sims} of {len(eligible)} eligible "
                  f"simulations instead of the full complement")
            eligible = sorted(rng.choice(eligible, size=n_sims, replace=False),
                              key=trailing_int)
        report["n_excluded_accessed"] = len(accessed)
        report["n_eval_sims_requested"] = len(eligible)

        data, skipped = {}, []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(measure_box_with_retry, bb, name): name
                       for name in eligible}
            for done, future in enumerate(as_completed(futures), start=1):
                name = futures[future]
                try:
                    data[name] = future.result()
                except Exception as exc:
                    print(f"  [skip] {name}: {exc}", flush=True)
                    skipped.append(name)
                if done % 25 == 0 or done == len(futures):
                    progress("measure", done, len(futures), t0,
                             f"{len(data)} usable")
        report["n_measure_skipped"] = len(skipped)

        if len(data) < MIN_EVAL_SIMS:
            report["error"] = f"only {len(data)} usable held-out simulations"
            print(f"[eval] ERROR: {report['error']}", flush=True)
        else:
            boxes = sorted(data, key=trailing_int)
            props = build_props(rows, boxes, data, with_archive)
            truth = np.vstack([data[b]["log_rho"] for b in boxes])
            report["n_eval_sims_used"] = int(len(boxes))
            half = len(boxes) // 2
            extra = ((EvalArchive(bb, {b: data[b]["hosts"] for b in boxes}),)
                     if with_archive else ())
            predict_timeout, fit_timeout = PREDICT_TIMEOUT, FIT_TIMEOUT

            baseline = compute_metrics(np.log10(props["rho_nbody"]), truth)
            report["baseline_twin_passthrough"] = baseline
            print(f"[eval] twin-passthrough baseline: "
                  f"rmse_full={baseline['rmse_full']:.4f} "
                  f"rmse_inner={baseline['rmse_inner']:.4f} dex", flush=True)

            try:
                pred, seconds = call_with_timeout(mech.predict_profile, predict_timeout,
                                                  props, list(mech.COEFFS), *extra)
                pred = np.asarray(pred, dtype=np.float64)
                in_suite = compute_metrics(pred, truth)
                in_suite["predict_seconds"] = round(seconds, 3)
                if in_suite.get("rmse_full") is not None:
                    in_suite["per_host"] = per_host_check(
                        mech, props, list(mech.COEFFS), pred, extra, seed=seed)
                    if not in_suite["per_host"]["passed"]:
                        print("  [note] predict_profile does not predict one host "
                              "at a time: a host's profile changes by up to "
                              f"{in_suite['per_host']['max_difference']:.3g} when it "
                              "is evaluated on its own; recorded for the judge, the "
                              "metrics are still computed", flush=True)
            except Exception as exc:
                in_suite = {"r2_full": None, "rmse_full": None,
                            "r2_inner": None, "rmse_inner": None, "error": str(exc)}
            report["in_suite"] = in_suite
            print(f"[eval] in-suite ({len(boxes)} sims, submitted constants): "
                  f"r2_full={in_suite['r2_full']} rmse_full={in_suite['rmse_full']} "
                  f"r2_inner={in_suite['r2_inner']} rmse_inner={in_suite['rmse_inner']}",
                  flush=True)

            fit_sel = np.zeros(len(boxes), dtype=bool)
            fit_sel[rng.permutation(len(boxes))[:half]] = True
            refit = {"n_fit": int(fit_sel.sum()), "n_test": int((~fit_sel).sum())}
            baseline_refit = compute_metrics(
                np.log10(props["rho_nbody"][~fit_sel]), truth[~fit_sel])
            report["baseline_twin_passthrough_refit_half"] = baseline_refit
            try:
                coeffs2, seconds = call_with_timeout(
                    mech.fit_coeffs, fit_timeout,
                    subset_props(props, fit_sel), truth[fit_sel], list(mech.COEFFS),
                    *extra)
                coeffs2 = [float(c) for c in coeffs2]
                if len(coeffs2) > MAX_COEFFS:
                    refit["contract_violation"] = (f"fit_coeffs returned {len(coeffs2)} "
                                                   f"constants, limit {MAX_COEFFS}")
                    print(f"[eval] contract violation, recorded and scored anyway: "
                          f"{refit['contract_violation']}", flush=True)
                refit["fit_seconds"] = round(seconds, 3)
                refit["coeffs"] = [round(c, 8) for c in coeffs2]
                pred2, seconds = call_with_timeout(
                    mech.predict_profile, predict_timeout,
                    subset_props(props, ~fit_sel), coeffs2, *extra)
                refit.update(compute_metrics(np.asarray(pred2, np.float64),
                                             truth[~fit_sel]))
                refit["predict_seconds"] = round(seconds, 3)
            except Exception as exc:
                refit.update({"r2_full": None, "rmse_full": None,
                              "r2_inner": None, "rmse_inner": None, "error": str(exc)})
            report["refit"] = refit
            print(f"[eval] refit ({refit['n_fit']} fit / {refit['n_test']} test sims): "
                  f"r2_full={refit['r2_full']} rmse_full={refit['rmse_full']} "
                  f"r2_inner={refit['r2_inner']} rmse_inner={refit['rmse_inner']}",
                  flush=True)

            report["pa_scores"] = {
                "PA1": round(rmse_score(in_suite, baseline), 4),
                "PA2": round(rmse_score(refit, baseline_refit), 4),
            }

    report["predictive_accuracy"] = round(
        sum(report["pa_scores"][p] for p in PA_IDS) / len(PA_IDS), 4)
    report["predictive_accuracy_total"] = len(PA_IDS)

    out_path = str(output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("=" * 64)
    print(f"[eval] predictive accuracy, each 1 - rmse / rmse of the twin passthrough, "
          f"floored at 0: {json.dumps(report['pa_scores'])} -> "
          f"{report['predictive_accuracy']}")
    if report.get("interface_problems"):
        print(f"[eval] contract violations recorded for the judge: "
              f"{report['interface_problems']}")
    if report.get("error"):
        print(f"[eval] evaluation incomplete: {report['error']}")
    print(f"[eval] results: {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    fire.Fire(main)
