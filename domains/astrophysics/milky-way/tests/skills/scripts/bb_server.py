import json
import os
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.environ.get("ASTRO_SRC_ROOT", "/opt/astrophysics"))

import h5py

from archive import CATALOG_ROOT, DreamsBlackBox


bb = None


def get_bb():
    global bb
    if bb is None:
        bb = DreamsBlackBox(data_dir=os.environ.get("ASTRO_DATA_DIR"))
    return bb


KIND_SUFFIX = {"hydro": "", "nbody": "_Nbody"}


def suite_for(kind):
    if kind not in KIND_SUFFIX:
        raise ValueError(f"unknown kind {kind!r}, available: {sorted(KIND_SUFFIX)}")
    return get_bb().suite_path() + KIND_SUFFIX[kind]


def get_group_catalog(sim_name, snapshot=90, kind="hydro"):
    if suite_for(kind) == get_bb().suite_path():
        return get_bb().get_group_catalog(sim_name, snapshot)
    path = get_bb().fetch(
        f"{suite_for(kind)}/{sim_name}/fof_subhalo_tab_{int(snapshot):03d}.hdf5")
    with h5py.File(path, "r") as f:
        header = dict(f["Header"].attrs)
        group_fields = sorted(f["Group"].keys()) if "Group" in f else []
        subhalo_fields = sorted(f["Subhalo"].keys()) if "Subhalo" in f else []
    n_subhalos = header.get("Nsubgroups_Total", header.get("Nsubhalos_Total", -1))
    return {"path": path, "paths": [path], "n_files": 1,
            "redshift": float(header["Redshift"]),
            "box_size_ckpc_h": float(header["BoxSize"]),
            "n_groups": int(header.get("Ngroups_Total", -1)),
            "n_subhalos": int(n_subhalos),
            "group_fields": group_fields,
            "subhalo_fields": subhalo_fields}


def get_snapshot(sim_name, snapshot=90, kind="hydro"):
    tail = suite_for(kind)[len(CATALOG_ROOT):]
    path = get_bb().fetch(f"/Sims{tail}/{sim_name}/snap_{int(snapshot):03d}.hdf5")
    with h5py.File(path, "r") as f:
        header = dict(f["Header"].attrs)
        part_types = sorted(k for k in f.keys() if k.startswith("PartType"))
        fields = {p: sorted(f[p].keys()) for p in part_types}
    return {"path": path,
            "redshift": float(header["Redshift"]),
            "box_size_ckpc_h": float(header["BoxSize"]),
            "n_particles_by_type": [int(n) for n in header["NumPart_Total"]],
            "mass_table_1e10Msun_h": [float(m) for m in header["MassTable"]],
            "particle_fields": fields}


def get_merger_tree(sim_name, kind="hydro", extended=False):
    name = "tree_extended.hdf5" if extended else "tree.hdf5"
    path = get_bb().fetch(f"{suite_for(kind)}/{sim_name}/{name}")
    with h5py.File(path, "r") as f:
        content = {}
        for key in sorted(f.keys()):
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                content[key] = {"shape": list(obj.shape),
                                "fields": sorted(obj.dtype.names)
                                if obj.dtype.names else str(obj.dtype)}
            else:
                content[key] = {"group": sorted(obj.keys())}
    return {"path": path, "content": content}


LOCAL_METHODS = {"get_group_catalog": get_group_catalog,
                 "get_snapshot": get_snapshot,
                 "get_merger_tree": get_merger_tree}

ALLOWED_METHODS = {"list_simulations", "get_params", "get_group_catalog",
                   "get_snapshot", "get_merger_tree"}


def map_paths(value, src, dst):
    if isinstance(value, str):
        return value.replace(src, dst)
    if isinstance(value, dict):
        return {k: map_paths(v, src, dst) for k, v in value.items()}
    if isinstance(value, list):
        return [map_paths(v, src, dst) for v in value]
    return value


def call_method(method, params):
    if method == "__ping__":
        return "pong"
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method in LOCAL_METHODS:
        result = LOCAL_METHODS[method](**params)
    else:
        result = getattr(get_bb(), method)(**params)
    prefix = os.environ.get("ASTRO_REPORT_PREFIX")
    if prefix:
        result = map_paths(result, get_bb().data_dir, prefix)
    return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        method = body["method"]
        params = body.get("params", {})
        try:
            result = call_method(method, params)
            resp = json.dumps({"success": True, "result": result}, default=str)
        except Exception as e:
            traceback.print_exc()
            error = str(e)
            prefix = os.environ.get("ASTRO_REPORT_PREFIX")
            if prefix and bb is not None:
                error = error.replace(bb.data_dir, prefix)
            resp = json.dumps({"success": False, "error": error}, default=str)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.encode())

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("BB_SERVER_PORT", sys.argv[1] if len(sys.argv) > 1 else "9100"))
    print(f"[bb_server] Starting on port {port}", flush=True)
    get_bb()
    print(f"[bb_server] DREAMS blackbox ready, data_dir={bb.data_dir}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
