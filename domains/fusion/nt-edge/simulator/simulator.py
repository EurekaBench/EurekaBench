import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# the whole TORAX configuration surface; nothing here is narrowed for the caller
CONFIG_SECTIONS = ("profile_conditions", "numerics", "plasma_composition", "geometry",
                   "sources", "neoclassical", "solver", "transport", "pedestal",
                   "mhd", "edge", "time_step_calculator", "restart")

MAX_T_FINAL = 100.0
MAX_N_RHO = 200
MAX_STEPS = 200000
DEFAULT_TIMEOUT_S = 3600.0

COMMANDS = ("capabilities", "config_schema", "simulate", "read", "equilibrium", "ballooning",
            "run")
GPEC_DRIVER = Path(__file__).resolve().parent / "gpec_ballooning.jl"
GPEC = dict(n_scan=32, mpsi=48, mtheta=256, max_alpha_scale=8.0)
N_PSI = 201
EQUILIBRIUM_REQUIRED = ("R0", "a", "kappa", "delta", "Ip", "B0", "p_axis")
EQUILIBRIUM_KEYS = EQUILIBRIUM_REQUIRED + (
    "Z0", "kappa_lower", "delta_lower", "x_point", "machine_scale", "pp", "ffp", "kinetic",
    "n_boundary", "n_psi", "maxits", "urf", "nl_tol", "coil_weight", "coil_bound_factor",
    "isoflux_weight", "coil_targets", "coil_vsc")
DEFAULT_URF = 0.2
DEFAULT_NL_TOL = 1e-6
# pressure of one species per eV of temperature, Pa per (m^-3 eV)
E_KEV_PA = 1.602176634e-19
DIIID_COIL_TARGETS = {
    "ECOILA": -16030.961914062493, "ECOILB": -15782.150390625,
    "F1A": 1999.5169719827586, "F2A": 1031.1705953663793, "F3A": -530.3450296336207,
    "F4A": -691.5127963362069, "F5A": 12.465672986260776, "F6A": -2142.877414772727,
    "F7A": 620.4805397727273, "F8A": -975.8820716594828, "F9A": 4302.279545454546,
    "F1B": 2213.2513469827586, "F2B": 1316.7264278017242, "F3B": -385.2729660560345,
    "F4B": -2295.533943965517, "F5B": 6.891535265692344, "F6B": -2792.180113636364,
    "F7B": 754.6947443181818, "F8B": -886.030239762931, "F9B": 4588.732102272727,
}


def coil_weight_of(name):
    if name.startswith("ECOIL"):
        return 61.0
    if name.startswith("F5"):
        return 100.0
    return 1.0


def check_name(output_name):
    name = str(output_name)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("output_name may only contain letters, digits, '.', '_', '-'")
    return name


def jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)


def coerce(value, name, kind):
    if value is None or isinstance(value, kind):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON: {exc}") from exc
        if isinstance(parsed, kind):
            return parsed
    raise ValueError(f"{name} must be a JSON {kind.__name__}")


class ToraxSimulator:
    def __init__(self, export_dir=".", export_prefix=None, qlk_exec_path=None):
        import torax

        self.torax = torax
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_prefix = export_prefix or str(self.export_dir)
        if qlk_exec_path:
            os.environ["TORAX_QLK_EXEC_PATH"] = str(qlk_exec_path)
        self.registered = False
        self.registration = {}

    def check_name(self, output_name):
        return check_name(output_name)

    def register_external_models(self):
        # qualikiz is not in the default config union and its import needs qualikiz_tools
        if self.registered:
            return
        self.registered = True
        try:
            import torax.transport
            from torax._src.transport_model import qualikiz_transport_model as qlk

            torax.transport.register_transport_model(qlk.QualikizTransportModelConfig)
            self.registration = {"qualikiz": "available"}
        except Exception as exc:
            self.registration = {"qualikiz": f"unavailable: {type(exc).__name__}: {exc}"}

    def check_config(self, config):
        unknown = sorted(set(config) - set(CONFIG_SECTIONS))
        if unknown:
            raise ValueError(f"unknown configuration sections: {unknown}; "
                             f"TORAX accepts {list(CONFIG_SECTIONS)}")
        numerics = config.get("numerics") or {}
        t_final = float(numerics.get("t_final", 5.0))
        if not (0 < t_final <= MAX_T_FINAL):
            raise ValueError(f"numerics.t_final must be in (0, {MAX_T_FINAL:g}]")
        n_rho = (config.get("geometry") or {}).get("n_rho")
        if n_rho is not None and not (1 <= int(n_rho) <= MAX_N_RHO):
            raise ValueError(f"geometry.n_rho must be in [1, {MAX_N_RHO}]")
        return config

    def capabilities(self, output_name="capabilities"):
        from typing import get_args
        from torax._src.torax_pydantic import model_config

        self.register_external_models()
        name = self.check_name(output_name)

        def names_of(annotation):
            out = []
            for model in get_args(annotation):
                field = model.model_fields.get("model_name")
                default = getattr(field, "default", None) if field else None
                out.append(default if isinstance(default, str) else model.__name__)
            return out

        fields = model_config.ToraxConfig.model_fields
        payload = {"torax_version": self.torax.version.TORAX_VERSION,
                   "jax_backend": self.torax.jax.default_backend(),
                   "jax_devices": [str(d) for d in self.torax.jax.devices()],
                   "config_sections": list(CONFIG_SECTIONS),
                   "transport_models": names_of(fields["transport"].annotation),
                   "external_models": dict(self.registration),
                   "limits": {"t_final": MAX_T_FINAL, "n_rho": MAX_N_RHO,
                              "steps": MAX_STEPS, "timeout_s": DEFAULT_TIMEOUT_S}}
        for section in ("pedestal", "solver", "time_step_calculator", "geometry"):
            try:
                payload[f"{section}_options"] = names_of(fields[section].annotation)
            except Exception:
                payload[f"{section}_options"] = None
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=1, default=str)
        payload["file"] = f"{self.export_prefix}/{name}.json"
        return payload

    def config_schema(self, section, output_name="schema"):
        from torax._src.torax_pydantic import model_config

        self.register_external_models()
        name = self.check_name(output_name)
        section = str(section)
        if section not in CONFIG_SECTIONS:
            raise ValueError(f"section must be one of {list(CONFIG_SECTIONS)}")
        annotation = model_config.ToraxConfig.model_fields[section].annotation
        from typing import get_args

        models = list(get_args(annotation)) or [annotation]
        payload = {}
        for model in models:
            fields = getattr(model, "model_fields", None)
            if fields is None:
                continue
            entry = {}
            for key, field in fields.items():
                entry[key] = {"type": str(field.annotation),
                              "default": field.default
                              if field.default is not None else None}
            payload[getattr(model, "__name__", str(model))] = entry
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=1, default=str)
        return {"file": f"{self.export_prefix}/{name}.json",
                "section": section, "variants": list(payload)}

    def simulate(self, config, output_name="run1", progress_bar=False,
                 log_timestep_info=False, max_steps=0, out_dir=None):
        self.register_external_models()
        name = self.check_name(output_name)
        config = self.check_config(coerce(config, "config", dict) or {})
        out_root = Path(out_dir) if out_dir else self.export_dir
        geometry = config.get("geometry") or {}
        candidate = out_root / Path(str(geometry.get("geometry_file") or "")).name
        if geometry.get("geometry_file") and candidate.is_file():
            geometry["geometry_file"] = candidate.name
            geometry["geometry_directory"] = str(out_root)
        cap = int(max_steps) or None
        if cap is not None and not (1 <= cap <= MAX_STEPS):
            raise ValueError(f"max_steps must be in [1, {MAX_STEPS}]")

        started = time.time()
        torax_config = self.torax.ToraxConfig.from_dict(config)
        data_tree, history = self.torax.run_simulation(
            torax_config, progress_bar=bool(progress_bar),
            log_timestep_info=bool(log_timestep_info), max_steps=cap)
        seconds = time.time() - started

        out_path = out_root / f"{name}.nc"
        data_tree.to_netcdf(out_path)
        summary = {"file": f"{self.export_prefix}/{name}.nc",
                   "sim_error": str(history.sim_error),
                   "n_times": int(data_tree.scalars.sizes["time"]),
                   "t_final_reached": float(np.asarray(data_tree.scalars.time)[-1]),
                   "seconds": round(seconds, 2),
                   "groups": {}}
        for group in ("numerics", "profiles", "scalars", "edge"):
            if group in data_tree.children:
                dataset = data_tree[group].to_dataset()
                summary["groups"][group] = {
                    "variables": sorted(dataset.data_vars),
                    "coords": {k: int(v.size) for k, v in dataset.coords.items()}}
        return summary

    def read(self, file, variables=None, time_index=-1, output_name="read1"):
        import xarray as xr

        name = self.check_name(output_name)
        # the caller names the file as it sees it, which may be a path inside a container
        path = Path(str(file))
        if not path.is_file():
            path = self.export_dir / path.name
        if not path.is_file():
            raise ValueError(f"no simulation output at {path}")
        variables = coerce(variables, "variables", list)
        tree = xr.open_datatree(path)
        wanted, payload = variables, {}
        for group in tree.children:
            dataset = tree[group].to_dataset()
            for key in dataset.data_vars:
                if wanted is not None and key not in wanted:
                    continue
                array = dataset[key]
                if "time" in array.dims and time_index is not None:
                    array = array.isel(time=int(time_index))
                payload[key] = np.asarray(array).tolist()
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as handle:
            json.dump(payload, handle, default=str)
        return {"file": f"{self.export_prefix}/{name}.json",
                "variables": sorted(payload), "time_index": time_index}


class EquilibriumSimulator:
    def __init__(self, export_dir=".", export_prefix=None, mesh_file=None, nthreads=4):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_prefix = export_prefix or str(self.export_dir)
        self.mesh_file = mesh_file or os.environ.get("TOKAMAKER_MESH", "")
        self.mesh_dir = os.environ.get("TOKAMAKER_MESH_DIR", "") or str(self.export_dir)
        self.nthreads = int(nthreads)
        self.lock = threading.Lock()
        self.env = None
        self.gs = None
        self.loaded = (None, None)
        self.coils = []

    def status(self):
        root = os.environ.get("OFT_ROOTPATH")
        if root and os.path.join(root, "python") not in sys.path:
            sys.path.append(os.path.join(root, "python"))
        try:
            import OpenFUSIONToolkit.TokaMaker
        except Exception as exc:
            return f"unavailable: {type(exc).__name__}: {exc}"
        if not (self.mesh_file and os.path.isfile(self.mesh_file)):
            return "unavailable: no TokaMaker mesh file; TOKAMAKER_MESH names one"
        return "available"

    def geometry_file(self):
        return os.path.join(os.path.dirname(self.mesh_file), "DIIID_geom.json")

    def load(self):
        status = self.status()
        if status != "available":
            raise RuntimeError(f"equilibrium is {status}")
        if self.env is None:
            from OpenFUSIONToolkit import OFT_env
            from OpenFUSIONToolkit.TokaMaker import TokaMaker

            self.env = OFT_env(nthreads=self.nthreads)
            self.gs = TokaMaker(self.env)

    def machine_mesh(self, scale):
        if abs(float(scale) - 1.0) < 1e-9:
            return self.mesh_file
        cache = Path(self.mesh_dir) / f"machine_x{float(scale):.3f}.h5"
        if cache.is_file():
            return str(cache)
        from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain, save_gs_mesh

        with open(self.geometry_file()) as handle:
            geom = json.load(handle)
        s = float(scale)
        scaled = lambda pts: (np.asarray(pts, dtype=float) * s).tolist()
        plasma_dx, coil_dx, vv_dx, vac_dx = 0.04 * s, 0.03 * s, 0.04 * s, 0.10 * s
        domain = gs_Domain()
        domain.define_region("air", vac_dx, "boundary")
        domain.define_region("plasma", plasma_dx, "plasma")
        domain.define_region("vacuum", vv_dx, "vacuum", allow_xpoints=True)
        for i, segment in enumerate(geom["vv"]):
            domain.define_region(f"vv{i}", vv_dx, "conductor", eta=segment[1])
        for key, coil in geom["coils"].items():
            if key.startswith("ECOIL"):
                for i, sub in enumerate(coil):
                    domain.define_region(f"{key}_{i}", coil_dx, "coil", coil_set=key,
                                         nTurns=sub["nturns"])
            else:
                domain.define_region(key, coil_dx, "coil", nTurns=coil["nturns"])
        domain.add_polygon(scaled(geom["limiter"]), "plasma", parent_name="vacuum")
        domain.add_enclosed([1.75 * s, 1.25 * s], "vacuum")
        for i, segment in enumerate(geom["vv"]):
            domain.add_polygon(scaled(segment[0]), f"vv{i}", parent_name="air")
        for key, coil in geom["coils"].items():
            if key.startswith("ECOIL"):
                for i, sub in enumerate(coil):
                    domain.add_polygon(scaled(sub["pts"]), f"{key}_{i}", parent_name="air")
            else:
                domain.add_polygon(scaled(coil["pts"]), key, parent_name="air")
        pts, tris, regions = domain.build_mesh()
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_gs_mesh(pts, tris, regions, domain.get_coils(), domain.get_conductors(),
                     str(cache))
        return str(cache)

    def prepare(self, mesh_file, f0):
        if self.loaded == (mesh_file, f0):
            return
        from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh

        pts, tris, regions, coil_dict, cond_dict = load_gs_mesh(mesh_file)
        self.gs.reset()
        self.gs.setup_mesh(pts, tris, regions)
        self.gs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
        self.gs.setup(order=2, F0=f0)
        # a coil is addressed by its coil set: the sub-regions of a set share one current
        self.coils = sorted({str(entry.get("coil_set", name)) for name, entry in coil_dict.items()})
        self.loaded = (mesh_file, f0)

    def check_config(self, config):
        missing = [key for key in EQUILIBRIUM_REQUIRED if key not in config]
        if missing:
            raise ValueError(f"equilibrium config lacks {missing}")
        unknown = sorted(set(config) - set(EQUILIBRIUM_KEYS))
        if unknown:
            raise ValueError(f"unknown equilibrium keys: {unknown}; accepted "
                             f"{list(EQUILIBRIUM_KEYS)}")
        if str(config.get("x_point", "lower")) not in ("lower", "none"):
            raise ValueError("x_point is 'lower' or 'none'")
        if "kinetic" in config and not all(k in config["kinetic"] for k in ("n_e", "T_e")):
            raise ValueError("kinetic profiles need n_e and T_e")
        return config

    def boundary(self, config):
        from OpenFUSIONToolkit.TokaMaker.util import (create_isoflux, create_isoflux_xpts,
                                                      xpoints_from_moments)

        r0, z0, a = float(config["R0"]), float(config.get("Z0", 0.0)), float(config["a"])
        kappa, delta = float(config["kappa"]), float(config["delta"])
        kappa_lower = float(config.get("kappa_lower", kappa))
        delta_lower = float(config.get("delta_lower", delta))
        npts = int(config.get("n_boundary", 80))
        smooth = create_isoflux(npts, r0, z0, a, kappa, delta, kappaL=kappa_lower,
                                deltaL=delta_lower)
        if str(config.get("x_point", "lower")) == "none":
            return smooth, None
        pointed = create_isoflux_xpts(npts, r0, z0, a, kappa, delta, kappa_lower=kappa_lower,
                                      delta_lower=delta_lower)
        pts = np.concatenate([smooth[smooth[:, 1] >= z0], pointed[pointed[:, 1] < z0]])
        order = np.argsort(np.arctan2(pts[:, 1] - z0, pts[:, 0] - r0))
        x_lower = xpoints_from_moments(r0, z0, a, kappa, delta, kappa_lower, delta_lower)[1]
        return pts[order], x_lower

    def kinetic_profiles(self, kinetic, n_psi):
        psi = np.linspace(0.0, 1.0, n_psi)
        out = {}
        for key in ("n_e", "T_e", "T_i"):
            values = np.asarray(kinetic.get(key, kinetic["T_e"]), dtype=float)
            out[key] = np.interp(psi, np.linspace(0.0, 1.0, values.size), values)
        z_eff = kinetic.get("Z_eff", 1.0)
        if np.ndim(z_eff) == 0:
            out["Z_eff"] = np.full(n_psi, float(z_eff))
        else:
            values = np.asarray(z_eff, dtype=float)
            out["Z_eff"] = np.interp(psi, np.linspace(0.0, 1.0, values.size), values)
        z_imp = float(kinetic.get("Z_imp", 6.0))
        # a single impurity of charge Z_imp behind Z_eff: n_imp/n_e = (Z_eff-1)/(Z(Z-1))
        imp_frac = np.clip((out["Z_eff"] - 1.0) / (z_imp * (z_imp - 1.0)), 0.0, 1.0 / z_imp)
        out["n_i"] = out["n_e"] * (1.0 - z_imp * imp_frac)
        out["Z_imp"] = z_imp
        return out

    def equilibrium(self, config, output_name="equilibrium1", psi_init=None, out_dir=None):
        name = check_name(output_name)
        config = self.check_config(coerce(config, "config", dict) or {})
        out_root = Path(out_dir) if out_dir else self.export_dir
        self.load()
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun

        r0, z0, a = float(config["R0"]), float(config.get("Z0", 0.0)), float(config["a"])
        kappa, delta = float(config["kappa"]), float(config["delta"])
        delta_lower = float(config.get("delta_lower", delta))
        ip, b0, p_axis = float(config["Ip"]), float(config["B0"]), float(config["p_axis"])
        n_psi = int(config.get("n_psi", N_PSI))
        pp = [float(v) for v in config.get("pp", [1.5, 1.5])]
        ffp = [float(v) for v in config.get("ffp", [1.5, 1.5])]
        kinetic = config.get("kinetic")
        started = time.time()
        with self.lock:
            mesh_file = self.machine_mesh(config.get("machine_scale", 1.0))
            self.prepare(mesh_file, r0 * b0)
            gs = self.gs
            gs.settings.maxits = int(config.get("maxits", 400))
            gs.settings.urf = float(config.get("urf", DEFAULT_URF))
            gs.settings.nl_tol = float(config.get("nl_tol", DEFAULT_NL_TOL))
            gs.settings.pm = False
            gs.update_settings()
            if config.get("coil_vsc"):
                gs.set_coil_vsc(dict(config["coil_vsc"]))
            scale = float(config.get("machine_scale", 1.0)) ** 2
            weight = float(config.get("coil_weight", 0.05))
            given = config.get("coil_targets") or {}
            targets = {coil: float(given[coil]) if coil in given else DIIID_COIL_TARGETS[coil]
                       * scale for coil in self.coils if coil in given or coil in DIIID_COIL_TARGETS}
            terms = []
            for coil in self.coils:
                if coil in targets:
                    terms.append(gs.coil_reg_term({coil: 1.0}, target=targets[coil],
                                                  weight=weight * coil_weight_of(coil)))
                else:
                    terms.append(gs.coil_reg_term({coil: 1.0}, target=0.0, weight=1e-2 * weight))
            if config.get("coil_vsc"):
                terms.append(gs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1e-2))
            gs.set_coil_reg(reg_terms=terms)
            factor = float(config.get("coil_bound_factor", 0.0))
            if factor > 0.0:
                largest = max(abs(v) for k, v in DIIID_COIL_TARGETS.items() if k.startswith("F"))
                bounds = {}
                for coil in self.coils:
                    reference = abs(DIIID_COIL_TARGETS.get(coil, largest)) * scale
                    bounds[coil] = [-factor * reference, factor * reference]
                gs.set_coil_bounds(bounds)
            else:
                gs.set_coil_bounds(None)
            boundary, x_lower = self.boundary(config)
            # the boundary points and the X-point weigh this much against the coil prior
            weight_iso = float(config.get("isoflux_weight", 1.0))
            gs.set_isoflux_constraints(boundary, weights=np.full(len(boundary), weight_iso))
            gs.set_saddle_constraints(None if x_lower is None
                                      else np.asarray([x_lower], dtype=float),
                                      weights=None if x_lower is None
                                      else np.asarray([weight_iso], dtype=float))
            gs.set_targets(Ip=ip, pax=p_axis)
            gs.set_profiles(ffp_prof=create_power_flux_fun(n_psi, *ffp),
                            pp_prof=create_power_flux_fun(n_psi, *pp))
            if psi_init is not None:
                gs.set_psi(np.asarray(psi_init, dtype=float), update_bounds=True)
            else:
                gs.init_psi(r0, z0, a, kappa, 0.5 * (delta + delta_lower))
            iterations = gs.solve(return_its=True)[1]
            bootstrap = None
            if kinetic:
                from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap

                prof = self.kinetic_profiles(kinetic, n_psi)
                p_prof = E_KEV_PA * (prof["n_e"] * prof["T_e"] + prof["n_i"] * prof["T_i"])
                gs.set_targets(Ip=ip, pax=float(p_prof[0]))
                bootstrap = solve_with_bootstrap(
                    gs, prof["n_e"], prof["T_e"], prof["n_i"], prof["T_i"], prof["Z_eff"], ip,
                    inductive_jphi=create_power_flux_fun(n_psi, 1.5, 1.5)["y"],
                    Zis=[prof["Z_imp"]], isolate_edge_jBS=True, diagnostic_plots=False,
                    verbose=False)
            stats = gs.get_stats()
            try:
                coil_currents = {str(coil): float(value)
                                 for coil, value in gs.get_coil_currents()[0].items()}
            except Exception as exc:
                coil_currents = {"error": f"{type(exc).__name__}: {exc}"}
            psi_fsa = np.linspace(0.002, 0.998, 200)
            fsa = gs.get_fsa(psi=psi_fsa)
            x_points, diverted = gs.get_xpoints()
            psi_solution = gs.get_psi(False).copy()
            geqdsk = out_root / f"{name}.geqdsk"
            gs.save_eqdsk(str(geqdsk), nr=257, nz=257, lcfs_pad=0.001, run_info=name[:40],
                          cocos=7)
        seconds = time.time() - started
        summary = {"config": config, "iterations": iterations, "stats": stats,
                   "coil_currents": coil_currents,
                   "fsa": fsa, "x_points": x_points, "diverted": diverted,
                   "boundary": boundary, "saddle": x_lower, "seconds": round(seconds, 2)}
        if bootstrap is not None:
            summary["j_BS"] = bootstrap.get("j_BS")
            summary["j_phi"] = bootstrap.get("total_j_phi")
        with open(out_root / f"{name}.json", "w") as handle:
            json.dump(summary, handle, default=jsonable)
        result = {"equilibrium": f"{self.export_prefix}/{name}.geqdsk",
                  "summary": f"{self.export_prefix}/{name}.json",
                  "iterations": iterations, "diverted": bool(diverted),
                  "coil_currents": coil_currents,
                  "stats": {key: jsonable(stats[key]) for key in
                            ("Ip", "beta_pol", "beta_n", "l_i", "q_95", "R_geo", "a_geo",
                             "kappa", "kappaU", "kappaL", "delta", "deltaU", "deltaL")
                            if key in stats},
                  "seconds": round(seconds, 2)}
        return result, psi_solution


class BallooningSimulator:
    def __init__(self, export_dir=".", export_prefix=None, gpec_project=None, julia="julia"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_prefix = export_prefix or str(self.export_dir)
        self.project = gpec_project or os.environ.get("GPEC_PROJECT", "")
        self.julia = julia

    def status(self):
        if not shutil.which(self.julia):
            return f"unavailable: no {self.julia} executable on PATH"
        if not (self.project and os.path.isfile(os.path.join(self.project, "Project.toml"))):
            return "unavailable: GPEC_PROJECT does not name a GPEC checkout"
        if not GPEC_DRIVER.is_file():
            return f"unavailable: no driver at {GPEC_DRIVER}"
        return "available"

    def ballooning(self, file, both=False, output_name="ballooning1", timeout=DEFAULT_TIMEOUT_S,
                   out_dir=None, max_alpha_scale=None):
        return self.ballooning_many([file], both, [output_name], timeout, out_dir,
                                    max_alpha_scale)[0]

    def ballooning_many(self, files, both=False, output_names=(), timeout=DEFAULT_TIMEOUT_S,
                        out_dir=None, max_alpha_scale=None):
        names = [check_name(n) for n in output_names]
        scale = float(GPEC["max_alpha_scale"] if max_alpha_scale is None else max_alpha_scale)
        if not scale > 0.0:
            raise ValueError("max_alpha_scale is a positive number")
        if len(names) != len(files):
            raise ValueError("one output name per equilibrium file")
        status = self.status()
        if status != "available":
            raise RuntimeError(f"ballooning is {status}")
        out_root = Path(out_dir) if out_dir else self.export_dir
        paths = []
        for file in files:
            path = Path(str(file))
            if not path.is_file():
                path = out_root / path.name
            if not path.is_file():
                raise ValueError(f"no equilibrium file at {path}")
            paths.append(path)
        workdir = Path(tempfile.mkdtemp(prefix="gpec_"))
        for path in paths:
            shutil.copy2(path, workdir / path.name)
        threads = os.environ.get("GPEC_THREADS", "4")
        cmd = [self.julia, "-t", str(threads), f"--project={self.project}", str(GPEC_DRIVER),
               f"--n_scan={GPEC['n_scan']}", f"--mpsi={GPEC['mpsi']}", f"--mtheta={GPEC['mtheta']}",
               f"--max_alpha_scale={scale}"]
        cmd += (["--both"] if both else []) + [str(workdir / path.name) for path in paths]
        started = time.time()
        results = []
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout),
                                  cwd=str(workdir))
            if proc.returncode != 0:
                raise RuntimeError(f"GPEC exited with {proc.returncode}: "
                                   f"{proc.stderr[-3000:]}")
            for path, name in zip(paths, names):
                written = workdir / f"ballooning_boundary_{path.stem}.json"
                if not written.is_file():
                    raise RuntimeError(f"GPEC wrote no result for {path.name}: "
                                       f"{proc.stdout[-2000:]}")
                out_path = out_root / f"{name}.json"
                shutil.move(str(written), str(out_path))
                with open(out_path) as handle:
                    payload = json.load(handle)
                results.append({"file": f"{self.export_prefix}/{name}.json",
                                "equilibrium": path.name, "variables": sorted(payload),
                                "n_surfaces": len(payload.get("psi", []))})
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"GPEC exceeded {float(timeout):.0f}s on "
                               f"{[p.name for p in paths]}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        seconds = round(time.time() - started, 2)
        for result in results:
            result["seconds"] = seconds
        return results


class EdgeSimulator:
    def __init__(self, export_dir=".", export_prefix=None, qlk_exec_path=None, mesh_file=None,
                 gpec_project=None, julia="julia", nthreads=4):
        self.torax = ToraxSimulator(export_dir, export_prefix, qlk_exec_path)
        self.equil = EquilibriumSimulator(export_dir, export_prefix, mesh_file, nthreads)
        self.stability = BallooningSimulator(export_dir, export_prefix, gpec_project, julia)

    def register_external_models(self):
        self.torax.register_external_models()

    def capabilities(self, output_name="capabilities"):
        name = check_name(output_name)
        payload = self.torax.capabilities(name)
        payload["external_models"] = dict(payload["external_models"],
                                          tokamaker=self.equil.status(),
                                          gpec=self.stability.status())
        payload["commands"] = list(COMMANDS)
        # the name of the machine mesh, never the path it sits at on the host
        payload["tokamaker_mesh"] = os.path.basename(self.equil.mesh_file or "") or None
        payload["gpec_numerics"] = dict(GPEC)
        with open(self.torax.export_dir / f"{name}.json", "w") as handle:
            json.dump({k: v for k, v in payload.items() if k != "file"}, handle, indent=1,
                      default=str)
        return payload

    def config_schema(self, section, output_name="schema"):
        return self.torax.config_schema(section, output_name)

    def simulate(self, config, output_name="run1", progress_bar=False,
                 log_timestep_info=False, max_steps=0):
        return self.torax.simulate(config, output_name, progress_bar, log_timestep_info,
                                   max_steps)

    def read(self, file, variables=None, time_index=-1, output_name="read1"):
        return self.torax.read(file, variables, time_index, output_name)

    def equilibrium(self, config, output_name="equilibrium1"):
        return self.equil.equilibrium(config, output_name)[0]

    def ballooning(self, file, both=False, output_name="ballooning1", timeout=DEFAULT_TIMEOUT_S,
                   max_alpha_scale=None):
        return self.stability.ballooning(file, both, output_name, timeout,
                                         max_alpha_scale=max_alpha_scale)

    def run(self, config, output_name="run1"):
        import coupled

        name = check_name(output_name)
        config = coerce(config, "config", dict) or {}
        # as for the stability command, the working files stay outside the export directory
        workdir = Path(tempfile.mkdtemp(prefix=f"run_{name}_"))
        try:
            summary = coupled.run_setting(config, self, name, workdir, reveal=False)
            coupled.publish(summary, name, workdir, self.torax.export_dir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        prefix = self.torax.export_prefix
        return {"summary": f"{prefix}/{name}.json",
                "equilibrium": f"{prefix}/{name}_eq.geqdsk",
                "run": f"{prefix}/{name}_run.nc",
                "sim_error": summary["sim_error"],
                "edge": {key: round(float(v), 4) for key, v in summary["truth"].items()},
                "seconds": summary["timing"]["total"]}


PROTOCOL = "rest"


def Simulator(export_dir, export_prefix, work_dir):
    simulator = EdgeSimulator(export_dir=export_dir, export_prefix=export_prefix,
                              nthreads=int(os.environ.get("FUSION_OFT_THREADS", "4")))
    simulator.register_external_models()
    return simulator
