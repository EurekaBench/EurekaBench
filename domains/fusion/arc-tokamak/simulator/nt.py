import copy
import json
import re
import time
from pathlib import Path

import numpy as np

import nt_protocol as protocol
import equilibrium

CONFIG_SECTIONS = ("profile_conditions", "numerics", "plasma_composition", "geometry",
                   "sources", "neoclassical", "solver", "transport", "pedestal",
                   "mhd", "edge", "time_step_calculator", "restart")

MAX_T_FINAL = 100.0
MAX_N_RHO = 200
MAX_STEPS = 200000
MAX_N_THETA = 512
DEFAULT_TIMEOUT_S = 3600.0


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


def library_shapes():
    kappa = protocol.MACHINE["elongation"]
    return {equilibrium.shape_name(1.0, kappa, d): {"delta": d, "kappa": kappa, "size": 1.0}
            for d in protocol.SCAN_DELTAS}


class ToraxSimulator:
    def __init__(self, export_dir=".", export_prefix=None, equilibria_dir=None):
        import torax

        self.torax = torax
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_prefix = export_prefix or str(self.export_dir)
        self.equilibria_dir = Path(equilibria_dir or protocol.EQUILIBRIA_DIR)
        self.catalogue = library_shapes()
        self.library = equilibrium.ensure_library(
            self.equilibria_dir, protocol.MACHINE,
            [(s["size"], s["kappa"], s["delta"]) for s in self.catalogue.values()])
        self.equilibria = {}

    def check_name(self, output_name):
        name = str(output_name)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError("output_name may only contain letters, digits, '.', '_', '-'")
        return name

    def check_shape(self, shape):
        name = str(shape)
        if name.endswith(".eqdsk"):
            name = name[:-len(".eqdsk")]
        if name not in self.catalogue:
            raise ValueError(f"unknown shape {shape!r}; the library holds "
                             f"{list(self.catalogue)}")
        return name

    def check_config(self, config):
        unknown = sorted(set(config) - set(CONFIG_SECTIONS))
        if unknown:
            raise ValueError(f"unknown configuration sections: {unknown}; "
                             f"TORAX accepts {list(CONFIG_SECTIONS)}")
        numerics = config.get("numerics") or {}
        t_final = float(numerics.get("t_final", 5.0))
        if not (0 < t_final <= MAX_T_FINAL):
            raise ValueError(f"numerics.t_final must be in (0, {MAX_T_FINAL:g}]")
        geometry = config.get("geometry") or {}
        n_rho = geometry.get("n_rho")
        if n_rho is not None and not (1 <= int(n_rho) <= MAX_N_RHO):
            raise ValueError(f"geometry.n_rho must be in [1, {MAX_N_RHO}]")
        if str(geometry.get("geometry_type", "")).lower() == "eqdsk":
            config = copy.deepcopy(config)
            name = self.check_shape(geometry.get("geometry_file", ""))
            config["geometry"]["geometry_file"] = f"{name}.eqdsk"
            config["geometry"]["geometry_directory"] = str(self.equilibria_dir)
            config["geometry"].pop("eqdsk_object", None)
        return config

    def equilibrium_of(self, name):
        if name not in self.equilibria:
            shape = self.catalogue[name]
            self.equilibria[name] = equilibrium.build(
                protocol.MACHINE, shape["size"], shape["kappa"], shape["delta"])
        return self.equilibria[name]

    def capabilities(self, output_name="capabilities"):
        from typing import get_args
        from torax._src.torax_pydantic import model_config

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
                   "shapes": list(self.catalogue),
                   "limits": {"t_final": MAX_T_FINAL, "n_rho": MAX_N_RHO,
                              "steps": MAX_STEPS, "n_theta": MAX_N_THETA,
                              "timeout_s": DEFAULT_TIMEOUT_S}}
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

    def props(self, output_name="props"):
        name = self.check_name(output_name)
        payload = {"rule": "every profile and scalar of a run that the equilibrium, the "
                           "prescribed density, the composition and the heating settle "
                           "before the plasma is solved, under the name the simulator "
                           "gives it, together with the flux surfaces, the heating "
                           "delivered to each species, the boundary condition and the "
                           "grids",
                   "radial": sorted(set(protocol.SETTLED_PROFILES)
                                    | {"p_ext_i", "p_ext_e", "rho"}),
                   "surface": list(protocol.PROPS_SURFACE) + ["theta"],
                   "scalar": sorted(set(protocol.SETTLED_SCALARS)
                                    | {"T_bc", "rho_bc", "n_i_over_n_e"}),
                   "face": ["rho_face"],
                   "shapes": {"radial": "(n_rho + 2,) on the rho grid",
                              "surface": "(n_rho + 2, n_theta) on the rho grid at theta",
                              "face": "(n_rho + 1,)", "scalar": "one number"},
                   "absent": ["every temperature", "every transport coefficient",
                              "every quantity built from them", "the triangularity",
                              "the elongation"],
                   "note": "a radial entry appears when the run reports it"}
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=1, default=str)
        payload["file"] = f"{self.export_prefix}/{name}.json"
        return payload

    def shapes(self, output_name="shapes"):
        name = self.check_name(output_name)
        payload = {"machine": {k: protocol.MACHINE[k] for k in
                               ("R_major", "a_minor", "B_0", "Ip", "elongation")},
                   "shapes": {}}
        for shape_name, shape in self.catalogue.items():
            entry = dict(shape)
            entry.update({k: v for k, v in self.library.get(shape_name, {}).items()
                          if k in ("R_axis", "q_near_axis", "q_boundary",
                                   "psi_axis_wb", "file")})
            payload["shapes"][shape_name] = entry
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=1, default=str)
        payload["file"] = f"{self.export_prefix}/{name}.json"
        return payload

    def geometry(self, shape, n_rho=protocol.N_RHO, n_theta=equilibrium.N_THETA,
                 output_name="geometry"):
        name = self.check_name(output_name)
        shape_name = self.check_shape(shape)
        n_rho, n_theta = int(n_rho), int(n_theta)
        if not (1 <= n_rho <= MAX_N_RHO) or not (8 <= n_theta <= MAX_N_THETA):
            raise ValueError(f"n_rho must be in [1, {MAX_N_RHO}] and n_theta in "
                             f"[8, {MAX_N_THETA}]")
        eq = self.equilibrium_of(shape_name)
        rho = np.concatenate([[0.0], (np.arange(n_rho) + 0.5) / n_rho, [1.0]])
        psi = eq.psi_of_rho(rho)
        theta, surfaces = eq.surfaces(psi, n_theta)
        q = np.empty(rho.size)
        q[1:] = eq.safety_factor(psi[1:])
        q[0] = np.polyval(np.polyfit(psi[1:4], q[1:4], 2), psi[0])
        payload = {"shape": shape_name, "delta": self.catalogue[shape_name]["delta"],
                   "kappa": self.catalogue[shape_name]["kappa"],
                   "R_major": eq.R_major, "a_minor": eq.a_minor, "B_0": eq.B_0,
                   "Ip": eq.Ip, "R_axis": eq.R_major * eq.x_axis,
                   "rho": rho.tolist(), "psi": psi.tolist(), "q": q.tolist(),
                   "theta": theta.tolist(),
                   "R_surface": surfaces["R"].tolist(),
                   "Z_surface": surfaces["Z"].tolist(),
                   "B_surface": surfaces["B"].tolist(),
                   "B_pol_surface": surfaces["B_pol"].tolist(),
                   "note": "rho is the normalised toroidal flux grid of a run with n_rho "
                           "cells: the cell centres with the axis and the last closed "
                           "flux surface at its two ends. psi is the poloidal flux in Wb "
                           "at each of them, the surfaces are sampled at theta about the "
                           "magnetic axis, B is the field strength in T and B_pol its "
                           "poloidal part."}
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as handle:
            json.dump(payload, handle, default=str)
        return {"file": f"{self.export_prefix}/{name}.json", "shape": shape_name,
                "n_rho": n_rho, "n_theta": n_theta,
                "arrays": ["rho", "psi", "q", "theta", "R_surface", "Z_surface",
                           "B_surface", "B_pol_surface"]}

    def simulate(self, config, output_name="run1", progress_bar=False,
                 log_timestep_info=False, max_steps=0):
        name = self.check_name(output_name)
        config = self.check_config(coerce(config, "config", dict) or {})
        cap = int(max_steps) or None
        if cap is not None and not (1 <= cap <= MAX_STEPS):
            raise ValueError(f"max_steps must be in [1, {MAX_STEPS}]")

        started = time.time()
        torax_config = self.torax.ToraxConfig.from_dict(config)
        data_tree, history = self.torax.run_simulation(
            torax_config, progress_bar=bool(progress_bar),
            log_timestep_info=bool(log_timestep_info), max_steps=cap)
        seconds = time.time() - started

        out_path = self.export_dir / f"{name}.nc"
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
