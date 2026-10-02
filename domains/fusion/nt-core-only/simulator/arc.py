import json
import os
import re
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
        name = str(output_name)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError("output_name may only contain letters, digits, '.', '_', '-'")
        return name

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
                 log_timestep_info=False, max_steps=0):
        self.register_external_models()
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
