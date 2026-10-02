import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


PARAMETER_SETS = ("A", "B", "C", "C0", "C1", "C2")

POLARITY_VALUES = {"exc", "inh"}
STIMULUS_KINDS = {"pulse", "sine"}

MAX_DURATION_MS = 20000.0
MAX_AMPLITUDE_PA = 1000.0

BODY_SANDBOX_ENV = "NEURO_BODY_SANDBOX"
BODY_MAX_DURATION_MS = 200.0


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


class C302Simulator:
    def __init__(self, export_dir=".", export_prefix=None, work_dir=None):
        import c302

        self.c302 = c302
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_prefix = export_prefix or str(self.export_dir)
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="c302_sim_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.data_reader = c302.DEFAULT_DATA_READER
        self.neuron_names = None
        self.muscle_names = None

    def load_names(self):
        if self.neuron_names is None:
            cells, _ = self.c302.get_cell_names_and_connection(self.data_reader)
            _, muscles, _ = self.c302.get_cell_muscle_names_and_connection(self.data_reader)
            self.neuron_names = sorted(cells)
            self.muscle_names = sorted(muscles)
        return self.neuron_names, self.muscle_names

    def check_name(self, output_name):
        name = str(output_name)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError("output_name may only contain letters, digits, '.', '_', '-'")
        return name

    def conn_record(self, conn):
        gap = "_GJ" in conn.synclass
        return {"pre": conn.pre_cell, "post": conn.post_cell,
                "kind": "gap_junction" if gap else "chemical",
                "neurotransmitter": None if gap else conn.synclass,
                "number": conn.number}

    def get_connectome(self, output_name="connectome"):
        name = self.check_name(output_name)
        cells, conns = self.c302.get_cell_names_and_connection(self.data_reader)
        _, muscles, muscle_conns = self.c302.get_cell_muscle_names_and_connection(
            self.data_reader)
        payload = {
            "neurons": sorted(cells),
            "muscles": sorted(muscles),
            "connections": [self.conn_record(c) for c in conns],
            "neuron_to_muscle_connections": [self.conn_record(c) for c in muscle_conns
                                             if c.post_cell in muscles],
        }
        out_path = self.export_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1)
        return {"file": f"{self.export_prefix}/{name}.json",
                "n_neurons": len(payload["neurons"]),
                "n_muscles": len(payload["muscles"]),
                "n_connections": len(payload["connections"]),
                "n_neuron_to_muscle_connections":
                    len(payload["neuron_to_muscle_connections"])}

    def check_stimuli(self, stimuli, allowed):
        stimulated = set()
        for stim in stimuli:
            if not isinstance(stim, dict):
                raise ValueError("each stimulus must be a JSON object")
            kind = str(stim.get("kind", "pulse"))
            if kind not in STIMULUS_KINDS:
                raise ValueError(f"stimulus kind must be one of {sorted(STIMULUS_KINDS)}")
            required = {"cell", "delay_ms", "duration_ms", "amplitude_pa"}
            if kind == "sine":
                required = required | {"period_ms"}
            missing = required - set(stim)
            if missing:
                raise ValueError(f"{kind} stimulus missing fields: {sorted(missing)}")
            cell = str(stim["cell"])
            if cell not in allowed:
                raise ValueError(f"stimulated cell {cell!r} is not an included cell")
            if float(stim["delay_ms"]) < 0 or float(stim["duration_ms"]) <= 0:
                raise ValueError("stimulus delay_ms must be >= 0 and duration_ms > 0")
            if abs(float(stim["amplitude_pa"])) > MAX_AMPLITUDE_PA:
                raise ValueError(f"stimulus |amplitude_pa| must be <= {MAX_AMPLITUDE_PA:g}")
            if kind == "sine" and float(stim["period_ms"]) <= 0:
                raise ValueError("sine stimulus period_ms must be > 0")
            stimulated.add(cell)
        return stimulated

    def simulate_network(self, cells=None, muscles=None, stimuli=None, parameter_set="C1",
                         duration=500.0, dt=0.05, save_every_ms=None,
                         remove_connections=None, keep_only_connections=None,
                         connection_number_override=None, connection_number_scaling=None,
                         connection_polarity_override=None, param_overrides=None,
                         data_reader=None, output_name="sim1"):
        import neuroml.writers as writers
        from neuroml import SineGenerator
        from pyneuroml import pynml

        name = self.check_name(output_name)
        parameter_set = str(parameter_set)
        if parameter_set not in PARAMETER_SETS:
            raise ValueError(f"parameter_set must be one of {list(PARAMETER_SETS)}")
        duration = float(duration)
        dt = float(dt)
        if not (0 < duration <= MAX_DURATION_MS):
            raise ValueError(f"duration (ms) must be in (0, {MAX_DURATION_MS:g}]")
        if not (0.01 <= dt <= 1.0):
            raise ValueError("dt (ms) must be in [0.01, 1.0]")
        save_every_ms = dt if save_every_ms is None else float(save_every_ms)
        if save_every_ms < dt:
            raise ValueError("save_every_ms must be >= dt")

        cells = coerce(cells, "cells", list)
        muscles = coerce(muscles, "muscles", list) or []
        stimuli = coerce(stimuli, "stimuli", list) or []
        remove_connections = coerce(remove_connections, "remove_connections", list) or []
        keep_only_connections = coerce(keep_only_connections, "keep_only_connections",
                                       list) or []
        number_override = coerce(connection_number_override,
                                 "connection_number_override", dict)
        number_scaling = coerce(connection_number_scaling,
                                "connection_number_scaling", dict)
        polarity_override = coerce(connection_polarity_override,
                                   "connection_polarity_override", dict)
        param_overrides = coerce(param_overrides, "param_overrides", dict) or {}
        for key, value in param_overrides.items():
            if key == "mirrored_elec_conn_params":
                if not isinstance(value, dict):
                    raise ValueError("param_overrides.mirrored_elec_conn_params must be "
                                     "a JSON object")
            elif isinstance(value, (dict, list)):
                raise ValueError(f"param_overrides[{key!r}] must be a string with a unit")
        reader = self.data_reader if data_reader is None else str(data_reader)
        if not reader.startswith("cect.readers."):
            raise ValueError("data_reader must be a module name 'cect.readers.<Reader>'")
        if reader != self.data_reader:
            cells_known, _ = self.c302.get_cell_names_and_connection(reader)
            _, muscles_known, _ = self.c302.get_cell_muscle_names_and_connection(reader)
            neuron_names, muscle_names = sorted(cells_known), sorted(muscles_known)
        else:
            neuron_names, muscle_names = self.load_names()

        if cells is not None:
            unknown = sorted(set(cells) - set(neuron_names))
            if unknown:
                raise ValueError(f"unknown neuron names in cells: {unknown}")
        unknown = sorted(set(muscles) - set(muscle_names))
        if unknown:
            raise ValueError(f"unknown muscle names in muscles: {unknown}")
        if number_override is not None:
            number_override = {k: float(v) for k, v in number_override.items()}
        if number_scaling is not None:
            number_scaling = {k: float(v) for k, v in number_scaling.items()}
        if polarity_override is not None:
            bad = {k: v for k, v in polarity_override.items() if v not in POLARITY_VALUES}
            if bad:
                raise ValueError(f"connection_polarity_override values must be one of "
                                 f"{sorted(POLARITY_VALUES)}, got {bad}")
        stimulated = self.check_stimuli(
            stimuli, (set(neuron_names) if cells is None else set(cells)) | set(muscles))

        module = importlib.import_module(f"c302.parameters_{parameter_set}")
        params = module.ParameterisedModel()
        net_id = f"net_{name}"
        run_dir = self.work_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        nml_doc = self.c302.generate(
            net_id, params, data_reader=reader,
            cells=list(cells) if cells is not None else None,
            muscles_to_include=list(muscles),
            cells_to_stimulate=[],
            conns_to_include=list(keep_only_connections),
            conns_to_exclude=list(remove_connections),
            conn_number_override=number_override,
            conn_number_scaling=number_scaling,
            conn_polarity_override=polarity_override,
            param_overrides=param_overrides,
            duration=duration, dt=dt,
            target_directory=str(run_dir), verbose=False,
        )
        net = nml_doc.networks[0]
        n_connections = (len(net.projections) + len(net.electrical_projections)
                         + len(net.continuous_projections))
        for stim in stimuli:
            cell = str(stim["cell"])
            delay = f"{float(stim['delay_ms'])}ms"
            length = f"{float(stim['duration_ms'])}ms"
            amplitude = f"{float(stim['amplitude_pa'])}pA"
            if str(stim.get("kind", "pulse")) == "pulse":
                self.c302.add_new_input(nml_doc, cell, delay, length, amplitude, params)
                continue
            n = 1 + sum(1 for s in nml_doc.sine_generators
                        if s.id.startswith(f"sine_{cell}_"))
            gen = SineGenerator(id=f"sine_{cell}_{n}", delay=delay, duration=length,
                                amplitude=amplitude, period=f"{float(stim['period_ms'])}ms",
                                phase=str(float(stim.get("phase_rad", 0.0))))
            nml_doc.sine_generators.append(gen)
            self.c302.append_input_to_nml_input_list(gen, nml_doc, cell, params)
        writers.NeuroMLWriter.write(nml_doc, str(run_dir / f"{net_id}.net.nml"))

        results = pynml.run_lems_with_jneuroml(
            f"LEMS_{net_id}.xml", exec_in_dir=str(run_dir), max_memory="4G",
            nogui=True, load_saved_data=True, verbose=False, exit_on_fail=False)
        if not isinstance(results, dict):
            raise RuntimeError(
                "the simulation failed to complete; this usually means the numerical "
                "integration became unstable for the chosen configuration (a smaller dt "
                "often fixes it) or the network was too large for the available memory")

        step = max(int(round(save_every_ms / dt)), 1)
        data = {"t": (np.asarray(results["t"], dtype=float) * 1000.0)[::step]}
        for key, series in results.items():
            if key == "t":
                continue
            parts = key.split("/")
            quantity = parts[-1]
            arr = np.asarray(series, dtype=np.float32)[::step]
            if quantity == "v":
                arr = arr * 1000.0
            data[f"{quantity}_{parts[0]}"] = arr
        out_path = self.export_dir / f"{name}.npz"
        np.savez(out_path, **data)

        units = {"t": "ms", "v_<CELL>": "mV"}
        for quantity in sorted({k.split("_")[0] for k in data
                                if k != "t" and not k.startswith("v_")}):
            units[f"{quantity}_<CELL>"] = "simulator native units"
        return {
            "file": f"{self.export_prefix}/{name}.npz",
            "arrays": {k: list(v.shape) for k, v in data.items()},
            "units": units,
            "parameter_set": parameter_set,
            "data_reader": reader,
            "n_param_overrides": len(param_overrides),
            "duration_ms": duration,
            "dt_ms": dt,
            "save_every_ms": save_every_ms,
            "n_cells_included": len([k for k in data if k.startswith("v_")]),
            "n_connections_included": n_connections,
            "stimulated": sorted(stimulated),
            "seconds": round(time.time() - t0, 1),
        }

    def simulate_body(self, duration_ms=50.0, output_name="body1"):
        name = self.check_name(output_name)
        duration = float(duration_ms)
        if not (0 < duration <= BODY_MAX_DURATION_MS):
            raise ValueError(f"duration_ms must be in (0, {BODY_MAX_DURATION_MS:g}]")
        sandbox = os.environ.get(BODY_SANDBOX_ENV, "")
        if not sandbox or not Path(sandbox, "home", "ow").is_dir():
            raise RuntimeError("the whole-worm body simulator is not installed on this "
                               "host, so simulate_body is unavailable in this session")
        run_dir = self.work_dir / f"body_{name}"
        shared = run_dir / "shared"
        shutil.rmtree(run_dir, ignore_errors=True)
        shared.mkdir(parents=True)
        Path(sandbox, "home", "ow", "shared").mkdir(parents=True, exist_ok=True)
        cmd = ["apptainer", "exec", "--cleanenv", "--pid", "--writable", "--no-home",
               "--bind", f"{shared}:/home/ow/shared",
               "--env", "OW_OUT_DIR=/home/ow/shared",
               "--env", f"DURATION={duration:g}",
               "--env", "DISPLAY=:44", "--pwd", "/home/ow", sandbox,
               "bash", "-c", "export HOME=/home/ow && python3 master_openworm.py"]
        t0 = time.time()
        with open(run_dir / "body_sim.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        reports = sorted((shared / "output").glob("*/report.json")) \
            if (shared / "output").is_dir() else []
        if not reports:
            with open(run_dir / "body_sim.log") as f:
                tail = f.read()[-3000:]
            raise RuntimeError(f"the body simulation produced no report; log tail:\n{tail}")
        with open(reports[-1]) as f:
            report = json.load(f)
        # the whole output directory of the run, exactly as OpenWorm wrote it
        dest = self.export_dir / name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(reports[-1].parent), str(dest))
        shutil.rmtree(run_dir, ignore_errors=True)
        return {"directory": f"{self.export_prefix}/{name}",
                "files": sorted(p.name for p in dest.iterdir()),
                "duration_ms": duration,
                "report": report,
                "seconds": round(time.time() - t0, 1)}


Simulator = C302Simulator
COMMANDS = ("get_connectome", "simulate_network", "simulate_body")
