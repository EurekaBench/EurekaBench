import hashlib
import pickle
import re
import time
from pathlib import Path

import numpy as np


MODEL_REPO = "facebook/tribev2"
CHECKPOINT_NAME = "best.ckpt"

# the suffixes tribev2 accepts, one argument of get_events_dataframe each
INPUT_KINDS = {
    "text_path": {".txt"},
    "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
    "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}

# the events dataframe carries the path of the file on this side of the bind mount
HOST_COLUMNS = ("filepath",)


class TribeSimulator:
    def __init__(self, export_dir=".", export_prefix=None, workspace_dir=None,
                 workspace_prefix="/workspace", work_dir=None, cache_dir=None,
                 device="auto"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_prefix = export_prefix or str(self.export_dir)
        self.workspace_dir = Path(workspace_dir or self.export_dir)
        self.workspace_prefix = workspace_prefix
        self.work_dir = Path(work_dir or self.export_dir / ".tribe_work")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir or self.work_dir / "features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.model = None

    def check_name(self, output_name):
        name = str(output_name)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError("output_name may only contain letters, digits, '.', "
                             "'_', '-'")
        return name

    def resolve(self, input_path):
        given = str(input_path)
        relative = given[len(self.workspace_prefix):].lstrip("/") \
            if given.startswith(self.workspace_prefix) else given.lstrip("/")
        path = (self.workspace_dir / relative).resolve()
        if not str(path).startswith(str(self.workspace_dir.resolve())):
            raise ValueError(f"input_path must be under {self.workspace_prefix}")
        if not path.is_file():
            raise ValueError(f"no such file: {given}")
        suffix = path.suffix.lower()
        for argument, suffixes in INPUT_KINDS.items():
            if suffix in suffixes:
                return path, argument
        allowed = sorted(s for group in INPUT_KINDS.values() for s in group)
        raise ValueError(f"{given} has suffix {suffix!r}; the accepted ones are "
                         f"{allowed}")

    def load_model(self):
        if self.model is None:
            from tribev2 import TribeModel
            self.model = TribeModel.from_pretrained(
                MODEL_REPO, checkpoint_name=CHECKPOINT_NAME,
                cache_folder=str(self.cache_dir), device=self.device)
        return self.model

    def events_for(self, path, argument, transcribe=True):
        stat = path.stat()
        key = hashlib.sha1(f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{int(transcribe)}"
                           .encode()).hexdigest()[:16]
        cached = self.work_dir / f"events_{key}.pkl"
        if cached.is_file():
            with open(cached, "rb") as f:
                return pickle.load(f)
        if transcribe or argument == "text_path":
            events = self.load_model().get_events_dataframe(**{argument: str(path)})
        else:
            # an input with nothing spoken in it has no words to recover
            import pandas as pd
            from tribev2.demo_utils import get_audio_and_text_events
            frame = pd.DataFrame([{"type": "Video" if argument == "video_path"
                                   else "Audio", "filepath": str(path), "start": 0,
                                   "timeline": "default", "subject": "default"}])
            events = get_audio_and_text_events(frame, audio_only=True)
        with open(cached, "wb") as f:
            pickle.dump(events, f)
        return events

    def get_events(self, input_path, output_name="events", transcribe=True):
        name = self.check_name(output_name)
        path, argument = self.resolve(input_path)
        t0 = time.time()
        events = self.events_for(path, argument, transcribe)
        table = events.drop(columns=[c for c in HOST_COLUMNS if c in events.columns])
        out = self.export_dir / f"{name}.csv"
        table.to_csv(out, index=False)
        counts = table["type"].value_counts().to_dict()
        return {"file": f"{self.export_prefix}/{name}.csv",
                "input": str(input_path),
                "n_events": int(len(table)),
                "events_per_type": {str(k): int(v) for k, v in counts.items()},
                "columns": [str(c) for c in table.columns],
                "seconds": round(time.time() - t0, 1)}

    def simulate(self, input_path, output_name="run1", transcribe=True):
        name = self.check_name(output_name)
        path, argument = self.resolve(input_path)
        model = self.load_model()
        t0 = time.time()
        events = self.events_for(path, argument, transcribe)
        predicted, segments = model.predict(events=events, verbose=False)
        t = np.asarray([float(s.start) for s in segments], dtype=float)
        order = np.argsort(t, kind="stable")
        t, bold = t[order], np.asarray(predicted, dtype=np.float32)[order]
        np.savez(self.export_dir / f"{name}.npz", t=t, bold=bold)
        return {"file": f"{self.export_prefix}/{name}.npz",
                "input": str(input_path),
                "arrays": {"t": list(t.shape), "bold": list(bold.shape)},
                "units": {"t": "s", "bold": "z-scored, arbitrary units"},
                "tr_s": float(model.data.TR),
                "n_timesteps": int(bold.shape[0]),
                "n_vertices": int(bold.shape[1]),
                "summary": {"min": float(bold.min()), "max": float(bold.max()),
                            "mean": float(bold.mean()), "std": float(bold.std())},
                "seconds": round(time.time() - t0, 1)}


class Simulator(TribeSimulator):
    def __init__(self, export_dir, export_prefix, work_dir):
        super().__init__(export_dir=export_dir, export_prefix=export_prefix,
                         workspace_dir=Path(export_dir).parent, workspace_prefix="/workspace",
                         work_dir=work_dir, cache_dir=None, device="auto")
        self.load_model()


COMMANDS = ("get_events", "simulate")
