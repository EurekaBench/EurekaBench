import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arc
import nt

PROTOCOL = "rest"
COMMANDS = ("capabilities", "config_schema", "shapes", "geometry", "props", "simulate", "read")
ARC_COMMANDS = ("capabilities", "config_schema", "simulate", "read")


def jax_backend():
    import jax
    import jax.numpy as jnp

    jnp.zeros(1).block_until_ready()
    return {"jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()]}


def Simulator(export_dir, export_prefix, work_dir, variant):
    global COMMANDS
    if variant == "arc":
        simulator = arc.ToraxSimulator(export_dir=export_dir, export_prefix=export_prefix)
        simulator.register_external_models()
        COMMANDS = ARC_COMMANDS
    elif variant == "nt":
        simulator = nt.ToraxSimulator(export_dir=export_dir, export_prefix=export_prefix,
                                      equilibria_dir=HERE / "equilibria")
    else:
        raise ValueError(f"unknown torax variant {variant!r}; arc or nt")
    print(f"[simulator] {variant}: JAX on {jax_backend()}", flush=True)
    return simulator
