import json
import sys
import urllib.error
import urllib.request

import fire

PORT = 8765
COMMANDS = ("capabilities", "config_schema", "simulate", "read", "equilibrium", "ballooning",
            "run")
RAW_FLAGS = ("--config", "--variables")
RAW = {}


def stash_raw(argv):
    out, i = [], 0
    while i < len(argv):
        arg = argv[i]
        hit = False
        for flag in RAW_FLAGS:
            if arg == flag and i + 1 < len(argv):
                RAW[flag] = argv[i + 1]
                out += [flag, f"__{flag[2:]}__"]
                i += 2
                hit = True
                break
            if arg.startswith(flag + "="):
                RAW[flag] = arg[len(flag) + 1:]
                out.append(f"{flag}=__{flag[2:]}__")
                i += 1
                hit = True
                break
        if not hit:
            out.append(arg)
            i += 1
    return out


def call(command, payload, timeout):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/{command}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())
    except urllib.error.URLError as exc:
        return {"error": f"cannot reach the simulator server: {exc.reason}"}


def load(value, name):
    value = RAW.get(name, value)
    if value is None or isinstance(value, (dict, list)):
        return value
    text = str(value)
    if text.endswith(".json"):
        with open(text) as handle:
            return json.load(handle)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} is not valid JSON and is not a .json path: {exc}")


def named(output_name):
    return {} if output_name is None else {"output_name": str(output_name)}


def finish(command, payload, timeout):
    result = call(command, payload, float(timeout))
    print(json.dumps(result, indent=1, default=str))
    sys.exit(1 if isinstance(result, dict) and "error" in result else 0)


def capabilities(output_name=None, timeout=3600.0):
    finish("capabilities", named(output_name), timeout)


def config_schema(section, output_name=None, timeout=3600.0):
    payload = named(output_name)
    payload["section"] = str(section)
    finish("config_schema", payload, timeout)


def simulate(config, output_name=None, progress_bar=False, log_timestep_info=False,
             max_steps=0, timeout=3600.0):
    payload = named(output_name)
    payload.update(config=load(config, "--config"), progress_bar=bool(progress_bar),
                   log_timestep_info=bool(log_timestep_info), max_steps=int(max_steps))
    finish("simulate", payload, timeout)


def read(file, variables=None, time_index=-1, output_name=None, timeout=3600.0):
    payload = named(output_name)
    payload.update(file=str(file), time_index=int(time_index))
    if variables is not None:
        payload["variables"] = load(variables, "--variables")
    finish("read", payload, timeout)


def equilibrium(config, output_name=None, timeout=3600.0):
    payload = named(output_name)
    payload["config"] = load(config, "--config")
    finish("equilibrium", payload, timeout)


def ballooning(file, both=False, output_name=None, timeout=3600.0, max_alpha_scale=8.0):
    payload = named(output_name)
    payload.update(file=str(file), both=bool(both), timeout=float(timeout),
                   max_alpha_scale=float(max_alpha_scale))
    finish("ballooning", payload, timeout)


def run(config, output_name=None, timeout=7200.0):
    payload = named(output_name)
    payload["config"] = load(config, "--config")
    finish("run", payload, timeout)


if __name__ == "__main__":
    fire.Fire({"capabilities": capabilities, "config_schema": config_schema,
               "simulate": simulate, "read": read, "equilibrium": equilibrium,
               "ballooning": ballooning, "run": run}, command=stash_raw(sys.argv[1:]))
