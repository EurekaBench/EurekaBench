import argparse
import json
import sys
import urllib.error
import urllib.request

PORT = 8765
COMMANDS = ("capabilities", "config_schema", "shapes", "geometry", "props", "simulate",
            "read")


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
    if value is None:
        return None
    text = str(value)
    if text.endswith(".json"):
        with open(text) as handle:
            return json.load(handle)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} is not valid JSON and is not a .json path: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI access to the TORAX tokamak transport simulator and the "
                    "library of matched equilibria")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", help="full TORAX config, JSON string or .json path")
    parser.add_argument("--section", help="config_schema: which config section")
    parser.add_argument("--shape", help="geometry: a shape of the library")
    parser.add_argument("--n_rho", type=int, default=25,
                        help="geometry: radial cells of the grid the surfaces are on")
    parser.add_argument("--n_theta", type=int, default=64,
                        help="geometry: poloidal angles per surface")
    parser.add_argument("--file", help="read: the .nc written by simulate")
    parser.add_argument("--variables", help="read: JSON list, omit for every variable")
    parser.add_argument("--time_index", type=int, default=-1)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--log_timestep_info", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    payload = {}
    if args.output_name:
        payload["output_name"] = args.output_name
    if args.command == "simulate":
        if not args.config:
            raise SystemExit("simulate needs --config")
        payload["config"] = load(args.config, "--config")
        payload["progress_bar"] = args.progress_bar
        payload["log_timestep_info"] = args.log_timestep_info
        payload["max_steps"] = args.max_steps
    elif args.command == "config_schema":
        if not args.section:
            raise SystemExit("config_schema needs --section")
        payload["section"] = args.section
    elif args.command == "geometry":
        if not args.shape:
            raise SystemExit("geometry needs --shape")
        payload["shape"] = args.shape
        payload["n_rho"] = args.n_rho
        payload["n_theta"] = args.n_theta
    elif args.command == "read":
        if not args.file:
            raise SystemExit("read needs --file")
        payload["file"] = args.file
        payload["time_index"] = args.time_index
        if args.variables:
            payload["variables"] = load(args.variables, "--variables")

    result = call(args.command, payload, args.timeout)
    print(json.dumps(result, indent=1, default=str))
    sys.exit(1 if isinstance(result, dict) and "error" in result else 0)


if __name__ == "__main__":
    main()
