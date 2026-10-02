import importlib.util
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import fire

MERGED_SETTINGS = ("connection_polarity_override", "connection_number_scaling", "param_overrides")
PING = b'{"method": "__ping__", "params": {}}'
HOSTS = b'{"method": "__hosts__", "params": {}}'
ARCHIVE = "/archive/"


def serve(simulator, port, export_dir, export_prefix, work_dir, settings="{}"):
    simulator = Path(simulator)
    sys.path.insert(0, str(simulator.parent))
    spec = importlib.util.spec_from_file_location("simulator", simulator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = json.loads(settings) if isinstance(settings, str) else dict(settings or {})
    args = dict(settings.get("args", {}))
    if settings.get("verifier") and hasattr(module, "VERIFIER_COMMANDS"):
        args["bridge_key"] = settings.get("bridge_key", "")
    instance = module.Simulator(export_dir=export_dir, export_prefix=export_prefix, work_dir=work_dir, **args)
    commands = list(getattr(module, "VERIFIER_COMMANDS", module.COMMANDS) if settings.get("verifier")
                    else module.COMMANDS)
    if getattr(module, "PROTOCOL", "methods") == "rest":
        handler = rest_handler(instance, commands)
    else:
        handler = methods_handler(module, instance, commands, settings.get("defaults", {}))
    print(f"[simulator] ready on port {port}, commands {commands}"
          + (f", settings {json.dumps(settings)}" if settings else ""), flush=True)
    ThreadingHTTPServer(("127.0.0.1", int(port)), handler).serve_forever()


def methods_handler(module, instance, commands, defaults):
    handlers = {name: getattr(instance, name) for name in commands}

    def simulate_network(**params):
        if "parameter_set" in defaults:
            params["parameter_set"] = defaults["parameter_set"]
        for key in MERGED_SETTINGS:
            if key in defaults:
                params[key] = {**defaults[key], **(module.coerce(params.get(key), key, dict) or {})}
        return instance.simulate_network(**params)

    if defaults:
        handlers["simulate_network"] = simulate_network

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            method = body["method"]
            if method == "__ping__":
                reply = {"success": True, "result": "pong"}
            elif method == "__hosts__":
                reply = {"success": True, "result": list(getattr(instance, "extra_hosts", list)())}
            elif method not in handlers:
                reply = {"success": False, "error": "request rejected"}
            else:
                try:
                    reply = {"success": True, "result": handlers[method](**body.get("params", {}))}
                except Exception as exc:
                    traceback.print_exc()
                    reply = {"success": False, "error": str(exc)}
            data = json.dumps(reply, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.passthrough("GET")

        def do_HEAD(self):
            self.passthrough("HEAD")

        def passthrough(self, verb):
            if not (self.path.startswith(ARCHIVE) and hasattr(instance, "passthrough")):
                self.send_response(404)
                self.end_headers()
                return
            try:
                status, headers, body = instance.passthrough(verb, unquote(self.path[len(ARCHIVE) - 1:]),
                                                             dict(self.headers))
            except Exception as exc:
                traceback.print_exc()
                status, headers, body = 502, {"Content-Type": "text/plain"}, str(exc).encode()
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            if "Content-Length" not in headers:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if verb != "HEAD":
                self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def rest_handler(instance, commands):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def reply(self, status, payload):
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ok", "commands": list(commands)})
            else:
                self.reply(404, {"error": f"no endpoint {self.path}"})

        def do_POST(self):
            command = self.path.lstrip("/")
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) or b"{}"
            if command == "" and raw == PING:
                self.reply(200, {"success": True, "result": "pong"})
                return
            if command == "" and raw == HOSTS:
                self.reply(200, {"success": True, "result": list(getattr(instance, "extra_hosts", list)())})
                return
            if command not in commands:
                self.reply(404, {"error": f"unknown command {command!r}; "
                                          f"available: {list(commands)}"})
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                self.reply(400, {"error": f"body is not valid JSON: {exc}"})
                return
            try:
                result = getattr(instance, command)(**payload)
                self.reply(200, result)
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-3000:]})

    return Handler


if __name__ == "__main__":
    fire.Fire(serve)
