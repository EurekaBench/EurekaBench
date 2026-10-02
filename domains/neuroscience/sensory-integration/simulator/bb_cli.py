import json
import os
import sys
import urllib.request

import fire


SERVER_URL = os.environ.get("BB_SERVER_URL", "http://127.0.0.1:9100")


def call(method, **kwargs):
    payload = json.dumps({"method": method, "params": kwargs}).encode()
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        print(json.dumps(data, indent=2, default=str))
        sys.exit(1)
    return data.get("result", data)


class CLI:
    def get_events(self, input_path, output_name="events", transcribe=True):
        return call("get_events", input_path=input_path, output_name=output_name,
                    transcribe=transcribe)

    def simulate(self, input_path, output_name="run1", transcribe=True):
        return call("simulate", input_path=input_path, output_name=output_name,
                    transcribe=transcribe)


if __name__ == "__main__":
    fire.Fire(CLI, serialize=lambda x: json.dumps(x, indent=2, default=str))
