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
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        print(json.dumps(data, indent=2, default=str))
        sys.exit(1)
    return data.get("result", data)


class CLI:
    def list_simulations(self):
        return call("list_simulations")

    def get_params(self):
        return call("get_params")

    def get_group_catalog(self, sim_name, snapshot=90):
        return call("get_group_catalog", sim_name=sim_name, snapshot=snapshot)


if __name__ == "__main__":
    fire.Fire(CLI, serialize=lambda x: json.dumps(x, indent=2, default=str))
