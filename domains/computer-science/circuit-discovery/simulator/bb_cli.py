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
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        print(json.dumps(data, indent=2, default=str))
        sys.exit(1)
    return data.get("result", data)


def as_list(value):
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


class CLI:
    def load_model(self):
        return call("load_model")

    def load_graph(self):
        return call("load_graph")

    def load_circuit(self, circuit_file):
        return call("load_circuit", circuit_file=circuit_file)

    def run_forward_with_steering(self, prompt, steering_components, steering_values_file):
        return call("run_forward_with_steering", prompt=prompt,
                    steering_components=as_list(steering_components),
                    steering_values_file=steering_values_file)



if __name__ == "__main__":
    fire.Fire(CLI, serialize=lambda x: json.dumps(x, indent=2, default=str))
