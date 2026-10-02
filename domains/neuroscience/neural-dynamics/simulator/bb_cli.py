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
    def get_connectome(self, output_name="connectome"):
        return call("get_connectome", output_name=output_name)

    def simulate_network(self, cells=None, muscles=None, stimuli=None, parameter_set="C1",
                         duration=500.0, dt=0.05, save_every_ms=None,
                         remove_connections=None, keep_only_connections=None,
                         connection_number_override=None, connection_number_scaling=None,
                         connection_polarity_override=None, param_overrides=None,
                         data_reader=None, output_name="sim1"):
        return call("simulate_network", cells=cells, muscles=muscles, stimuli=stimuli,
                    parameter_set=parameter_set, duration=duration, dt=dt,
                    save_every_ms=save_every_ms,
                    remove_connections=remove_connections,
                    keep_only_connections=keep_only_connections,
                    connection_number_override=connection_number_override,
                    connection_number_scaling=connection_number_scaling,
                    connection_polarity_override=connection_polarity_override,
                    param_overrides=param_overrides, data_reader=data_reader,
                    output_name=output_name)

    def simulate_body(self, duration_ms=50.0, output_name="body1"):
        return call("simulate_body", duration_ms=duration_ms, output_name=output_name)


if __name__ == "__main__":
    fire.Fire(CLI, serialize=lambda x: json.dumps(x, indent=2, default=str))
