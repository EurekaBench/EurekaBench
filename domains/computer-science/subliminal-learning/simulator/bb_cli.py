import json
import os
import sys
import urllib.request

import fire

SERVER_URL = os.environ.get("BB_SERVER_URL", "http://127.0.0.1:9100")


def call_server(method, **kwargs):
    payload = json.dumps({"method": method, "params": kwargs}).encode()
    req = urllib.request.Request(SERVER_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        print(json.dumps(data, indent=2, default=str))
        sys.exit(1)
    return data.get("result", data)


class CLI:
    def generate(self, example_id=None, prompt=None, model_id=None,
                 max_new_tokens=256, temperature=1.0, num_samples=1):
        return call_server("generate", example_id=example_id, prompt=prompt, model_id=model_id,
                           max_new_tokens=max_new_tokens,
                           temperature=temperature, num_samples=num_samples)

    def get_logits(self, example_id=None, prompt=None, model_id=None, top_k=10):
        return call_server("get_logits", example_id=example_id, prompt=prompt,
                           model_id=model_id, top_k=top_k)

    def finetune(self, data_path, name=None):
        return call_server("finetune", data_path=data_path, name=name)

    def list_ckpts(self):
        return call_server("list_ckpts")


if __name__ == "__main__":
    fire.Fire(CLI, serialize=lambda x: json.dumps(x, indent=2, default=str))
