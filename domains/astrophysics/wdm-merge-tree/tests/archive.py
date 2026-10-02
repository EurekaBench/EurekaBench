import json
import os
import re
import urllib.request
from pathlib import Path

URL = os.environ["EUREKA_SIMULATOR_URL"]
KEY = os.environ["EUREKA_SIMULATOR_KEY"]
CATALOG_ROOT = "/FOF_Subfind"


def trailing_int(name):
    m = re.search(r"(\d+)$", Path(name).stem)
    return int(m.group(1)) if m else None


def call(method, **params):
    request = urllib.request.Request(URL, data=json.dumps({"method": method, "params": params}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=7200) as response:
        reply = json.loads(response.read())
    if not reply.get("success"):
        raise RuntimeError(reply.get("error", "the simulator rejected the request"))
    return reply.get("result")


class Auth:
    def get_authorization_header(self):
        return f"Bearer {KEY}"


class Archive:
    def __init__(self, data_dir=None, **settings):
        self.data_dir = call("container_data_dir")
        self.https_server = f"{URL}/archive"
        self.https_auth = Auth()
        self.tc = None

    def connect(self):
        pass

    def __getattr__(self, name):
        def method(*args, **kwargs):
            return call("call", key=KEY, name=name, args=list(args), kwargs=kwargs)
        return method


DreamsSimulator = DreamsBlackBox = BasicCamelsBlackBox = Archive
