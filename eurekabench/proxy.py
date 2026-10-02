import http.client
import json
import os
import re
import ssl
import threading
import time
import unicodedata
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PLACEHOLDER_KEY = "sk-acp-proxy-placeholder"
SOURCES_INCLUDE = "web_search_call.action.sources"
STOP_WORDS = {"a", "an", "the", "of", "in", "from", "for", "and", "to", "on", "by", "with", "at", "is", "are", "its",
              "their", "that", "this", "via", "using", "into", "as", "or"}
VARIANTS = {"modelling": "modeling", "behaviour": "behavior", "behavioural": "behavioral", "organisation": "organization",
            "caenorhabditis": "c", "brainwide": "brain wide", "wholebrain": "whole brain", "nematode": "worm"}
GENERIC = ("c", "elegans", "caenorhabditis", "worm", "nervous", "system", "neural", "neuronal", "neuron", "activity",
           "brain", "dynamics", "motor", "behavior", "model", "modeling", "network", "population", "locomotion",
           "moving", "movement", "framework", "quantitative", "state", "data", "whole", "interneuron", "circuit",
           "navigation", "chemotaxis", "chemotactic", "sensory", "olfactory", "learning", "oscillator", "oscillation",
           "oscillatory", "control", "controlling", "memory", "short", "term", "simple", "integrated", "integrative",
           "simulating", "simulation", "biophysical", "biological", "biology", "based", "driven", "method", "body",
           "environment", "interaction", "analysis", "evolution", "functional", "organization", "operation", "premotor",
           "switching", "strategy", "variability", "feedback", "reinforcement", "deep", "agent", "drive", "discovering",
           "capturing", "continuous", "complexity", "context", "dependent", "underlies", "minimal", "generates",
           "human", "cortex", "cortical", "visual", "vision", "auditory", "audition", "audiovisual", "multisensory",
           "language", "linguistic", "speech", "narrative", "comprehension", "response", "fmri", "bold", "encoding",
           "prediction", "predictive", "predicts", "processing", "hierarchy", "hierarchical", "natural", "scene", "image",
           "object", "dataset", "stimulus", "stimuli", "representation", "distributed", "integration", "semantic",
           "spatial", "field", "regime", "foundation", "large", "scale", "recognition", "convolutional", "level", "high",
           "mid", "ventral", "stream", "sites", "detection", "effect", "categorization", "temporal", "superior", "sulcus",
           "massive", "bridge", "cognitive", "artificial", "intelligence", "computational", "neuroscience", "silico",
           "coding", "listening", "organization", "principles", "functional", "occipitotemporal", "dimensions",
           "properties", "system", "information", "updating", "cross",
           "tokamak", "plasma", "fusion", "triangularity", "negative", "positive", "confinement", "transport", "mode",
           "edge", "core", "pedestal", "ballooning", "stability", "instability", "shape", "shaped", "diverted",
           "magnetohydrodynamics", "mhd", "performance", "power", "pilot", "plant", "physics", "basis", "reactor",
           "relevant", "normalized", "density", "current", "pressure", "access", "equilibrium", "equilibria", "code",
           "tool", "design", "axisymmetric", "device", "simulator", "time", "open", "source", "toolkit", "engineering",
           "education", "differentiable", "fast", "experimental", "progress", "future", "prospect", "enhanced",
           "achievement", "avoidance", "localized", "gradient", "formation", "simultaneous", "strongly", "path",
           "overview", "brief", "history", "generalized", "perturbed", "ideal", "grad", "shafranov",
           "ice", "shelf", "shelves", "antarctic", "flow", "law", "viscosity", "rheology", "catchment", "streamflow",
           "rainfall", "runoff", "hydrological", "hydrologic", "hydrology", "conceptual", "parameter", "parameterization",
           "calibration", "hybrid", "regionalized", "process", "prediction", "accuracy", "watershed", "meteorology",
           "attributes", "contiguous", "usa", "characteristics", "assessment", "regional", "variability", "interpretability",
           "generalization", "learnable", "outputs", "physical", "approach", "state", "art",
           "dark", "matter", "galaxy", "galaxies", "galactic", "milky", "way", "halo", "halos", "haloes", "feedback",
           "baryonic", "baryon", "baryons", "simulation", "simulations", "simulated", "cosmology", "cosmological",
           "warm", "cold", "particle", "mass", "masses", "merger", "mergers", "tree", "trees", "history",
           "deep", "learning", "network", "networks", "neural", "project", "suite", "suites", "zoom", "zooms", "box",
           "boxes", "environment", "environments", "environmental", "impact", "local", "evolution", "gravitational",
           "wave", "waves", "background", "black", "hole", "holes", "growth", "massive", "supermassive", "properties",
           "property", "profile", "profiles", "density", "different", "new", "one", "via", "camels", "dreams",
           "illustristng", "tng", "simba", "astrid", "swift", "eagle", "eagles", "subhalo", "subhalos", "subhalos",
           "stellar", "star", "stars", "gas", "wind", "winds", "supernova", "supernovae", "agn", "omega",
           "sigma", "redshift", "catalog", "catalogs", "catalogue", "catalogues", "snapshot", "snapshots",
           "subfind", "sublink", "fof", "friends", "globus", "flatiron", "hydrodynamic", "hydrodynamical",
           "zoom-in", "nbody", "twin", "twins", "gravity", "only", "assembly", "progenitor", "progenitors",
           "descendant", "branch", "nanohertz", "pulsar", "timing", "array", "nanograv", "amplitude", "physics",
           "uncertainty", "analytic", "formula", "relating", "impact", "across", "with", "energy", "speed",
           "efficiency", "concentration", "contraction",
           "adiabatic", "core", "cusp", "profile", "inner", "outer", "virial", "radius", "velocity", "dispersion",
           "metallicity", "compactness", "formation", "rate", "sfr", "quenching", "quenched", "satellite",
           "satellites", "central", "centrals", "cgm", "circumgalactic", "medium", "fraction", "fractions",
           "overdense", "underdense", "overdensity", "underdensity", "neighbour", "neighbor", "nearest",
           "belief", "change", "mechanistic", "benchmark", "through")
SURNAME_LETTERS = 4
HELD_ITEMS = ("web_search_call", "message")
NOTICE = ("[EurekaBench sandbox] This response was stopped before it reached you: a web search, a page or a citation "
          "in it named one of the source studies (or the simulator's paper) that this task keeps out of reach. That "
          "material is not available here; continue with the simulator and your own analysis.")
NO_USAGE = {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 0}
INTERNAL_KEYS = ("internal_chat_message_metadata_passthrough",)
ANTHROPIC_WEB_PREFIXES = ("web_search_", "web_fetch_")
OPENAI_WEB_TOOLS = ("web_search", "web_search_2025_08_26")
UNFILTERABLE_WEB_TOOLS = ("web_search_preview", "web_search_preview_2025_03_11")
OPENAI_DOMAIN_LIMIT = 100
DROPPED_HEADERS = ("host", "content-length", "connection", "keep-alive", "transfer-encoding",
                   "accept-encoding")


def upstream(agent, model):
    if model.startswith("litellm_proxy/"):
        return os.environ["LLM_API_KEY"], os.environ["LLM_BASE_URL"], model[len("litellm_proxy/"):]
    if agent == "codex":
        return os.environ["OPENAI_API_KEY"], "https://api.openai.com/v1", model
    return os.environ["ANTHROPIC_API_KEY"], "https://api.anthropic.com", model


def drop_internal_keys(node):
    dropped = False
    if isinstance(node, dict):
        for key in INTERNAL_KEYS:
            if node.pop(key, None) is not None:
                dropped = True
        for value in node.values():
            dropped = drop_internal_keys(value) or dropped
    elif isinstance(node, list):
        for value in node:
            dropped = drop_internal_keys(value) or dropped
    return dropped


def domain_blocked(domain, blocked_domains):
    domain = str(domain).lower().strip().strip(".")
    return any(domain == b or domain.endswith("." + b) for b in blocked_domains)


def restrict_domains(holder, blocked_domains, limit=None):
    allowed = holder.get("allowed_domains")
    if allowed:
        kept = [d for d in allowed if not domain_blocked(d, blocked_domains)]
        if kept:
            holder["allowed_domains"] = kept
            return
        holder.pop("allowed_domains")
    merged = sorted(set(holder.get("blocked_domains") or []) | set(blocked_domains))
    if limit is not None and len(merged) > limit:
        raise ValueError(f"this web tool takes at most {limit} blocked domains and the run names "
                         f"{len(merged)}")
    holder["blocked_domains"] = merged


def restrict_web_tools(payload, blocked_domains):
    restricted = False
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        kind = str(tool.get("type", ""))
        if kind in UNFILTERABLE_WEB_TOOLS:
            raise ValueError(f"{kind} takes no domain filter, so the blocked hosts cannot travel with it")
        if kind in OPENAI_WEB_TOOLS:
            restrict_domains(tool.setdefault("filters", {}), blocked_domains, OPENAI_DOMAIN_LIMIT)
        elif kind.startswith(ANTHROPIC_WEB_PREFIXES):
            restrict_domains(tool, blocked_domains)
        else:
            continue
        restricted = True
    return restricted


def canonical(word):
    word = VARIANTS.get(word, word)
    if len(word) >= 5 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) >= 5 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return VARIANTS.get(word, word)


def words(text):
    text = unicodedata.normalize("NFKD", urllib.parse.unquote_plus(str(text))).encode("ascii", "ignore").decode()
    text = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", re.sub(r"[^a-z0-9]+", " ", text.lower()))
    return [c for w in text.split() for c in canonical(w).split()]


def significant(text):
    return [w for w in words(text) if w not in STOP_WORDS]


def padded(sequence):
    return f" {' '.join(sequence)} "


GENERIC_WORDS = {c for w in GENERIC for c in canonical(w).split()}


def literature_hit(text, literature):
    seen = words(text)
    present, joined, joined_sig = set(seen), padded(seen), padded([w for w in seen if w not in STOP_WORDS])
    for paper in literature:
        label = f"{paper['authors'][0]} {paper['years'][0]}"
        for ident in paper["ids"]:
            if padded(words(ident)) in joined:
                return f"{label}: {ident}"
        for phrase in paper.get("phrases", ()):
            if padded(significant(phrase)) in joined_sig:
                return f"{label}: '{phrase}'"
        authors = [words(a)[-1] for a in paper["authors"]]
        authors = [a for a in authors if len(a) >= SURNAME_LETTERS] or authors
        named = [a for a in authors if a in present]
        dated = any(str(y) in present for y in paper["years"])
        specific = set()
        for title in paper["titles"]:
            title = significant(title)
            rare = [w not in GENERIC_WORDS and len(w) >= 3 and not w.isdigit() for w in title]
            specific |= {w for w, r in zip(title, rare) if r}
            grams = [title[i:i + 3] for i in range(len(title) - 2) if any(rare[i:i + 3])]
            grams += [title[i:i + 2] for i in range(len(title) - 1) if all(rare[i:i + 2]) or (named and any(rare[i:i + 2]))]
            if any(padded(g) in joined_sig for g in grams):
                return f"{label}: title"
            if len(title) >= 5 and sum(w in present for w in title) >= 0.7 * len(title):
                return f"{label}: title"
        named = set(named)
        if len(named) >= 3 or (len(named) >= 2 and named & {authors[0], authors[-1]}):
            return f"{label}: authors {sorted(named)}"
        if dated and (authors[0] in present or authors[-1] in present):
            return f"{label}: author and year"
        if dated and any(padded(significant(v)) in joined_sig for v in paper.get("venues", ())) and (
                named or present & specific or present & {"elegan", "worm"}):
            return f"{label}: venue and year"
    return None


def without(text, vocabulary):
    seen = words(text)
    for phrase in vocabulary:
        target, kept, i = words(phrase), [], 0
        while i < len(seen):
            if target and seen[i:i + len(target)] == target:
                i += len(target)
            else:
                kept.append(seen[i])
                i += 1
        seen = kept
    return " ".join(seen)


def item_texts(item, vocabulary=()):
    action = item.get("action") or {}
    texts = [action.get(k) for k in ("query", "url", "pattern")] + list(action.get("queries") or [])
    texts += [s.get("url") for s in (action.get("sources") or item.get("sources") or []) if isinstance(s, dict)]
    for part in item.get("content") or []:
        if isinstance(part, dict):
            text = part.get("text")
            texts.append(without(text, vocabulary) if text and vocabulary else text)
            texts += [a.get(k) for a in part.get("annotations") or [] if isinstance(a, dict) for k in ("url", "title")]
    return [t for t in texts if t]


class Filter:
    def __init__(self, literature, record, vocabulary=()):
        self.literature, self.record, self.vocabulary = literature, record, vocabulary
        self.pending, self.held, self.response_id, self.stopped = b"", {}, "resp_sandbox", False

    def feed(self, chunk):
        self.pending += chunk
        out = []
        while not self.stopped and (cut := self.pending.find(b"\n\n")) >= 0:
            event, self.pending = self.pending[:cut + 2], self.pending[cut + 2:]
            out.append(self.handle(event))
        return b"".join(out)

    def handle(self, raw):
        lines = raw.decode("utf-8", "replace").replace("\r\n", "\n").split("\n")
        try:
            event = json.loads("\n".join(line[5:].strip() for line in lines if line.startswith("data:")))
        except ValueError:
            return raw
        if not isinstance(event, dict):
            return raw
        kind, index, item = event.get("type", ""), event.get("output_index"), event.get("item") or {}
        if kind == "response.created":
            self.response_id = (event.get("response") or {}).get("id", self.response_id)
        if kind == "response.output_item.added" and item.get("type") in HELD_ITEMS:
            self.held[index] = [raw]
            return b""
        if index in self.held:
            self.held[index].append(raw)
            if kind != "response.output_item.done":
                return b""
            for text in item_texts(item, self.vocabulary):
                if hit := literature_hit(text, self.literature):
                    self.stopped = True
                    self.record(f"stopped {item.get('type')} naming {hit}: {str(text)[:200]!r}")
                    del self.held[index]
                    return self.notice(index)
            return b"".join(self.held.pop(index))
        if kind == "response.completed":
            return b"".join(b"".join(self.held.pop(i)) for i in sorted(self.held)) + raw
        return raw

    def notice(self, index):
        message = {"type": "message", "id": "msg_sandbox", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": NOTICE, "annotations": []}]}
        events = [{"type": "response.output_item.done", "output_index": index, "item": message},
                  {"type": "response.completed", "response": {"id": self.response_id, "object": "response",
                                                              "status": "completed", "output": [message],
                                                              "usage": NO_USAGE}}]
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def start(port, agent, model, blocked_domains=(), literature=(), log=None, vocabulary=()):
    real_key, base_url, force_model = upstream(agent, model)
    blocked = sorted({str(d).lower() for d in blocked_domains})
    papers = list(literature)
    vocabulary = list(vocabulary)

    def record(line):
        if log is not None:
            with open(log, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    base = urllib.parse.urlparse(base_url)
    https = base.scheme != "http"
    host = base.hostname
    host_port = base.port or (443 if https else 80)
    prefix = base.path.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def read_body(self):
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                chunks = []
                while True:
                    size = int(self.rfile.readline().strip().split(b";")[0] or b"0", 16)
                    if size == 0:
                        while self.rfile.readline().strip():
                            pass
                        return b"".join(chunks)
                    chunks.append(self.rfile.read(size))
                    self.rfile.read(2)
            length = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(length) if length else None

        def forward(self):
            body = self.read_body()
            payload, searching = None, False
            if body and self.command == "POST":
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    edited = drop_internal_keys(payload)
                    tools = [t for t in payload.get("tools") or [] if isinstance(t, dict)]
                    searching = any(str(t.get("type", "")) in OPENAI_WEB_TOOLS for t in tools)
                    try:
                        if blocked and restrict_web_tools(payload, blocked):
                            edited = True
                            record(f"web tools {[t.get('type') for t in payload['tools'] if 'web' in str(t.get('type'))]} "
                                   f"carry {len(blocked)} blocked hosts")
                    except ValueError as exc:
                        record(f"refused: {exc}")
                        self.send_error(400, "the blocked hosts cannot be enforced", str(exc))
                        return
                    if searching and papers and SOURCES_INCLUDE not in (payload.get("include") or []):
                        payload["include"] = list(payload.get("include") or []) + [SOURCES_INCLUDE]
                        edited = True
                    if "model" in payload and payload["model"] != force_model:
                        payload["model"] = force_model
                        edited = True
                    if edited:
                        body = json.dumps(payload).encode()
            headers = {}
            for key, value in self.headers.items():
                lowered = key.lower()
                if lowered in DROPPED_HEADERS:
                    continue
                if lowered == "x-api-key":
                    value = real_key
                elif lowered == "authorization":
                    value = f"Bearer {real_key}"
                headers[key] = value
            headers["Host"] = host
            def send(body):
                if https:
                    connection = http.client.HTTPSConnection(host, host_port, timeout=600,
                                                             context=ssl.create_default_context())
                else:
                    connection = http.client.HTTPConnection(host, host_port, timeout=600)
                connection.request(self.command, prefix + self.path, body=body, headers=headers)
                return connection, connection.getresponse()

            connection = None
            try:
                connection, response = send(body)
                answer = None
                if response.status == 400 and searching and papers and SOURCES_INCLUDE in body.decode("utf-8", "replace"):
                    answer = response.read()
                    connection.close()
                    if b"include" in answer:
                        record(f"the upstream refuses {SOURCES_INCLUDE}; the addresses of search results go unchecked")
                        payload["include"].remove(SOURCES_INCLUDE)
                        connection, response = send(json.dumps(payload).encode())
                        answer = None
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in ("connection", "keep-alive", "transfer-encoding", "content-length"):
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                if answer is not None:
                    self.wfile.write(answer)
                    return
                streaming = "text/event-stream" in (response.getheader("Content-Type") or "")
                guard = Filter(papers, record, vocabulary) if searching and papers and streaming else None
                while chunk := response.read1(8192):
                    if guard:
                        chunk = guard.feed(chunk)
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    if guard and guard.stopped:
                        break
            except Exception as exc:
                try:
                    self.send_error(502, str(exc))
                except Exception:
                    pass
            finally:
                if connection is not None:
                    connection.close()

        do_GET = do_POST = do_PUT = do_DELETE = forward

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()
