"""OpenAI-compatible chat client (streaming) + llama-server lifecycle management."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import httpx

from .config import Config, STATE_DIR

PID_FILE = STATE_DIR / "llama-server.pid"
LOG_FILE = STATE_DIR / "llama-server.log"
INFO_FILE = STATE_DIR / "llama-server.json"


def _files(tier: str) -> tuple[Path, Path, Path]:
    """(pid, log, info) files for a managed server tier: 'main' or 'fast'."""
    if tier == "main":
        return PID_FILE, LOG_FILE, INFO_FILE
    return STATE_DIR / f"llama-server-{tier}.pid", STATE_DIR / f"llama-server-{tier}.log", STATE_DIR / f"llama-server-{tier}.json"


class BackendError(RuntimeError):
    pass


@dataclass
class StreamResult:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # native API tool calls, if the server parsed any
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, base_url: str, model_name: str = "", temperature: float = 0.2, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name or "default"
        self.temperature = temperature
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    # -------------------------------------------------------------- health
    def health(self) -> str:
        """'up' | 'loading' | 'down'."""
        try:
            r = self._http.get(self.base_url + "/health", timeout=2.0)
            if r.status_code == 200:
                return "up"
            if r.status_code == 503:  # llama.cpp while the model loads
                return "loading"
        except httpx.HTTPError:
            pass
        try:  # Ollama / LM Studio / vLLM
            r = self._http.get(self.base_url + "/v1/models", timeout=2.0)
            return "up" if r.status_code == 200 else "down"
        except httpx.HTTPError:
            return "down"

    def is_up(self) -> bool:
        return self.health() == "up"

    def wait_until_up(self, progress: Callable[[str], None] | None = None, timeout: float = 180.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self.health()
            if h == "up":
                return True
            if h == "down":
                return False
            if progress:
                progress(f"model is still loading at {self.base_url}… {time.time() - t0:.0f}s")
            time.sleep(0.5)
        return False

    def server_props(self) -> dict:
        """llama.cpp exposes /props (model path, n_ctx). Others: fall back to /v1/models."""
        try:
            r = self._http.get(self.base_url + "/props", timeout=3.0)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, ValueError):
            pass
        try:
            r = self._http.get(self.base_url + "/v1/models", timeout=3.0)
            if r.status_code == 200:
                return {"models": r.json().get("data", [])}
        except (httpx.HTTPError, ValueError):
            pass
        return {}

    def describe(self) -> str:
        p = self.server_props()
        if "model_path" in p:
            return Path(p["model_path"]).stem
        if "model_alias" in p and p["model_alias"]:
            return p["model_alias"]
        ms = p.get("models") or []
        if ms:
            return ms[0].get("id", "?")
        return self.model_name

    def n_ctx(self, default: int) -> int:
        p = self.server_props()
        try:
            return int(p["default_generation_settings"]["n_ctx"])
        except (KeyError, TypeError, ValueError):
            return default

    # -------------------------------------------------------------- chat
    def stream_chat(self, messages: list[dict], max_tokens: int = 2048, stop: list[str] | None = None,
                    temperature: float | None = None, extra: dict | None = None) -> Iterator[tuple[str, object]]:
        """Yields ("text", delta) events, then one ("done", StreamResult)."""
        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
        }
        if stop:
            body["stop"] = stop
        if extra:
            body.update(extra)
        res = StreamResult()
        native_calls: dict[int, dict] = {}
        try:
            with self._http.stream("POST", self.base_url + "/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    r.read()
                    raise BackendError(f"{r.status_code} from {self.base_url}: {r.text[:400]}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "usage" in chunk and chunk["usage"]:
                        res.usage = chunk["usage"]
                    if "timings" in chunk and chunk["timings"]:
                        res.timings = chunk["timings"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        txt = delta.get("content")
                        if txt:
                            res.content += txt
                            yield ("text", txt)
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = native_calls.setdefault(idx, {"name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                        if ch.get("finish_reason"):
                            res.finish_reason = ch["finish_reason"]
        except httpx.ReadTimeout as e:
            raise BackendError("model timed out generating a response") from e
        except httpx.HTTPError as e:
            raise BackendError(f"cannot reach {self.base_url}: {e}") from e
        for idx in sorted(native_calls):
            res.tool_calls.append(native_calls[idx])
        yield ("done", res)

    def complete(self, messages: list[dict], **kw) -> str:
        out = ""
        for kind, payload in self.stream_chat(messages, **kw):
            if kind == "text":
                out += payload
        return out


# ------------------------------------------------------------------ model discovery
@dataclass
class ModelFile:
    path: Path
    size: int

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def size_gb(self) -> float:
        return self.size / 1e9


def find_models(dirs: list[str]) -> list[ModelFile]:
    seen: dict[Path, ModelFile] = {}
    for d in dirs:
        p = Path(os.path.expanduser(d))
        if not p.is_dir():
            continue
        for f in p.rglob("*.gguf"):
            try:
                st = f.stat()
            except OSError:
                continue
            if st.st_size < 50_000_000:  # skip mmproj / vocab shards
                continue
            rp = f.resolve()
            if rp not in seen:
                seen[rp] = ModelFile(path=f, size=st.st_size)
    return sorted(seen.values(), key=lambda m: m.size)


def system_ram() -> int:
    try:
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 16 * 1024**3


def pick_model(cfg: Config, override: str = "") -> ModelFile:
    spec = override or cfg.server.model
    if spec and Path(os.path.expanduser(spec)).is_file():
        p = Path(os.path.expanduser(spec))
        return ModelFile(path=p, size=p.stat().st_size)
    models = find_models(cfg.server.model_dirs)
    if not models:
        raise BackendError(
            "no .gguf models found in " + ", ".join(cfg.server.model_dirs)
            + "\nPut a model there or set [server].model in " + str(cfg.path)
        )
    if spec:
        hits = [m for m in models if spec.lower() in m.path.name.lower()]
        if not hits:
            raise BackendError(f"no model matching {spec!r}; have: " + ", ".join(m.name for m in models))
        return hits[-1]  # largest match
    budget = system_ram() / 2.5   # 24 GB → 14B q4 fits; 16 GB → 7B; 8 GB → 3B
    fitting = [m for m in models if m.size <= budget]
    return fitting[-1] if fitting else models[0]


def find_chat_template(cfg: Config, model: ModelFile) -> Path | None:
    if cfg.server.chat_template:
        p = Path(os.path.expanduser(cfg.server.chat_template))
        return p if p.is_file() else None
    # Convention: a *tools*.jinja next to the model, or one sharing its family prefix.
    for cand in sorted(model.path.parent.glob("*.jinja")):
        if "tool" in cand.name.lower():
            return cand
    return None


# ------------------------------------------------------------------ server lifecycle
def _read_pid(tier: str = "main") -> int | None:
    try:
        pid = int(_files(tier)[0].read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return pid


def server_info(tier: str = "main") -> dict:
    try:
        return json.loads(_files(tier)[2].read_text())
    except (OSError, ValueError):
        return {}


def start_server(cfg: Config, model: ModelFile, progress: Callable[[str], None] | None = None,
                 wait: float = 180.0, tier: str = "main", port: int | None = None) -> LLMClient:
    import shutil

    pid_file, log_file, info_file = _files(tier)
    port = port or cfg.server.port
    exe = shutil.which("llama-server")
    if not exe:
        raise BackendError("llama-server not found on PATH. Install llama.cpp (brew install llama.cpp) or point\n"
                           "[server].base_url at a running Ollama/LM Studio and set manage = false.")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    cmd = [exe, "--model", str(model.path), "--alias", model.name, "--host", "127.0.0.1", "--port", str(port),
           "--ctx-size", str(cfg.server.ctx_size), "--parallel", "1", "--flash-attn", "on", "--cache-reuse", "256",
           "--n-gpu-layers", "999", "--jinja", "--reasoning-format", "none"]
    tpl = find_chat_template(cfg, model)
    if tpl:
        cmd += ["--chat-template-file", str(tpl)]
    cmd += list(cfg.server.extra_args)
    log = open(log_file, "ab")
    log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    pid_file.write_text(str(proc.pid))
    info_file.write_text(json.dumps({"pid": proc.pid, "model": str(model.path), "base_url": base_url,
                                     "started": time.time(), "cmd": cmd, "tier": tier}))
    client = LLMClient(base_url, model.name, cfg.agent.temperature)
    t0 = time.time()
    while time.time() - t0 < wait:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log_file.read_text(errors="replace")[-1500:]
            except OSError:
                pass
            raise BackendError(f"llama-server exited with code {proc.returncode}. Log tail:\n{tail}")
        if client.is_up():
            return client
        if progress:
            progress(f"loading {model.name} ({model.size_gb:.1f} GB) into memory… {time.time() - t0:.0f}s")
        time.sleep(0.4)
    raise BackendError(f"llama-server did not become ready within {wait:.0f}s; see {log_file}")


def stop_server(tier: str = "main") -> bool:
    pid = _read_pid(tier)
    pid_file = _files(tier)[0]
    if pid is None:
        try:
            pid_file.unlink()
        except OSError:
            pass
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(50):
        if _read_pid(tier) is None:
            break
        time.sleep(0.1)
    try:
        pid_file.unlink()
    except OSError:
        pass
    return True


def fast_tier_enabled(cfg: Config) -> tuple[bool, str]:
    """Resolve [fast].enabled. "auto" means: only if main + fast models fit in half of RAM."""
    e = cfg.fast.enabled
    if e is True or (isinstance(e, str) and e.lower() == "true"):
        return True, "forced on"
    if e is False or (isinstance(e, str) and e.lower() == "false"):
        return False, "disabled"
    try:
        main = pick_model(cfg)
        fast = pick_model(cfg, cfg.fast.model)
    except BackendError as ex:
        return False, str(ex).splitlines()[0]
    need, have = main.size + fast.size, system_ram() / 2
    if need <= have:
        return True, f"auto: {need / 1e9:.1f} GB of models fits in half of RAM"
    return False, f"auto: {main.name} + {fast.name} = {need / 1e9:.1f} GB > half of RAM ({have / 1e9:.0f} GB); set \\[fast].enabled = true to force"


def connect_fast(cfg: Config, progress: Callable[[str], None] | None = None) -> LLMClient | None:
    """Client for the fast tier, starting it if needed. None when disabled or unavailable (caller falls
    back to the main model)."""
    if not fast_tier_enabled(cfg)[0]:
        return None
    base_url = f"http://127.0.0.1:{cfg.fast.port}"
    client = LLMClient(base_url, "", cfg.agent.temperature)
    state = client.health()
    if state == "loading":
        if progress:
            progress("fast model is loading…")
        return client if client.wait_until_up(progress) else None
    if state == "up":
        return client
    try:
        model = pick_model(cfg, cfg.fast.model)
    except BackendError:
        return None
    if progress:
        progress(f"starting fast model {model.name}…")
    try:
        return start_server(cfg, model, progress=progress, tier="fast", port=cfg.fast.port)
    except BackendError:
        return None


def connect(cfg: Config, model_override: str = "", url_override: str = "",
            progress: Callable[[str], None] | None = None) -> LLMClient:
    """Return a client for a live server, starting llama-server if allowed and necessary."""
    base_url = url_override or cfg.server.base_url
    client = LLMClient(base_url, cfg.server.model_name, cfg.agent.temperature)
    state = client.health()
    if state == "loading":
        if progress:
            progress("model is loading…")
        if not client.wait_until_up(progress):
            raise BackendError(f"a server at {base_url} never finished loading; see {LOG_FILE}")
        state = "up"
    if state == "up" and not model_override:
        return client
    if state == "down" and _read_pid():
        # Stale pid file (crashed server) – clean up so we can start a fresh one.
        try:
            PID_FILE.unlink()
        except OSError:
            pass
    if model_override and client.is_up():
        # A server is up but the user asked for a specific model: only restart if we own it.
        info = server_info()
        if _read_pid() and info.get("base_url") == base_url:
            want = pick_model(cfg, model_override)
            if Path(info.get("model", "")) != want.path:
                stop_server()
            else:
                return client
        else:
            raise BackendError(f"a server we don't manage is already running at {base_url}; "
                               f"stop it or change [server].port to use --model")
    if not cfg.server.manage and not model_override:
        raise BackendError(f"nothing is listening at {base_url} and [server].manage is false")
    if url_override and url_override != f"http://127.0.0.1:{cfg.server.port}":
        raise BackendError(f"nothing is listening at {url_override}")
    model = pick_model(cfg, model_override)
    if progress:
        progress(f"starting llama-server with {model.name}…")
    client = start_server(cfg, model, progress=progress)
    return client
