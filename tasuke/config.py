"""Configuration: ~/.config/tasuke/config.toml with sane defaults, plus XDG paths."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tasuke"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tasuke"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "tasuke"
CONFIG_PATH = CONFIG_DIR / "config.toml"
SESSIONS_DIR = DATA_DIR / "sessions"

DEFAULT_CONFIG_TOML = """\
# tasuke configuration. Everything here has a default; delete a line to get it back.

[server]
# Any OpenAI-compatible chat endpoint works: llama-server, Ollama (http://127.0.0.1:11434),
# LM Studio (http://127.0.0.1:1234), vLLM... tasuke only needs /v1/chat/completions.
base_url = "http://127.0.0.1:8097"

# When nothing answers at base_url, start our own llama-server with the model below.
manage = true
port = 8097

# A .gguf path, or a substring of a filename found in model_dirs (e.g. "7b").
# Empty = pick the largest model that fits in 40% of system RAM (24 GB → 14B, 16 GB → 7B).
model = ""
model_dirs = ["~/models", "~/.cache/llama.cpp", "~/research/komori/experiments/models"]

# Optional chat template (.jinja). Not required: tasuke handles tool calls itself.
chat_template = ""
ctx_size = 16384
# Extra llama-server flags, e.g. ["--threads", "8"]
extra_args = []
# Name sent in the request. Only matters for Ollama / LM Studio, which route on it.
model_name = ""

[fast]
# A second, smaller model for the quick jobs: `tasuke suggest` (Ctrl-G), `tasuke explain`.
# Fixes and conversations stay on [server]. "auto" enables it only when both models fit in half of
# RAM (a 24 GB Mac running the 14B does not qualify; a 32 GB one does). true/false force it.
enabled = "auto"
model = "7b"
port = 8096
roles = ["suggest", "explain"]

[agent]
# "ask": run read-only commands automatically, confirm anything that changes state.
# "safe": like ask, but only destructive commands need confirmation.
# "yolo": never ask. You asked for it.
approval = "ask"
max_rounds = 20
command_timeout = 60
temperature = 0.2
# How much command output the model gets to see per call (characters). Head+tail kept.
max_tool_output_chars = 6000

[ui]
# Lines of command output shown on screen (the model may see more). -v shows everything.
output_lines = 25
code_theme = "ansi_dark"
"""


@dataclass
class ServerConfig:
    base_url: str = "http://127.0.0.1:8097"
    manage: bool = True
    port: int = 8097
    model: str = ""
    model_dirs: list[str] = field(default_factory=lambda: [
        "~/models", "~/.cache/llama.cpp", "~/research/komori/experiments/models",
    ])
    chat_template: str = ""
    ctx_size: int = 16384
    extra_args: list[str] = field(default_factory=list)
    model_name: str = ""


@dataclass
class FastConfig:
    enabled: object = "auto"   # "auto" | True | False
    model: str = "7b"
    port: int = 8096
    roles: list[str] = field(default_factory=lambda: ["suggest", "explain"])


@dataclass
class AgentConfig:
    approval: str = "ask"  # ask | safe | yolo
    max_rounds: int = 20
    command_timeout: int = 60
    temperature: float = 0.2
    max_tool_output_chars: int = 6000


@dataclass
class UIConfig:
    output_lines: int = 25
    code_theme: str = "ansi_dark"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    fast: FastConfig = field(default_factory=FastConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    path: Path = CONFIG_PATH

    def as_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def _apply(section_obj, data: dict, name: str):
    for k, v in data.items():
        if hasattr(section_obj, k):
            setattr(section_obj, k, v)
        else:
            raise ValueError(f"unknown key [{name}].{k}")


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    p = path or CONFIG_PATH
    cfg.path = p
    if p.exists():
        with open(p, "rb") as f:
            data = tomllib.load(f)
        if "server" in data:
            _apply(cfg.server, data["server"], "server")
        if "fast" in data:
            _apply(cfg.fast, data["fast"], "fast")
        if "agent" in data:
            _apply(cfg.agent, data["agent"], "agent")
        if "ui" in data:
            _apply(cfg.ui, data["ui"], "ui")
    if cfg.agent.approval not in ("ask", "safe", "yolo"):
        raise ValueError(f"[agent].approval must be ask|safe|yolo, got {cfg.agent.approval!r}")
    return cfg


def write_default_config(force: bool = False) -> bool:
    """Write the commented default config if none exists. Returns True if written."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists() and not force:
        return False
    CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return True


def ensure_dirs():
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, SESSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
