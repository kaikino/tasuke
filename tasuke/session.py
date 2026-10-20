"""Conversation persistence: one JSON per session under ~/.local/share/tasuke/sessions."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import SESSIONS_DIR


@dataclass
class Session:
    path: Path
    messages: list[dict] = field(default_factory=list)
    cwd: str = ""
    created: float = field(default_factory=time.time)
    title: str = ""
    changes: list[str] = field(default_factory=list)

    def outcome(self) -> str:
        """First line of the last real assistant answer (not a tool call)."""
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and "<tool_call>" not in m.get("content", ""):
                for line in m["content"].strip().splitlines():
                    line = line.strip().replace("**", "").lstrip("*-• #").strip()
                    if line:
                        return line
        return ""

    @classmethod
    def new(cls, cwd: str) -> "Session":
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + ".json"
        return cls(path=SESSIONS_DIR / name, cwd=cwd)

    @classmethod
    def load(cls, path: Path) -> "Session":
        d = json.loads(path.read_text())
        return cls(path=path, messages=d.get("messages", []), cwd=d.get("cwd", ""), created=d.get("created", 0),
                   title=d.get("title", ""), changes=d.get("changes", []))

    @classmethod
    def latest(cls) -> "Session | None":
        if not SESSIONS_DIR.is_dir():
            return None
        files = sorted(SESSIONS_DIR.glob("*.json"))
        return cls.load(files[-1]) if files else None

    @classmethod
    def list(cls, n: int = 20) -> list["Session"]:
        if not SESSIONS_DIR.is_dir():
            return []
        out = []
        for f in sorted(SESSIONS_DIR.glob("*.json"))[-n:]:
            try:
                out.append(cls.load(f))
            except (OSError, ValueError):
                continue
        return out

    def save(self):
        if not self.title:
            for m in self.messages:
                if m.get("role") == "user" and not m.get("content", "").startswith("<tool_response>"):
                    self.title = m["content"].strip().splitlines()[0][:80]
                    break
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"messages": self.messages, "cwd": self.cwd, "created": self.created,
                                   "title": self.title, "changes": self.changes}, ensure_ascii=False, indent=1))
        tmp.replace(self.path)
