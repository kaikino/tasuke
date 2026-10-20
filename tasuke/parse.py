"""Tolerant extraction of tool calls from model output.

Small local models are inconsistent about the wrapper: we have seen `<tool_call>…</tool_call>`,
```json fences, bare JSON, and even `<tools>…</tools>`. We don't care about the wrapper — we scan
for JSON objects shaped like {"name": <known tool>, "arguments": {...}} and lift them out.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_NAME_KEY_RE = re.compile(r'\{\s*"(?:name|function|tool)"\s*:', re.S)
_OPEN_WRAP_RE = re.compile(r'(?:```(?:json|tool_call|tool|xml)?\s*|<tool_call>\s*|<tools?>\s*|<function_call>\s*)$', re.S)
_CLOSE_WRAP_RE = re.compile(r'^\s*(?:```|</tool_call>|</tools?>|</function_call>)', re.S)
_HOLD_MARKERS = ("```", "<tool", "<function", '{"', "{\n", "{ ")


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    raw: str = ""

    def canonical(self) -> str:
        return "<tool_call>\n" + json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False) + "\n</tool_call>"


def _coerce_args(obj: dict) -> dict | None:
    args = None
    for k in ("arguments", "parameters", "params", "args", "input"):
        if k in obj:
            args = obj[k]
            break
    if args is None:
        # Flat style: {"name": "run_shell", "command": "ls"}
        args = {k: v for k, v in obj.items() if k not in ("name", "function", "tool", "type", "id")}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args if isinstance(args, dict) else None


def extract_tool_calls(text: str, known: set[str]) -> tuple[str, list[ToolCall]]:
    """Return (prose_without_tool_calls, calls). Order preserved."""
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    dec = json.JSONDecoder()
    pos = 0
    while True:
        m = _NAME_KEY_RE.search(text, pos)
        if not m:
            break
        start = m.start()
        try:
            obj, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        end_abs = end
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("function") or obj.get("tool")
            if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
                inner = name
                name = inner.get("name")
                obj = inner
            if isinstance(name, str) and name in known:
                args = _coerce_args(obj)
                if args is not None:
                    # Widen span to swallow the wrapper.
                    s, e = start, end_abs
                    mo = _OPEN_WRAP_RE.search(text[:s])
                    if mo:
                        s = mo.start()
                    mc = _CLOSE_WRAP_RE.match(text[e:])
                    if mc:
                        e = e + mc.end()
                    calls.append(ToolCall(name=name, arguments=args, raw=text[start:end_abs]))
                    spans.append((s, e))
        pos = max(end_abs, start + 1)
    if not calls:
        return text, []
    out = []
    last = 0
    for s, e in spans:
        out.append(text[last:s])
        last = e
    out.append(text[last:])
    prose = "".join(out)
    # Clean leftover empty fences / stray tags
    prose = re.sub(r"```\s*```", "", prose)
    prose = re.sub(r"</?tool_call>|</?tools?>|</?function_call>", "", prose)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    return prose, calls


def safe_prefix_len(text: str) -> int:
    """How much of a streaming buffer can be shown as prose right now.

    Everything before the earliest possible tool-call wrapper is safe to render; the rest is held
    back until the stream ends and we know whether it was a tool call. Fenced blocks with a
    non-JSON language (```bash ...) are prose and stream through."""
    i, n = 0, len(text)
    jsonish = ("json", "tool_call", "tool", "xml", "javascript", "js")
    while i < n:
        cands = [(text.find(mk, i), mk) for mk in _HOLD_MARKERS]
        cands = [(p, mk) for p, mk in cands if p != -1]
        if not cands:
            return n
        p, mk = min(cands)
        if mk != "```":
            return p
        nl = text.find("\n", p)
        if nl == -1:
            return p  # fence header still streaming
        lang = text[p + 3:nl].strip().lower()
        if lang and lang not in jsonish:
            close = text.find("```", nl + 1)
            if close == -1:
                return n  # inside an unfinished code block: let it stream
            i = close + 3
            continue
        return p
    return n


SHELL_LANGS = {"", "sh", "bash", "zsh", "shell", "console", "shell-session", "fish", "terminal", "cmd"}
_FENCE_RE = re.compile(r"```([\w-]*)[^\n]*\n(.*?)```", re.S)


def shell_blocks(markdown: str) -> list[str]:
    """Shell commands the model suggested in fenced blocks (so the user can run them by number)."""
    out = []
    for lang, body in _FENCE_RE.findall(markdown):
        if lang.lower() not in SHELL_LANGS:
            continue
        lines = []
        for ln in body.strip().splitlines():
            ln = ln.rstrip()
            if not ln or ln.lstrip().startswith("#"):
                continue
            # strip "$ " prompts; skip lines that look like output under a console block
            if re.match(r"^\s*[\$%>]\s+", ln):
                ln = re.sub(r"^\s*[\$%>]\s+", "", ln)
            elif lang.lower() in ("console", "shell-session") and lines:
                continue
            lines.append(ln)
        if lines and (lang or all(_looks_like_command(ln) for ln in lines)):
            out.append(" && ".join(lines) if len(lines) <= 3 else "\n".join(lines))
    return out


_SHELL_WORDS = {"cd", "export", "source", "alias", "unalias", "set", "unset", "if", "for", "while", "then", "do", "done",
                "fi", "case", "esac", "echo", "printf", "eval", "exec", "sudo", "env", "time", "nohup", "test", "["}


def _looks_like_command(line: str) -> bool:
    """For an unlabeled fence: does this line start with something runnable? (Keeps `ls -l` output,
    stack traces and prose out of the "run by number" menu.)"""
    import shutil
    tok = line.strip().split(None, 1)[0] if line.strip() else ""
    if not tok or tok[0] in "-+|/<>#*{}[]\"" or tok.endswith((":", ",")):
        return False
    if "=" in tok and not tok.startswith("-"):  # VAR=value prefix
        return True
    return tok in _SHELL_WORDS or shutil.which(tok) is not None
