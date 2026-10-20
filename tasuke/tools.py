"""The tools the model can call, and how they run on the user's machine."""
from __future__ import annotations

import difflib
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import safety

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r")

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command on the user's machine and get back its exit code, stdout and stderr. "
                "Non-interactive: no TTY, stdin is closed, pagers are disabled. Each call starts in the same cwd; "
                "`cd` does not persist, so chain with && or use absolute paths. Long output is truncated (head and tail kept). "
                "Commands that change state are shown to the user for approval before running."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact shell command to run."},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed. Default 60; use more for installs/builds (max 900)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file (or part of it) with line numbers. Prefer this over `cat` for files you intend to edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-based first line (default 1)."},
                    "max_lines": {"type": "integer", "description": "Lines to return (default 200)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or replace a text file with the given content. The user sees a diff and must approve. "
                "For small edits to an existing file prefer `edit_file`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact text snippet in a file with another. `old` must appear exactly once in the file "
                "(copy it verbatim from read_file output, without the line numbers). The user sees a diff and must approve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a short question, only for things no command can reveal (a preference, a secret, a value "
                "that exists only in their head). Never ask for ports, paths, versions or error details: run the failing "
                "command, lsof, find, or read the file instead. Do not use it to ask permission; the tools already do that."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}


@dataclass
class ToolResult:
    text: str                       # what the model sees
    display: str = ""               # what the user sees (may be longer / formatted differently)
    exit_code: int | None = None
    elapsed: float = 0.0
    command: str = ""               # the command actually run (after edits/rewrites)
    skipped: bool = False
    meta: dict = field(default_factory=dict)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def truncate_middle(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = int(limit * 0.6)
    tail = limit - head
    omitted = s[head:-tail]
    return s[:head] + f"\n[... {omitted.count(chr(10))} lines / {len(omitted)} chars omitted ...]\n" + s[-tail:]


def _env_for_commands() -> dict:
    env = dict(os.environ)
    env.update({
        "PAGER": "cat", "GIT_PAGER": "cat", "MANPAGER": "cat", "SYSTEMD_PAGER": "cat", "LESS": "-FRX",
        "TERM": "dumb", "NO_COLOR": "1", "CLICOLOR": "0", "CLICOLOR_FORCE": "0", "FORCE_COLOR": "0",
        "HOMEBREW_NO_AUTO_UPDATE": "1", "HOMEBREW_NO_ENV_HINTS": "1", "HOMEBREW_COLOR": "0",
        "GIT_TERMINAL_PROMPT": "0", "DEBIAN_FRONTEND": "noninteractive", "PYTHONUNBUFFERED": "1",
        "TASUKE": "1",
    })
    for k in ("GIT_EDITOR", "EDITOR", "VISUAL"):
        env[k] = "true"  # anything that wants an editor just gets a no-op
    return env


class Executor:
    """Runs tools. Holds the agent's cwd (so a bare `cd` can be honoured between calls)."""

    def __init__(self, cwd: Path, shell: str | None = None, default_timeout: int = 60, max_output_chars: int = 6000):
        self.cwd = Path(cwd)
        self.shell = shell or os.environ.get("SHELL") or "/bin/sh"
        if not Path(self.shell).exists():
            self.shell = "/bin/sh"
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.env = _env_for_commands()
        self.last_interrupted = False

    # ------------------------------------------------------------ run_shell
    def run_shell(self, command: str, timeout: int | None = None, on_start=None) -> ToolResult:
        command = command.strip()
        timeout = max(1, min(int(timeout or self.default_timeout), 900))
        m = re.fullmatch(r"cd\s+(.+)", command)
        if m:  # bare cd: honour it in our own state instead of a throwaway subshell
            target = Path(os.path.expanduser(m.group(1).strip().strip("'\""))).expanduser()
            if not target.is_absolute():
                target = (self.cwd / target).resolve()
            if target.is_dir():
                self.cwd = target
                return ToolResult(text=f"exit code: 0\n(cwd is now {target}; subsequent commands run there)", exit_code=0, command=command)
            return ToolResult(text=f"exit code: 1\ncd: no such directory: {target}", exit_code=1, command=command)
        if command == "pwd":
            return ToolResult(text=f"exit code: 0\n--- stdout ---\n{self.cwd}", exit_code=0, command=command, display=str(self.cwd))
        command = safety.rewrite(command)
        t0 = time.time()
        try:
            proc = subprocess.Popen(
                [self.shell, "-c", command], cwd=str(self.cwd), env=self.env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as e:
            return ToolResult(text=f"exit code: 127\nfailed to start shell: {e}", exit_code=127, command=command)
        if on_start:
            on_start(proc)
        timed_out = False
        interrupted = False
        try:
            out_b, err_b = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(proc)
            out_b, err_b = proc.communicate()
        except KeyboardInterrupt:
            # The child runs in its own session so the terminal's SIGINT never reaches it: do it ourselves.
            interrupted = True
            self.last_interrupted = True
            self._kill(proc)
            try:
                out_b, err_b = proc.communicate(timeout=2)
            except (subprocess.TimeoutExpired, ValueError):
                out_b, err_b = b"", b""
        elapsed = time.time() - t0
        out = strip_ansi(out_b.decode("utf-8", "replace"))
        err = strip_ansi(err_b.decode("utf-8", "replace"))
        code = proc.returncode if proc.returncode is not None else -1
        parts = [f"exit code: {code}" + (f"  (killed: exceeded {timeout}s timeout)" if timed_out else "")
                 + ("  (interrupted by the user with Ctrl-C)" if interrupted else "")]
        if out.strip():
            parts.append("--- stdout ---\n" + out.rstrip())
        if err.strip():
            parts.append("--- stderr ---\n" + err.rstrip())
        if not out.strip() and not err.strip():
            parts.append("(no output)")
        model_text = truncate_middle("\n".join(parts), self.max_output_chars)
        display = (out.rstrip() + ("\n" if out.strip() and err.strip() else "") + err.rstrip()).strip()
        return ToolResult(text=model_text, display=display, exit_code=code, elapsed=elapsed, command=command,
                          meta={"timed_out": timed_out, "interrupted": interrupted})

    @staticmethod
    def _kill(proc: subprocess.Popen):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(0.3)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # ------------------------------------------------------------ files
    def _resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        return p if p.is_absolute() else (self.cwd / p)

    def read_file(self, path: str, start_line: int | None = None, max_lines: int | None = None) -> ToolResult:
        p = self._resolve(path)
        start = max(1, int(start_line or 1))
        count = max(1, min(int(max_lines or 200), 2000))
        try:
            if p.is_dir():
                entries = sorted(os.listdir(p))
                listing = "\n".join(entries[:300])
                return ToolResult(text=f"{p} is a directory with {len(entries)} entries:\n{listing}", display=f"{len(entries)} entries")
            data = p.read_bytes()
        except OSError as e:
            return ToolResult(text=f"error: {e}", exit_code=1)
        if b"\x00" in data[:4096]:
            return ToolResult(text=f"{p} looks binary ({len(data)} bytes); not showing it.", display="binary file")
        lines = data.decode("utf-8", "replace").splitlines()
        chunk = lines[start - 1:start - 1 + count]
        width = len(str(start + len(chunk)))
        body = "\n".join(f"{i:>{width}}| {ln}" for i, ln in enumerate(chunk, start))
        more = ""
        if start - 1 + count < len(lines):
            more = f"\n[... {len(lines) - (start - 1 + count)} more lines; file has {len(lines)} lines ...]"
        text = truncate_middle(f"{p} (lines {start}-{start + len(chunk) - 1} of {len(lines)}):\n{body}{more}", self.max_output_chars * 2)
        return ToolResult(text=text, display=f"{len(chunk)} line{'s' if len(chunk) != 1 else ''} of {len(lines)}", exit_code=0)

    def diff_for_write(self, path: str, content: str) -> tuple[Path, str, bool]:
        p = self._resolve(path)
        old = ""
        exists = p.exists()
        if exists:
            try:
                old = p.read_text(errors="replace")
            except OSError:
                old = ""
        diff = "".join(difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                                            fromfile=f"{p} (current)" if exists else "/dev/null",
                                            tofile=f"{p} (new)", n=3))
        if not diff and not exists:
            diff = f"(new empty file {p})"
        return p, diff, exists

    def write_file(self, path: str, content: str) -> ToolResult:
        p = self._resolve(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except OSError as e:
            return ToolResult(text=f"error: {e}", exit_code=1)
        return ToolResult(text=f"wrote {len(content.encode())} bytes to {p}", display=f"wrote {p}", exit_code=0)

    def prepare_edit(self, path: str, old: str, new: str) -> tuple[Path, str | None, str]:
        """Return (path, new_content_or_None, diff_or_error)."""
        p = self._resolve(path)
        try:
            cur = p.read_text(errors="replace")
        except OSError as e:
            return p, None, f"error: {e}"
        n = cur.count(old)
        if n == 0:
            # Models copy the "12| " prefixes from read_file output; forgive that, and stray edge whitespace.
            o2, n2 = _strip_line_numbers(old), _strip_line_numbers(new)
            for cand_old, cand_new in ((o2, n2), (o2.strip(), n2.strip()), (old.strip(), new.strip())):
                if cand_old and cur.count(cand_old) == 1:
                    old, new = cand_old, cand_new
                    n = 1
                    break
        if n == 0:
            return p, None, ("error: `old` text not found in file. It must match the file exactly; do not include the "
                             "`N| ` line-number prefixes from read_file. If the file is short, use write_file with the full corrected content instead.")
        if n > 1:
            return p, None, f"error: `old` text occurs {n} times; include more surrounding lines to make it unique."
        if old == new:
            return p, None, "error: `old` and `new` are identical, so nothing would change. Re-read the file and check what actually needs to differ."
        updated = cur.replace(old, new, 1)
        diff = "".join(difflib.unified_diff(cur.splitlines(True), updated.splitlines(True),
                                            fromfile=f"{p} (current)", tofile=f"{p} (edited)", n=3))
        return p, updated, diff

    def apply_edit(self, p: Path, updated: str) -> ToolResult:
        try:
            p.write_text(updated)
        except OSError as e:
            return ToolResult(text=f"error: {e}", exit_code=1)
        return ToolResult(text=f"edited {p}", display=f"edited {p}", exit_code=0)


_LINENO_RE = re.compile(r"^\s*\d+\|\s?", re.M)


def _strip_line_numbers(text: str) -> str:
    lines = text.splitlines(True)
    if lines and all(_LINENO_RE.match(ln) for ln in lines if ln.strip()):
        return "".join(_LINENO_RE.sub("", ln, count=1) for ln in lines)
    return text


def args_summary(name: str, args: dict) -> str:
    if name == "run_shell":
        return args.get("command", "")
    if name in ("read_file", "write_file", "edit_file"):
        return str(args.get("path", ""))
    if name == "ask_user":
        return str(args.get("question", ""))
    return json.dumps(args)[:120]
