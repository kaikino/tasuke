"""A terse snapshot of the machine for the system prompt. Must be fast (<50ms): no subprocesses
unless cached, since this runs on every invocation."""
from __future__ import annotations

import datetime as _dt
import getpass
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path

_PKG_MANAGERS = ["brew", "port", "apt", "apt-get", "dnf", "yum", "pacman", "apk", "nix", "snap", "flatpak"]
_LANG_TOOLS = ["python3", "uv", "pip", "pipx", "node", "npm", "pnpm", "bun", "go", "cargo", "rustc", "java", "ruby"]
_OPS_TOOLS = ["git", "docker", "podman", "kubectl", "helm", "terraform", "ansible", "ssh", "tmux", "nvim", "vim",
              "code", "jq", "rg", "fd", "fzf", "eza", "bat", "launchctl", "systemctl", "lsof", "dtrace"]


def _os_line() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        ver, _, _ = platform.mac_ver()
        return f"macOS {ver} (Darwin {platform.release()}), {platform.machine()}"
    if sysname == "Linux":
        pretty = ""
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
                    break
        except OSError:
            pass
        return f"{pretty or 'Linux'} (kernel {platform.release()}), {platform.machine()}"
    return f"{sysname} {platform.release()}, {platform.machine()}"


def _git_branch(cwd: Path) -> str | None:
    # Walk up looking for .git; read HEAD directly to avoid spawning git.
    d = cwd
    for _ in range(40):
        g = d / ".git"
        if g.exists():
            try:
                head_file = g / "HEAD" if g.is_dir() else None
                if head_file is None:  # worktree: .git is a file pointing elsewhere
                    gitdir = g.read_text().strip().split(" ", 1)[1]
                    head_file = Path(gitdir) / "HEAD"
                    if not head_file.is_absolute():
                        head_file = d / head_file
                head = head_file.read_text().strip()
                if head.startswith("ref: refs/heads/"):
                    return head[len("ref: refs/heads/"):]
                return head[:8]
            except (OSError, IndexError):
                return "?"
        if d.parent == d:
            break
        d = d.parent
    return None


def _present(names: list[str]) -> list[str]:
    return [n for n in names if shutil.which(n)]


def snapshot(cwd: Path | None = None, brief: bool = False) -> str:
    cwd = cwd or Path.cwd()
    shell = os.path.basename(os.environ.get("SHELL", "sh"))
    if brief:  # for suggest/explain: every token here is prompt-eval latency on a keystroke
        return f"OS: {_os_line()}; shell: {shell}; cwd: {cwd}"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    lines = [
        f"OS: {_os_line()}",
        f"Shell: {shell}; commands run via `{shell} -c`, non-interactive, no TTY, cwd does not persist",
        f"User: {getpass.getuser()}@{socket.gethostname().split('.')[0]}  (not root; sudo may prompt the user for a password)",
        f"cwd: {cwd}",
    ]
    br = _git_branch(cwd)
    if br:
        lines.append(f"Git: inside a repository, branch {br}")
    pm = _present(_PKG_MANAGERS)
    if pm:
        lines.append(f"Package managers: {', '.join(pm)}")
    tools = _present(_LANG_TOOLS) + _present(_OPS_TOOLS)
    if tools:
        lines.append(f"Available tools: {', '.join(tools)}")
    lines.append(f"Date: {now}")
    return "\n".join(lines)


def run_quiet(cmd: list[str], timeout: float = 2.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
