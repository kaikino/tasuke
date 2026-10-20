"""Pre-flight context: facts the model would otherwise spend its first calls discovering.

For `tasuke fix` we already know the failing command, so we run it (through the normal approval gate)
and hand the model the real error together with a directory listing. For a free-form request we add the
listing and, when the request names a script or command that exists here and complains it fails, we run
that too. Each avoided round-trip is 3-8 s on a local model.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

_FAIL_WORDS = re.compile(r"\b(fail\w*|error\w*|broken|crash\w*|doesn'?t work|does not work|can'?t|cannot|won'?t|"
                         r"not working|refus\w*|denied|exit(ed)? (code|status|with)|segfault|hang\w*|stuck)\b", re.I)
# `./script.sh args`, `make target`, `npm test`, `python x.py` ... inside backticks or at a word boundary
_CMD_RE = re.compile(r"`([^`\n]{2,120})`|((?:\./|~/|/)[\w./-]+)")  # backticked (with args) or a bare path


def mentioned_command(text: str, cwd: Path) -> str | None:
    """A command the request complains about, if it names a runnable file in cwd (or by absolute path)."""
    if not _FAIL_WORDS.search(text):
        return None
    for m in _CMD_RE.finditer(text):
        cand = (m.group(1) or m.group(2) or "").strip().rstrip(".,;:!?")
        if not cand:
            continue
        try:
            head = shlex.split(cand)[0]
        except (ValueError, IndexError):
            continue
        p = Path(head).expanduser()
        if not p.is_absolute():
            p = cwd / p
        if p.is_file() and (p.stat().st_mode & 0o111 or p.suffix in (".sh", ".py", ".js", ".ts", ".rb", ".pl")):
            if not (p.stat().st_mode & 0o111) and head.startswith(("./", "/", "~")):
                return cand  # still worth running: that is probably the "permission denied"
            return cand
    return None


def listing(cwd: Path, limit: int = 60) -> str:
    """`ls -la`-style view of cwd: names, sizes, modes, symlink targets. No subprocess."""
    try:
        entries = sorted(cwd.iterdir(), key=lambda p: p.name.lower())
    except OSError as e:
        return f"(cannot list {cwd}: {e})"
    rows = []
    for p in entries[:limit]:
        try:
            st = p.lstat()
        except OSError:
            continue
        kind = "l" if p.is_symlink() else "d" if p.is_dir() else "-"
        mode = kind + "".join(c if st.st_mode & b else "-" for c, b in
                              zip("rwxrwxrwx", (0o400, 0o200, 0o100, 0o40, 0o20, 0o10, 0o4, 0o2, 0o1)))
        name = p.name + ("/" if kind == "d" else "")
        if kind == "l":
            try:
                target = p.readlink()
                exists = p.exists()
                name += f" -> {target}" + ("" if exists else "  (BROKEN: target does not exist)")
            except OSError:
                pass
        rows.append(f"{mode} {st.st_size:>9} {name}")
    more = f"\n... {len(entries) - limit} more entries" if len(entries) > limit else ""
    return "\n".join(rows) + more if rows else "(empty directory)"


def compose(request: str, cwd: Path, listing_text: str, ran: str | None = None, output: str | None = None) -> str:
    parts = [request.rstrip()]
    if ran is not None and output is not None:
        parts.append(f"\nI already ran `{ran}` for you. Its output:\n{output}")
    parts.append(f"\nDirectory listing of {cwd}:\n{listing_text}")
    return "\n".join(parts)
