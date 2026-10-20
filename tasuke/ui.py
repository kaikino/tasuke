"""Terminal presentation: streaming markdown, command panels, approvals. All output goes through here."""
from __future__ import annotations

import sys
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from .parse import safe_prefix_len

THEME = Theme({
    "cmd": "bold cyan",
    "cmd.prompt": "bold green",
    "out": "dim",
    "err": "red",
    "warn": "yellow",
    "ok": "green",
    "meta": "dim",
    "risk.safe": "dim green",
    "risk.ask": "bold yellow",
    "risk.danger": "bold red",
    "risk.blocked": "bold magenta",
    "user": "bold",
})

RISK_LABEL = {"safe": "", "ask": "changes state", "danger": "DESTRUCTIVE", "blocked": "cannot run"}


def _tty_in():
    """A readable handle on the controlling terminal even when stdin is a pipe."""
    if sys.stdin.isatty():
        return sys.stdin
    try:
        return open("/dev/tty")
    except OSError:
        return None


class UI:
    def __init__(self, verbose: bool = False, output_lines: int = 25, code_theme: str = "ansi_dark", quiet: bool = False,
                 color: bool | None = None):
        force_terminal = None
        if color is True:
            force_terminal = True
        # Piped output gets a wide console: an 80-column wrap of paths and commands helps nobody downstream.
        width = None if sys.stdout.isatty() else 160
        self.console = Console(theme=THEME, stderr=False, force_terminal=force_terminal, no_color=(color is False),
                               highlight=False, width=width)
        self.err = Console(theme=THEME, stderr=True, no_color=(color is False), highlight=False)
        self.verbose = verbose
        self.output_lines = output_lines
        self.code_theme = code_theme
        self.quiet = quiet
        self.interactive = sys.stdout.isatty()
        # If neither stdin nor stdout is a terminal, nobody is there to answer prompts.
        self.can_prompt = sys.stdin.isatty() or sys.stdout.isatty()
        self._tty = None

    # ----------------------------------------------------------- basics
    def print(self, *a, **k):
        self.console.print(*a, **k)

    def info(self, msg: str):
        self.err.print(f"[meta]{msg}[/]")

    def warn(self, msg: str):
        self.err.print(f"[warn]{msg}[/]")

    def error(self, msg: str):
        self.err.print(f"[err]error:[/] {msg}")

    def rule(self, title: str = ""):
        self.console.print(Rule(title, style="meta"))

    @contextmanager
    def status(self, msg: str):
        if not self.interactive:
            yield lambda m: None
            return
        with self.console.status(f"[meta]{msg}[/]", spinner="dots", spinner_style="meta") as st:
            yield lambda m: st.update(f"[meta]{m}[/]")

    # ----------------------------------------------------------- streaming assistant text
    @contextmanager
    def stream(self):
        """Usage: with ui.stream() as s: s.feed(delta) ...; s.finish(final_prose)."""
        s = _Streamer(self)
        try:
            yield s
        finally:
            s.close()

    def markdown(self, text: str):
        if text.strip():
            self.console.print(Markdown(text, code_theme=self.code_theme))

    # ----------------------------------------------------------- tool calls
    def show_command(self, cmd: str, level: str, reason: str = "", edited: bool = False, dry: bool = False):
        t = Text()
        t.append("$ " if not dry else "$ (would run) ", style="cmd.prompt" if not dry else "warn")
        first = True
        for line in cmd.splitlines() or [""]:
            if not first:
                t.append("\n  ", style="cmd.prompt")
            t.append(line, style="cmd")
            first = False
        label = RISK_LABEL.get(level, "")
        if edited:
            t.append("  (edited)", style="meta")
        if label and level != "safe":
            t.append(f"  [{label}{': ' + reason if reason else ''}]", style=f"risk.{level}")
        self.console.print(t)

    def show_output(self, text: str, exit_code: int | None, elapsed: float, timed_out: bool = False):
        lines = text.splitlines()
        shown = lines if (self.verbose or len(lines) <= self.output_lines) else lines[: self.output_lines]
        if shown:
            self.console.print(Text("\n".join(shown), style="out"), soft_wrap=True)
        hidden = len(lines) - len(shown)
        bits = []
        if hidden > 0:
            bits.append(f"… {hidden} more lines (use -v to see all)")
        if exit_code not in (0, None):
            bits.append(f"exit {exit_code}")
        if timed_out:
            bits.append("timed out")
        if elapsed >= 2:
            bits.append(f"{elapsed:.1f}s")
        if bits:
            style = "err" if (exit_code not in (0, None) or timed_out) else "meta"
            self.console.print(Text("  ".join(bits), style=style))

    def show_file_op(self, label: str, path: str):
        t = Text()
        t.append("✎ ", style="cmd.prompt")
        t.append(f"{label} ", style="cmd")
        t.append(path, style="cmd")
        self.console.print(t)

    def show_diff(self, diff: str):
        if not diff.strip():
            return
        lines = diff.splitlines()
        if not self.verbose and len(lines) > 60:
            diff = "\n".join(lines[:60]) + f"\n… {len(lines) - 60} more diff lines (use -v to see all)"
        self.console.print(Syntax(diff, "diff", theme=self.code_theme, background_color="default"))

    def show_blocked(self, cmd: str, why: str):
        t = Text()
        t.append("$ ", style="cmd.prompt")
        t.append(cmd, style="cmd")
        t.append(f"  [not run: {why}]", style="risk.blocked")
        self.console.print(t)

    # ----------------------------------------------------------- approvals
    def _readline(self, prompt: str) -> str | None:
        """Read a line from the terminal, or None if there is no terminal."""
        if not self.can_prompt:
            return None
        if self._tty is None:
            self._tty = _tty_in()
        if self._tty is None:
            return None
        self.console.print(prompt, end="")
        try:
            line = self._tty.readline()
        except (KeyboardInterrupt, EOFError):
            self.console.print()
            return "n"
        if line == "":
            self.console.print()
            return "n"
        return line.rstrip("\n")

    def approve(self, level: str, what: str = "run this") -> tuple[str, str | None]:
        """Returns (decision, edited_text). decision ∈ yes | no | edit | always | quit."""
        if level == "danger":
            ans = self._readline("[risk.danger]destructive command[/] — type [bold]yes[/] to run, anything else to skip: ")
            if ans is None:
                return "no", None
            return ("yes" if ans.strip().lower() == "yes" else "no"), None
        while True:
            ans = self._readline(f"[warn]{what}?[/] [bold]y[/]es / [bold]n[/]o / [bold]e[/]dit / [bold]a[/]lways this session / [bold]q[/]uit  ")
            if ans is None:
                return "no", None
            a = ans.strip().lower()
            if a in ("", "y", "yes"):
                return "yes", None
            if a in ("n", "no"):
                return "no", None
            if a in ("a", "always"):
                return "always", None
            if a in ("q", "quit"):
                return "quit", None
            if a in ("e", "edit"):
                return "edit", None
            if a.startswith("e "):
                return "edit", a[2:]

    def edit_command(self, current: str) -> str | None:
        """Let the user rewrite the command. Uses prompt_toolkit if we have a real terminal."""
        if sys.stdin.isatty():
            try:
                from prompt_toolkit import prompt as pt_prompt
                new = pt_prompt("$ ", default=current)
                return new.strip() or None
            except (ImportError, EOFError, KeyboardInterrupt):
                return None
        ans = self._readline("[meta]new command (empty = cancel):[/]\n$ ")
        return (ans or "").strip() or None

    def ask_user(self, question: str) -> str | None:
        self.console.print()
        self.console.print(Text("? " + question, style="warn"))
        ans = self._readline("[bold]›[/] ")
        return ans

    def read_prompt(self, history_path=None) -> str | None:
        """Interactive REPL prompt. None on EOF/interrupt."""
        if sys.stdin.isatty():
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.history import FileHistory
                if not hasattr(self, "_session"):
                    hist = FileHistory(str(history_path)) if history_path else None
                    self._session = PromptSession(history=hist)
                return self._session.prompt([("bold", "› ")], multiline=False)
            except KeyboardInterrupt:
                return ""      # Ctrl-C clears the line; Ctrl-D (EOF) leaves
            except EOFError:
                return None
        return self._readline("[bold]›[/] ")

    def changes(self, items: list[str], dry: bool = False):
        if not items:
            return
        self.console.print(Text("plan (nothing was executed):" if dry else "changed:", style="warn" if dry else "ok"))
        for it in items:
            self.console.print(Text("  • " + it, style="out" if not dry else "warn"))

    def turn_footer(self, model: str, prompt_tokens: int, gen_tokens: int, tps: float, ncmd: int, elapsed: float):
        bits = [model]
        if prompt_tokens:
            bits.append(f"{prompt_tokens / 1000:.1f}k ctx")
        if gen_tokens:
            bits.append(f"{gen_tokens} tok")
        if tps:
            bits.append(f"{tps:.0f} tok/s")
        if ncmd:
            bits.append(f"{ncmd} cmd{'s' if ncmd != 1 else ''}")
        bits.append(f"{elapsed:.0f}s")
        self.console.print(Text(" · ".join(bits), style="meta"), justify="right")


class _Streamer:
    """Renders assistant prose as it streams, holding back anything that might be a tool call."""

    def __init__(self, ui: UI):
        self.ui = ui
        self.buf = ""
        self.shown = 0
        self.live: Live | None = None
        self.started = False
        self._rendered_text = ""

    def feed(self, delta: str):
        self.buf += delta
        n = safe_prefix_len(self.buf)
        if n <= self.shown and self.started:
            return
        text = self.buf[:n]
        if not text.strip():
            return
        if not self.started:
            self.started = True
            if self.ui.interactive:
                self.live = Live(Markdown(text, code_theme=self.ui.code_theme), console=self.ui.console,
                                 refresh_per_second=15, vertical_overflow="visible")
                self.live.start()
        self.shown = n
        self._rendered_text = text
        if self.live:
            self.live.update(Markdown(text, code_theme=self.ui.code_theme))

    def finish(self, prose: str):
        """Final render with the tool-call-free prose."""
        prose = prose.strip()
        if self.live:
            if prose:
                self.live.update(Markdown(prose, code_theme=self.ui.code_theme))
            self.live.stop()
            self.live = None
            if not prose:
                # Nothing to show after all; the Live already painted something, clear it visually.
                pass
        elif prose:
            self.ui.markdown(prose)
        self.started = False

    def close(self):
        if self.live:
            self.live.stop()
            self.live = None
