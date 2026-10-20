"""Entry point. `tasuke <words>` asks; subcommands manage the server and shell integration."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import CONFIG_DIR, CONFIG_PATH, DATA_DIR, Config, ensure_dirs, load_config, write_default_config

SUBCOMMANDS = {"fix", "suggest", "explain", "serve", "stop", "status", "models", "config", "init", "sessions", "log", "help"}
HISTORY_PATH = DATA_DIR / "history"

USAGE = """\
tasuke — a local-LLM terminal assistant that fixes things by running commands.

  tasuke <what you want>          ask; it investigates and acts (asks before changing anything)
  tasuke                          interactive session
  cmd 2>&1 | tasuke [question]    diagnose piped output
  tasuke fix                      diagnose & fix the last failed command (needs shell integration)
  tasuke suggest <words>          print one shell command for the request (no tools, no chatter)
  tasuke explain <command>        explain a command
  tasuke -c ...                   continue the previous session

  tasuke init                     install zsh integration (?? alias, Ctrl-G, last-command tracking) + default config
  tasuke serve [--model X]        start the model server now (otherwise it starts on first use)
  tasuke stop | status | models   manage the server / list .gguf files found
  tasuke config                   show effective config and its path
  tasuke log [N]                  past sessions, one line each (request, outcome, what changed)

options: -y (only confirm destructive commands)  --yolo (never confirm)  -n/--dry-run (show, don't change)
         -m MODEL  --url URL  -v (full output)  -q (answer and exit)  --no-preflight  --no-color
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tasuke", add_help=False, usage=USAGE)
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("-V", "--version", action="store_true")
    p.add_argument("-c", "--continue", dest="cont", action="store_true", help="resume the last session")
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve everything except destructive commands")
    p.add_argument("--yolo", action="store_true", help="never ask for approval")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would be run or written; execute only read-only commands")
    p.add_argument("-m", "--model", default="", help=".gguf path or substring, e.g. 7b")
    p.add_argument("--url", default="", help="OpenAI-compatible server URL")
    p.add_argument("-v", "--verbose", action="store_true", help="show full command output")
    p.add_argument("-q", "--quiet", action="store_true", help="answer and exit; no interactive follow-up")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--cwd", default="", help="run as if from this directory")
    p.add_argument("--timeout", type=int, default=0, help="default per-command timeout in seconds")
    p.add_argument("--no-preflight", action="store_true", help="don't pre-run the failing command / attach a directory listing")
    p.add_argument("words", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"tasuke {__version__}")
        return 0
    if args.help:
        print(USAGE)
        return 0
    words = args.words
    if words and words[0] == "--":
        words = words[1:]
    sub = words[0] if words and words[0] in SUBCOMMANDS else None
    rest = words[1:] if sub else words
    if rest and rest[0] == "--":
        rest = rest[1:]

    try:
        cfg = load_config()
    except (ValueError, OSError) as e:
        print(f"tasuke: bad config {CONFIG_PATH}: {e}", file=sys.stderr)
        return 2
    if args.timeout:
        cfg.agent.command_timeout = args.timeout
    ensure_dirs()

    from .ui import UI
    ui = UI(verbose=args.verbose, output_lines=cfg.ui.output_lines, code_theme=cfg.ui.code_theme,
            quiet=args.quiet, color=False if args.no_color else None)

    try:
        if sub == "help":
            print(USAGE)
            return 0
        if sub == "init":
            return cmd_init(ui)
        if sub == "config":
            return cmd_config(cfg, ui)
        if sub == "models":
            return cmd_models(cfg, ui)
        if sub == "status":
            return cmd_status(cfg, ui, args.url)
        if sub == "stop":
            return cmd_stop(ui)
        if sub in ("sessions", "log"):
            return cmd_log(ui, rest)
        if sub == "serve":
            return cmd_serve(cfg, ui, args.model)
        if sub == "suggest":
            return cmd_suggest(cfg, ui, args, " ".join(rest))
        if sub == "explain":
            return cmd_explain(cfg, ui, args, " ".join(rest))
        if sub == "fix":
            return cmd_chat(cfg, ui, args, " ".join(rest), fix=True)
        return cmd_chat(cfg, ui, args, " ".join(rest))
    except KeyboardInterrupt:
        ui.console.print()
        return 130


# ---------------------------------------------------------------------------- helpers
def _read_piped_stdin() -> str:
    """Read stdin only when it is a pipe or a file. A tty is the user; a socket or anything else
    (daemons, odd harnesses) could block forever waiting for an EOF that never comes."""
    import stat
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError):
        return ""
    if not (stat.S_ISFIFO(mode) or stat.S_ISREG(mode)):
        return ""
    try:
        data = sys.stdin.read()
    except OSError:
        return ""
    return data.strip()


def _connect_role(cfg: Config, ui, args, role: str):
    """suggest/explain go to the fast tier when configured; anything else to the main server."""
    if role in cfg.fast.roles and not args.model and not args.url:
        from .backend import connect_fast
        with ui.status("connecting…") as upd:
            c = connect_fast(cfg, progress=upd)
        if c is not None:
            return c
    return _connect(cfg, ui, args)


def _connect(cfg: Config, ui, args, quiet_errors: bool = False):
    from .backend import BackendError, connect

    said = {"start": False}

    def progress_factory(upd):
        def progress(msg: str):
            upd(msg)
            if not ui.interactive and not said["start"] and ("starting" in msg or "loading" in msg):
                said["start"] = True
                ui.info(msg)  # a non-tty caller gets one line instead of a spinner
        return progress

    try:
        with ui.status("connecting…") as upd:
            return connect(cfg, model_override=args.model, url_override=args.url, progress=progress_factory(upd))
    except BackendError as e:
        if quiet_errors:
            raise
        ui.error(str(e))
        return None


def _approval_mode(cfg: Config, args) -> str:
    if args.dry_run:
        return "dry"
    if args.yolo:
        return "yolo"
    if args.yes:
        return "safe"
    return cfg.agent.approval


# ---------------------------------------------------------------------------- chat / fix
def cmd_chat(cfg: Config, ui, args, prompt: str, fix: bool = False) -> int:
    from .agent import Agent, UserQuit
    from .backend import BackendError
    from .session import Session
    from .sysinfo import snapshot
    from .tools import Executor

    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    piped = _read_piped_stdin()
    has_tty = sys.stdin.isatty() or os.path.exists("/dev/tty")
    interactive = has_tty and sys.stdout.isatty() and not args.quiet

    preflight_cmd = None
    if fix:
        last_cmd = (prompt or os.environ.get("TASUKE_LAST_CMD", "")).strip()
        status = os.environ.get("TASUKE_LAST_STATUS", "?") if not prompt else "?"
        if not last_cmd:
            ui.error("no previous command recorded. Install the shell integration with `tasuke init` "
                     "(or pass the command: tasuke fix -- <command>).")
            return 2
        if last_cmd.split()[0] in ("tasuke", "??", "fix"):
            ui.error("the last command was tasuke itself; nothing to fix.")
            return 2
        from .prompts import FIX_PROMPT
        extra = f"\nOutput the user pasted:\n```\n{piped}\n```" if piped else ""
        ui.print(f"[meta]last command:[/] [cmd]{last_cmd}[/] [meta](exit {status})[/]")
        prompt = FIX_PROMPT.format(cmd=last_cmd, status=status, extra=extra)
        preflight_cmd = None if piped else last_cmd
    elif piped:
        q = prompt or "Diagnose this output. If something is wrong, find the cause on this machine and fix it."
        from .tools import truncate_middle
        prompt = f"{q}\n\nOutput I'm looking at:\n```\n{truncate_middle(piped, 8000)}\n```"

    if not prompt and not interactive:
        ui.error("nothing to do: give me a question, pipe some output, or run in a terminal.")
        return 2

    client = _connect(cfg, ui, args)
    if client is None:
        return 1

    session = None
    messages = None
    if args.cont:
        session = Session.latest()
        if session is None:
            ui.warn("no previous session to continue; starting fresh")
        else:
            messages = session.messages
            ui.info(f"continuing: {session.title or session.path.name} ({len(messages)} messages)")
    if session is None:
        session = Session.new(str(cwd))

    ex = Executor(cwd=cwd, default_timeout=cfg.agent.command_timeout, max_output_chars=cfg.agent.max_tool_output_chars)
    def reconnect():
        c = _connect(cfg, ui, args, quiet_errors=True)
        return c

    agent = Agent(client, ui, ex, snapshot(cwd), approval=_approval_mode(cfg, args), max_rounds=cfg.agent.max_rounds,
                  n_ctx=client.n_ctx(cfg.server.ctx_size), messages=messages, reconnect=reconnect)
    model_name = client.describe()

    first_turn = {"pending": True}

    def do_turn(text: str):
        import time
        from . import preflight as pf
        t0 = time.time()
        if first_turn["pending"] and not args.no_preflight:
            first_turn["pending"] = False
            cmd = preflight_cmd or (pf.mentioned_command(text, ex.cwd) if not piped else None)
            out = agent.preflight_run(cmd) if cmd else None
            text = pf.compose(text, ex.cwd, pf.listing(ex.cwd), ran=cmd if out else None, output=out)
        try:
            final = agent.run_turn(text)
        except UserQuit:
            ui.info("stopped.")
            session.messages = agent.messages
            session.save()
            raise
        session.messages = agent.messages
        session.cwd = str(ex.cwd)
        session.save()
        if agent.changes:
            ui.changes(agent.changes, dry=(agent.approval == "dry"))
            session.changes.extend(agent.changes)
            agent.changes = []
            session.save()
        if interactive:
            ui.turn_footer(model_name, agent.stats.prompt_tokens, agent.stats.gen_tokens, agent.stats.tps,
                           agent.stats.commands, time.time() - t0)
            if agent.last_blocks:
                hints = "  ".join(f"[bold]{i + 1}[/]: [cmd]{b.splitlines()[0][:50]}{'…' if len(b) > 50 or chr(10) in b else ''}[/]"
                                  for i, b in enumerate(agent.last_blocks[:9]))
                ui.print(f"[meta]run a suggested command by number →[/] {hints}")
        return final

    try:
        if prompt:
            do_turn(prompt)
        if not interactive:
            return 0
        if not prompt:
            ui.print(f"[meta]tasuke · {model_name} · cwd {ex.cwd}[/]  [meta]type a request, !cmd to run a command, exit to quit[/]")
        while True:
            line = ui.read_prompt(HISTORY_PATH)
            if line is None:
                ui.console.print()
                break
            line = line.strip()
            if not line:
                continue
            if line in ("exit", "quit", "/q", ":q"):
                break
            if line in ("/v", "/verbose"):
                ui.verbose = not ui.verbose
                ui.info(f"verbose {'on' if ui.verbose else 'off'}")
                continue
            if line in ("/yolo",):
                agent.approval = "yolo" if agent.approval != "yolo" else cfg.agent.approval
                ui.info(f"approval mode: {agent.approval}")
                continue
            if line.isdigit() and agent.last_blocks and 1 <= int(line) <= len(agent.last_blocks):
                import time
                t0 = time.time()
                try:
                    agent.run_block(int(line) - 1)
                except UserQuit:
                    break
                session.messages = agent.messages
                session.save()
                ui.turn_footer(model_name, agent.stats.prompt_tokens, agent.stats.gen_tokens, agent.stats.tps,
                               agent.stats.commands, time.time() - t0)
                continue
            if line.startswith("!") and len(line) > 1:
                # User runs a command themselves; the model sees it too.
                cmd = line[1:].strip()
                from .safety import classify
                v = classify(cmd)
                ui.show_command(cmd, v.level, v.reason)
                with ui.status("running…"):
                    res = ex.run_shell(cmd)
                ui.show_output(res.display, res.exit_code, res.elapsed, res.meta.get("timed_out", False))
                agent.messages.append({"role": "user", "content": f"I ran this myself:\n$ {cmd}\n{res.text}"})
                agent.messages.append({"role": "assistant", "content": "Noted."})
                continue
            do_turn(line)
    except UserQuit:
        pass
    except BackendError as e:
        ui.error(str(e))
        return 1
    return 0


# ---------------------------------------------------------------------------- suggest / explain
def _clean_command(text: str) -> str:
    import re
    t = text.strip()
    m = re.search(r"```[\w-]*\n(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    lines = [ln for ln in t.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return ""
    cmd = lines[0].strip()
    cmd = re.sub(r"^[\$%>]\s+", "", cmd)
    cmd = cmd.strip("`")
    return cmd


def cmd_suggest(cfg: Config, ui, args, text: str) -> int:
    from .backend import BackendError
    from .prompts import SUGGEST_PROMPT
    from .sysinfo import snapshot

    piped = _read_piped_stdin()
    text = text or piped
    if not text:
        ui.error("usage: tasuke suggest <what you want to do>")
        return 2
    client = _connect_role(cfg, ui, args, "suggest")
    if client is None:
        return 1
    msgs = [{"role": "system", "content": SUGGEST_PROMPT + snapshot(brief=True)},
            {"role": "user", "content": text}]
    try:
        out = client.complete(msgs, max_tokens=200, temperature=0.0, stop=["\n\n"])
    except BackendError as e:
        ui.error(str(e))
        return 1
    cmd = _clean_command(out)
    if not cmd:
        return 1
    print(cmd)
    return 0


def cmd_explain(cfg: Config, ui, args, text: str) -> int:
    from .backend import BackendError
    from .prompts import EXPLAIN_PROMPT
    from .sysinfo import snapshot

    piped = _read_piped_stdin()
    text = text or piped
    if not text:
        ui.error("usage: tasuke explain <command>")
        return 2
    client = _connect_role(cfg, ui, args, "explain")
    if client is None:
        return 1
    msgs = [{"role": "system", "content": EXPLAIN_PROMPT + snapshot(brief=True)},
            {"role": "user", "content": f"`{text}`"}]
    try:
        with ui.status("thinking…"):
            gen = client.stream_chat(msgs, max_tokens=700)
            first = next(gen)
        with ui.stream() as s:
            if first[0] == "text":
                s.feed(first[1])
            for kind, payload in gen:
                if kind == "text":
                    s.feed(payload)
            s.finish(s.buf)
    except BackendError as e:
        ui.error(str(e))
        return 1
    return 0


# ---------------------------------------------------------------------------- server management
def cmd_serve(cfg: Config, ui, model: str) -> int:
    from .backend import BackendError, LLMClient, pick_model, start_server, server_info, _read_pid

    client = LLMClient(cfg.server.base_url)
    if client.is_up():
        ui.info(f"already serving {client.describe()} at {cfg.server.base_url}")
        if not model:
            return 0
        if _read_pid() and server_info().get("base_url") == cfg.server.base_url:
            from .backend import stop_server
            stop_server()
        else:
            ui.error("that server isn't managed by tasuke; stop it first")
            return 1
    try:
        m = pick_model(cfg, model)
        with ui.status("starting…") as upd:
            c = start_server(cfg, m, progress=upd)
        ui.print(f"[ok]serving[/] {m.name} ({m.size_gb:.1f} GB) at {c.base_url}")
    except BackendError as e:
        ui.error(str(e))
        return 1
    return 0


def cmd_stop(ui) -> int:
    from .backend import stop_server
    stopped = [t for t in ("main", "fast") if stop_server(t)]
    if stopped:
        ui.print(f"[ok]stopped[/] {', '.join(stopped)} llama-server")
    else:
        ui.info("no tasuke-managed server running")
    return 0


def cmd_status(cfg: Config, ui, url: str) -> int:
    from .backend import LLMClient, server_info, _read_pid, LOG_FILE
    base = url or cfg.server.base_url
    client = LLMClient(base)
    pid = _read_pid()
    info = server_info()
    if client.is_up():
        props = client.server_props()
        ctx = props.get("default_generation_settings", {}).get("n_ctx", "?")
        who = "managed by tasuke" if (pid and info.get("base_url") == base) else "external"
        ui.print(f"[ok]up[/]    {base}  model [cmd]{client.describe()}[/]  ctx {ctx}  ({who}{', pid ' + str(pid) if pid else ''})")
    else:
        ui.print(f"[warn]down[/]  nothing answering at {base}" + (f" (stale pid file: {pid})" if pid else ""))
        ui.info(f"log: {LOG_FILE}")
    if not url:
        from .backend import fast_tier_enabled
        on, why = fast_tier_enabled(cfg)
        fb = f"http://127.0.0.1:{cfg.fast.port}"
        fc = LLMClient(fb)
        fpid = _read_pid("fast")
        if fc.is_up():
            ui.print(f"[ok]fast[/]  {fb}  model [cmd]{fc.describe()}[/]  roles {', '.join(cfg.fast.roles)}" + (f"  (pid {fpid})" if fpid else "")
                     + ("" if on else "  [warn](running but not routed to: " + why + ")[/]"))
        elif on:
            ui.print(f"[meta]fast[/]  {fb}  not running; starts on first `suggest`/`explain` ({why})")
        else:
            ui.print(f"[meta]fast[/]  off ({why})")
    return 0


def cmd_models(cfg: Config, ui) -> int:
    from .backend import find_models, pick_model, system_ram, BackendError
    models = find_models(cfg.server.model_dirs)
    if not models:
        ui.warn("no .gguf files found in: " + ", ".join(cfg.server.model_dirs))
        ui.info("drop a model in ~/models or set [server].model_dirs in " + str(cfg.path))
        return 1
    try:
        default = pick_model(cfg).path.resolve()
    except BackendError:
        default = None
    ui.print(f"[meta]RAM {system_ram() / 1e9:.0f} GB → default picks the largest model under {system_ram() / 2.5e9:.0f} GB "
             f"(override with \\[server].model or -m)[/]")
    for m in models:
        mark = "[ok]*[/]" if m.path.resolve() == default else " "
        ui.print(f" {mark} {m.name:<48} {m.size_gb:5.1f} GB  [meta]{m.path.parent}[/]")
    return 0


def cmd_config(cfg: Config, ui) -> int:
    import json
    exists = cfg.path.exists()
    ui.print(f"[meta]config:[/] {cfg.path} {'' if exists else '[warn](not written yet; run tasuke init)[/]'}")
    ui.print(json.dumps({k: v for k, v in cfg.as_dict().items() if k != 'path'}, indent=2))
    return 0


def cmd_log(ui, rest: list[str]) -> int:
    """`tasuke log [N]`: past sessions, newest last, one line each: when · request · outcome · changes."""
    import time
    from .session import Session
    n = int(rest[0]) if rest and rest[0].isdigit() else 20
    ss = Session.list(n)
    if not ss:
        ui.info("no sessions yet")
        return 0
    for s in ss:
        when = time.strftime("%m-%d %H:%M", time.localtime(s.created)) if s.created else "?"
        ncmd = sum(m["content"].count('"run_shell"') for m in s.messages if m["role"] == "assistant")
        outcome = s.outcome()
        chg = f"  [ok]{len(s.changes)} change{'s' if len(s.changes) != 1 else ''}[/]" if s.changes else ""
        ui.print(f"[meta]{when}[/]  [bold]{(s.title or '(untitled)')[:60]}[/]  [meta]· {ncmd} cmd{'s' if ncmd != 1 else ''}[/]{chg}")
        if outcome:
            ui.print(f"           [out]{outcome[:110]}[/]")
    ui.print(f"[meta]resume the last one with  tasuke -c   · files in {ss[-1].path.parent}[/]")
    return 0


# ---------------------------------------------------------------------------- shell integration
def cmd_init(ui) -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    wrote_cfg = write_default_config()
    ui.print(f"[ok]config[/]  {CONFIG_PATH} {'written' if wrote_cfg else 'already exists, left alone'}")
    shell = os.path.basename(os.environ.get("SHELL", ""))
    if shell not in ("zsh", "bash"):
        ui.warn(f"your shell is {shell or 'unknown'}; integration exists for zsh and bash. "
                f"Source {CONFIG_DIR}/tasuke.zsh or tasuke.bash yourself if one applies.")
        shell = "zsh"
    for name in ("tasuke.zsh", "tasuke.bash"):
        shutil.copyfile(Path(__file__).parent / "shell" / name, CONFIG_DIR / name)
    dst = CONFIG_DIR / f"tasuke.{shell}"
    ui.print(f"[ok]shell[/]   {dst} installed")
    rc = Path.home() / (".zshrc" if shell == "zsh" else ".bashrc")
    line = f'[[ -f ~/.config/tasuke/tasuke.{shell} ]] && source ~/.config/tasuke/tasuke.{shell}  # tasuke'
    try:
        content = rc.read_text() if rc.exists() else ""
    except OSError:
        content = ""
    if f"tasuke.{shell}" in content:
        ui.print(f"[ok]rc[/]      {rc} already sources it")
    else:
        with open(rc, "a") as f:
            f.write("\n# tasuke: ?? <question>, Ctrl-G to turn text into a command, `tasuke fix` for the last failure\n")
            f.write(line + "\n")
        ui.print(f"[ok]rc[/]      added one line to {rc}")
    ui.print(f"\nOpen a new shell (or `source {rc}`), then try:  [cmd]?? what is listening on port 8080[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
