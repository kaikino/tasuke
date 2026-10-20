"""Pre-flight, fast tier gating, dry-run, change summary, session log, shell files."""
import os
import subprocess
from pathlib import Path

from tasuke import preflight as pf
from tasuke.agent import Agent, _diff_stat
from tasuke.backend import StreamResult, fast_tier_enabled
from tasuke.config import Config, load_config
from tasuke.session import Session
from tasuke.tools import Executor
from tasuke.ui import UI

SHELL_DIR = Path(__file__).resolve().parents[1] / "tasuke" / "shell"


# ------------------------------------------------------------------ preflight
def test_mentioned_command_only_for_runnable_files_that_fail(tmp_path):
    (tmp_path / "serve.sh").write_text("#!/bin/sh\n")
    os.chmod(tmp_path / "serve.sh", 0o755)
    (tmp_path / "run.sh").write_text("x")  # not executable: the "permission denied" case
    assert pf.mentioned_command("./serve.sh fails saying the address is in use", tmp_path) == "./serve.sh"
    assert pf.mentioned_command("`./serve.sh --port 9` keeps crashing", tmp_path) == "./serve.sh --port 9"
    assert pf.mentioned_command("./run.sh says permission denied", tmp_path) == "./run.sh"
    assert pf.mentioned_command("./serve.sh is great, explain it", tmp_path) is None   # no failure words
    assert pf.mentioned_command("./nothere.sh fails", tmp_path) is None               # not a file
    assert pf.mentioned_command("npm install fails here", tmp_path) is None            # not a path


def test_listing_marks_broken_symlinks(tmp_path):
    os.symlink("old/config.yml", tmp_path / "config.yml")
    (tmp_path / "shared").mkdir()
    text = pf.listing(tmp_path)
    assert "config.yml -> old/config.yml  (BROKEN" in text and "shared/" in text


def test_compose_includes_output_and_listing(tmp_path):
    msg = pf.compose("fix it", tmp_path, "-rw- 1 a", ran="./x", output="exit code: 1\nboom")
    assert "I already ran `./x`" in msg and "boom" in msg and "Directory listing" in msg


# ------------------------------------------------------------------ fast tier config
def test_fast_section_parses_and_auto_gates(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[fast]\nenabled = "auto"\nmodel = "small"\nport = 8096\n')
    cfg = load_config(cfg_path)
    assert cfg.fast.enabled == "auto" and cfg.fast.port == 8096
    # two fake models: 9 GB main, 5 GB fast
    mdir = tmp_path / "models"
    mdir.mkdir()
    for name, size in (("big-14b.gguf", 9_000_000_000), ("small-7b.gguf", 5_000_000_000)):
        with open(mdir / name, "wb") as f:
            f.truncate(size)
    cfg.server.model_dirs = [str(mdir)]
    cfg.server.model = "big"
    import tasuke.backend as be
    monkeypatch.setattr(be, "system_ram", lambda: 24 * 1024**3)
    on, why = fast_tier_enabled(cfg)
    assert not on and "half of RAM" in why
    monkeypatch.setattr(be, "system_ram", lambda: 64 * 1024**3)
    assert fast_tier_enabled(cfg)[0]
    cfg.fast.enabled = False
    assert fast_tier_enabled(cfg) == (False, "disabled")
    cfg.fast.enabled = True
    assert fast_tier_enabled(cfg)[0]


# ------------------------------------------------------------------ dry run + change summary
class _Scripted:
    """Replays canned assistant messages in order."""

    def __init__(self, replies):
        self.replies = list(replies)

    def stream_chat(self, messages, **kw):
        content = self.replies.pop(0) if self.replies else "Done."
        yield ("text", content)
        yield ("done", StreamResult(content=content, finish_reason="stop"))


def _agent(tmp_path, approval, replies):
    ui = UI(color=False)
    ui.can_prompt = False
    a = Agent(client=_Scripted(replies), ui=ui, executor=Executor(tmp_path), env_snapshot="OS: test", approval=approval)
    return a


def test_dry_run_executes_readonly_but_not_mutations(tmp_path):
    target = tmp_path / "f.txt"
    replies = [
        'Looking.\n<tool_call>\n{"name": "run_shell", "arguments": {"command": "ls"}}\n</tool_call>',
        'Fixing.\n<tool_call>\n{"name": "run_shell", "arguments": {"command": "touch f.txt"}}\n</tool_call>',
        'Writing.\n<tool_call>\n{"name": "write_file", "arguments": {"path": "g.txt", "content": "hi\\n"}}\n</tool_call>',
        "Done: I would create f.txt and g.txt.",
    ]
    a = _agent(tmp_path, "dry", replies)
    a.run_turn("make the files")
    assert not target.exists() and not (tmp_path / "g.txt").exists()
    assert a.changes == ["would run: touch f.txt", "would create: " + str(tmp_path / "g.txt")]
    # the model was told to assume success and continue
    assert any("DRY RUN" in m["content"] for m in a.messages if m["role"] == "user")


def test_change_summary_records_real_mutations(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    replies = [
        '<tool_call>\n{"name": "edit_file", "arguments": {"path": "a.txt", "old": "two", "new": "2"}}\n</tool_call>',
        '<tool_call>\n{"name": "run_shell", "arguments": {"command": "touch b.txt"}}\n</tool_call>',
        '<tool_call>\n{"name": "run_shell", "arguments": {"command": "cat a.txt"}}\n</tool_call>',
        "Done.",
    ]
    a = _agent(tmp_path, "yolo", replies)
    a.run_turn("fix a.txt")
    assert (tmp_path / "a.txt").read_text() == "one\n2\n"
    assert a.changes == [f"edited: {tmp_path / 'a.txt'}  (+1 -1)", "ran: touch b.txt"]  # cat is read-only: not a change


def test_diff_stat():
    assert _diff_stat("--- a\n+++ b\n@@\n-x\n+y\n+z\n") == "+2 -1"


# ------------------------------------------------------------------ session log
def test_session_outcome_and_changes_roundtrip(tmp_path):
    s = Session(path=tmp_path / "s.json", cwd="/x")
    s.messages = [
        {"role": "user", "content": "why is it broken"},
        {"role": "assistant", "content": '<tool_call>\n{"name":"run_shell","arguments":{"command":"ls"}}\n</tool_call>'},
        {"role": "user", "content": "<tool_response>\nok\n</tool_response>"},
        {"role": "assistant", "content": "**Fixed:** the config had a trailing comma.\nMore detail."},
    ]
    s.changes = ["edited: /x/config.json  (+1 -1)"]
    s.save()
    t = Session.load(s.path)
    assert t.title == "why is it broken" and t.outcome() == "Fixed: the config had a trailing comma." and t.changes == s.changes


# ------------------------------------------------------------------ shell files
def test_shell_files_parse():
    assert subprocess.run(["zsh", "-n", str(SHELL_DIR / "tasuke.zsh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(SHELL_DIR / "tasuke.bash")]).returncode == 0
    r = subprocess.run(["bash", "-c", f"source {SHELL_DIR / 'tasuke.bash'}; type ?? | head -1; echo $PROMPT_COMMAND"],
                       capture_output=True, text=True)
    assert "?? is a function" in r.stdout and "_tasuke_precmd" in r.stdout
