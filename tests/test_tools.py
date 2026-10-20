from pathlib import Path

from tasuke.tools import Executor, _strip_line_numbers, truncate_middle


def test_strip_line_numbers():
    assert _strip_line_numbers('1| {"a": 1') == '{"a": 1'
    assert _strip_line_numbers("12| x\n13|  y\n") == "x\n y\n"
    assert _strip_line_numbers("plain") == "plain"


def test_edit_forgives_line_numbers(tmp_path: Path):
    f = tmp_path / "p.json"
    f.write_text('{"name": "x"')
    ex = Executor(tmp_path)
    p, updated, diff = ex.prepare_edit("p.json", '1| {"name": "x"', '1| {"name": "x"}')
    assert updated == '{"name": "x"}' and "+{" in diff


def test_edit_rejects_noop_and_missing(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    ex = Executor(tmp_path)
    assert ex.prepare_edit("a.txt", "hello", "hello")[1] is None
    assert ex.prepare_edit("a.txt", "nope", "x")[1] is None


def test_run_shell_and_cd(tmp_path: Path):
    ex = Executor(tmp_path)
    r = ex.run_shell("echo hi; echo err 1>&2; exit 3")
    assert r.exit_code == 3 and "hi" in r.text and "err" in r.text
    sub = tmp_path / "sub"
    sub.mkdir()
    assert ex.run_shell("cd sub").exit_code == 0 and ex.cwd == sub
    assert ex.run_shell("pwd").display == str(sub)


def test_timeout(tmp_path: Path):
    r = Executor(tmp_path).run_shell("sleep 5", timeout=1)
    assert r.meta["timed_out"] and "timeout" in r.text


def test_truncate_middle():
    s = "\n".join(f"line {i}" for i in range(1000))
    t = truncate_middle(s, 500)
    assert len(t) < 700 and t.startswith("line 0") and t.endswith("line 999") and "omitted" in t


def test_interrupt_kills_child(tmp_path: Path, monkeypatch):
    """Ctrl-C while a command runs must kill the detached child and report it, not hang or leak."""
    import subprocess

    ex = Executor(tmp_path)
    real = subprocess.Popen.communicate
    state = {"first": True}

    def fake_communicate(self, *a, **k):
        if state["first"]:
            state["first"] = False
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(subprocess.Popen, "communicate", fake_communicate)
    r = ex.run_shell("sleep 30")
    assert r.meta["interrupted"] and ex.last_interrupted and "Ctrl-C" in r.text
    assert not subprocess.run(["pgrep", "-f", "^sleep 30$"], capture_output=True).stdout
