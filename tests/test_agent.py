"""Agent-level behaviour that does not need a model: fallbacks, intent detection, inferred calls."""
from pathlib import Path

from tasuke.agent import Agent, looks_like_unfinished_intent
from tasuke.parse import ToolCall
from tasuke.tools import Executor
from tasuke.ui import UI


def _agent(tmp_path: Path) -> Agent:
    ui = UI(color=False)
    ui.can_prompt = False  # nobody at the keyboard
    return Agent(client=None, ui=ui, executor=Executor(tmp_path), env_snapshot="OS: test", approval="yolo")


def test_ask_user_without_a_human_does_not_invite_guessing(tmp_path):
    out = _agent(tmp_path)._run_tool(ToolCall("ask_user", {"question": "Which port?"}))
    assert "Do not guess" in out and "stop" in out


def test_intent_detection():
    assert looks_like_unfinished_intent("The brace is missing. Let's fix that and run it again.")
    assert looks_like_unfinished_intent("I will now check the logs:")
    assert not looks_like_unfinished_intent("Port 8081 is used by llama-server (pid 58377).")
    assert not looks_like_unfinished_intent("Done. The service is running again.")


def test_inferred_write_from_code_block(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"a": 1,}')
    a = _agent(tmp_path)
    prose = 'The config.json has a trailing comma. Here is the corrected file:\n```json\n{"a": 1}\n```\nLet\'s write it.'
    calls = a._infer_calls(prose)
    assert calls and calls[0].name == "write_file" and calls[0].arguments["path"] == str(cfg)
    assert calls[0].arguments["content"] == '{"a": 1}\n'


def test_inferred_command_from_shell_block(tmp_path):
    a = _agent(tmp_path)
    calls = a._infer_calls("Now let's run the tests again:\n```bash\npytest -q\n```")
    assert calls and calls[0].name == "run_shell" and calls[0].arguments["command"] == "pytest -q"


def test_declined_blocked_command_tells_model_why(tmp_path):
    out = _agent(tmp_path)._run_tool(ToolCall("run_shell", {"command": "vim foo.txt"}))
    assert out.startswith("not run:") and "interactive" in out


class _RepeatingClient:
    """Always proposes the same failing command; the agent must notice and stop instead of looping."""

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages, **kw):
        self.calls += 1
        from tasuke.backend import StreamResult
        content = 'Let me check.\n<tool_call>\n{"name": "run_shell", "arguments": {"command": "ls /definitely-missing"}}\n</tool_call>'
        yield ("text", content)
        yield ("done", StreamResult(content=content, finish_reason="stop"))


def test_turn_wide_loop_detection_stops_the_turn(tmp_path):
    a = _agent(tmp_path)
    a.client = _RepeatingClient()
    a.max_rounds = 20
    a.run_turn("why is my thing broken")
    assert a.client.calls == 3, "should stop on the third identical call, not run to max_rounds"
    assert "do not repeat it" in a.messages[-1]["content"].lower()


def test_permission_questions_are_answered_by_the_gate_not_the_user(tmp_path):
    from tasuke.agent import is_permission_question
    assert is_permission_question("Are you sure you want to kill process 50251?")
    assert is_permission_question("Do you want me to delete the old venv and recreate it?")
    assert not is_permission_question("What is your email address?")
    assert not is_permission_question("Which port should the server use?")
    out = _agent(tmp_path)._run_tool(ToolCall("ask_user", {"question": "Should I proceed with the kill?"}))
    assert out.startswith("Yes, go ahead")


def test_loop_signature_ignores_timeout_and_whitespace():
    from tasuke.agent import call_signature
    a = ToolCall("run_shell", {"command": "node --version  18.x"})
    b = ToolCall("run_shell", {"command": "node --version 18.x", "timeout": 60})
    assert call_signature(a) == call_signature(b)
    assert call_signature(ToolCall("read_file", {"path": "x"})) != call_signature(ToolCall("read_file", {"path": "y"}))
