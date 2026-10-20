from tasuke.parse import extract_tool_calls, safe_prefix_len, shell_blocks

K = {"run_shell", "read_file", "edit_file"}


def test_wrappers():
    for wrapped in [
        'Prose.\n\n```json\n{"name": "run_shell", "arguments": {"command": "ls"}}\n```',
        'Prose.\n\n<tools>\n{"name": "run_shell", "arguments": {"command": "ls"}}\n</tools>',
        'Prose.\n<tool_call>\n{"name": "run_shell", "arguments": {"command": "ls"}}\n</tool_call>',
        'Prose.\n{"name": "run_shell", "arguments": "{\\"command\\": \\"ls\\"}"}',
        'Prose.\n{"name": "run_shell", "command": "ls"}',
    ]:
        prose, calls = extract_tool_calls(wrapped, K)
        assert prose == "Prose." and [(c.name, c.arguments) for c in calls] == [("run_shell", {"command": "ls"})], wrapped


def test_unknown_tool_and_plain_code_left_alone():
    text = 'Use ```bash\nls -la\n``` and {"name": "nope", "arguments": {}}'
    prose, calls = extract_tool_calls(text, K)
    assert calls == [] and prose == text


def test_multiple_calls_in_order():
    t = '<tool_call>{"name":"run_shell","arguments":{"command":"a"}}</tool_call><tool_call>{"name":"read_file","arguments":{"path":"b"}}</tool_call>'
    _, calls = extract_tool_calls(t, K)
    assert [c.name for c in calls] == ["run_shell", "read_file"]


def test_streaming_holdback():
    assert safe_prefix_len("Hello there ```json\n{") == len("Hello there ")
    assert safe_prefix_len('x {"name"') == 2
    assert safe_prefix_len("plain text") == len("plain text")
    s = "Hello ```bash\nls\n``` ok ```json\n{\"name\""
    assert s[: safe_prefix_len(s)] == "Hello ```bash\nls\n``` ok "
    s = "see ```bash\nls -la\n"
    assert safe_prefix_len(s) == len(s)


def test_shell_blocks():
    md = "Run:\n```bash\n$ ls -la\n# comment\ncd /tmp\n```\nthen\n```python\nprint(1)\n```\n```\nbrew doctor\n```"
    assert shell_blocks(md) == ["ls -la && cd /tmp", "brew doctor"]


def test_unlabeled_blocks_need_runnable_lines():
    assert shell_blocks("```\n-rw-r--r--  1 u  w  0 Oct 20 hello.txt\n```") == []
    assert shell_blocks("```\nTraceback (most recent call last):\n  File x\n```") == []
    assert shell_blocks("```\nls -la\n```") == ["ls -la"]
    assert shell_blocks("```\nFOO=1 make test\n```") == ["FOO=1 make test"]
