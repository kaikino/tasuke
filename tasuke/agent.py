"""The agent loop: model proposes a tool call, we gate it, run it, feed back the result, repeat."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import safety
from .backend import BackendError, LLMClient
from .config import STATE_DIR
from .parse import ToolCall, extract_tool_calls, shell_blocks
from .prompts import agent_system_prompt
from .tools import TOOL_NAMES, Executor
from .ui import UI

CHARS_PER_TOKEN = 3.6  # conservative for English + code on Qwen tokenizers

# Small models love to narrate an action instead of taking it ("Let's fix that and run it again.").
# If a reply has no tool call but reads like an intent to act, we nudge once or twice.
_INTENT_RE = re.compile(
    r"\b(let'?s|let me|i'?ll|i will|i am going to|i'm going to|we (?:can|need to|should|will)|now (?:i|we)|next,? (?:i|we)|"
    r"going to|first,? (?:i|we)|the next step is)\b[^.\n]{0,80}\b"
    r"(fix|run|re-?run|check|try|update|edit|change|add|remove|delete|install|look|see|verify|modify|create|write|"
    r"inspect|examine|investigate|restart|start|stop|kill|open|read|correct|apply|set|replace|append|test|confirm|find)\w*\b",
    re.IGNORECASE,
)
NUDGE = ("You described an action but did not make a tool call, so nothing happened. Do it now as a tool call, e.g.\n"
         "<tool_call>\n{\"name\": \"edit_file\", \"arguments\": {\"path\": \"<file>\", \"old\": \"<exact text>\", \"new\": \"<replacement>\"}}\n</tool_call>\n"
         "or run_shell / write_file. Do not show file contents or commands in code blocks; call the tool. "
         "If you are actually finished, give the final answer without announcing a next step.")

_FENCE_RE = re.compile(r"```([\w.-]*)[^\n]*\n(.*?)```", re.S)
_PATH_RE = re.compile(r"`?((?:~|\.{1,2})?/(?:[\w.-]+/)*[\w.-]+|(?:[\w.-]+/)*[\w-]*\.[\w-]+)`?")
_FILE_LANGS = {"json", "yaml", "yml", "toml", "ini", "xml", "html", "css", "js", "javascript", "ts", "typescript",
               "py", "python", "rb", "ruby", "go", "rust", "rs", "c", "cpp", "h", "java", "kt", "swift", "conf", "cfg",
               "env", "dockerfile", "makefile", "text", "txt", "plist", "nginx", "sql", "md", "markdown", "properties",
               "jsonc", "json5", "hcl", "tf", "lua", "php", "diff"}


def _diff_stat(diff: str) -> str:
    plus = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    minus = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return f"+{plus} -{minus}"


_PERMISSION_Q = re.compile(r"\b(are you sure|do you want( me)? to|would you like( me)? to|should i|shall i|may i|can i (go ahead|proceed)|"
                           r"is it (ok|okay|fine) (to|if)|proceed\?|confirm|go ahead)\b", re.I)


def is_permission_question(q: str) -> bool:
    """`ask_user` used as a second approval prompt. The real gate already asks before anything runs, so
    this is answered automatically instead of bothering the user twice."""
    return bool(_PERMISSION_Q.search(q))


def call_signature(call: ToolCall) -> str:
    """Identity of a tool call for loop detection: for run_shell only the command text matters."""
    if call.name == "run_shell":
        return "run_shell:" + " ".join(str(call.arguments.get("command", "")).split())
    return call.name + ":" + json.dumps(call.arguments, sort_keys=True)


def looks_like_unfinished_intent(prose: str) -> bool:
    tail = prose.strip()[-400:]
    if not tail:
        return False
    if tail.endswith(("...", "…", ":")):
        return True
    # Look at the last two sentences only; an intro sentence followed by a conclusion is fine.
    sentences = re.split(r"(?<=[.!?])\s+", tail)
    return bool(_INTENT_RE.search(" ".join(sentences[-2:])))


class UserQuit(Exception):
    pass


@dataclass
class TurnStats:
    rounds: int = 0
    commands: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    tps: float = 0.0
    started: float = field(default_factory=time.time)


class Agent:
    def __init__(self, client: LLMClient, ui: UI, executor: Executor, env_snapshot: str, approval: str = "ask",
                 max_rounds: int = 20, n_ctx: int = 16384, messages: list[dict] | None = None, reconnect=None):
        self.client = client
        self.reconnect = reconnect  # callable returning a fresh LLMClient, or None
        self.ui = ui
        self.ex = executor
        self.approval = approval  # ask | safe | yolo | dry
        self.changes: list[str] = []   # what this session changed, for the end-of-turn summary
        self.max_rounds = max_rounds
        self.n_ctx = n_ctx
        self.system = agent_system_prompt(env_snapshot)
        self.messages: list[dict] = messages if messages is not None else []
        self.always_allow: set[str] = set()   # command heads the user said "always" to
        self.last_blocks: list[str] = []      # shell blocks from the last answer, runnable by number
        self.last_file_path: str | None = None  # most recent file the model read/wrote (for inferred writes)
        self.stats = TurnStats()

    # ------------------------------------------------------------------ context budget
    def _estimate_tokens(self, msgs: list[dict]) -> int:
        return int(sum(len(m.get("content", "")) for m in msgs) / CHARS_PER_TOKEN) + 8 * len(msgs)

    def _fit_context(self, reserve: int = 1800) -> list[dict]:
        """Return system + history trimmed to fit n_ctx. Old tool outputs shrink first, then old turns drop."""
        sysmsg = {"role": "system", "content": self.system}
        budget = self.n_ctx - reserve
        msgs = [dict(m) for m in self.messages]
        if self._estimate_tokens([sysmsg] + msgs) <= budget:
            return [sysmsg] + msgs
        # Pass 1: compact tool responses except the last two.
        tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"].startswith("<tool_response>")]
        for i in tool_idx[:-2]:
            c = msgs[i]["content"]
            if len(c) > 400:
                msgs[i]["content"] = c[:300].rstrip() + "\n[... earlier output trimmed to save context ...]\n</tool_response>"
            if self._estimate_tokens([sysmsg] + msgs) <= budget:
                return [sysmsg] + msgs
        # Pass 2: drop oldest exchanges, but always keep the first user request and the last 6 messages.
        while len(msgs) > 7 and self._estimate_tokens([sysmsg] + msgs) > budget:
            del msgs[1]
        if self._estimate_tokens([sysmsg] + msgs) > budget and msgs:
            # Pass 3: truncate whatever is left, from the oldest.
            for m in msgs[:-1]:
                if len(m["content"]) > 1200:
                    m["content"] = m["content"][:1000] + "\n[... trimmed ...]"
        return [sysmsg] + msgs

    # ------------------------------------------------------------------ one model call
    FORCE_PREFIX = '\n<tool_call>\n{"name": "'

    def _call_model(self, prefill: str | None = None) -> tuple[str, list[ToolCall]]:
        """One model call. With `prefill`, the assistant message is started for the model (llama.cpp continues
        it; servers that ignore prefill just produce a fresh message, which we handle too) and nothing is
        rendered: the caller already showed the prose."""
        msgs = self._fit_context()
        if prefill is not None:
            msgs.append({"role": "assistant", "content": prefill})
        with self.ui.status("thinking…" if prefill is None else "deciding…") as _:
            gen = self.client.stream_chat(msgs, max_tokens=1500)
            first = next(gen)   # block here under the spinner until the first token
            if prefill is not None:
                result = None
                content = ""
                for kind, payload in [first] + list(gen):
                    if kind == "text":
                        content += payload
                    else:
                        result = payload
                if content.startswith(prefill):
                    pass                      # server continued our message (llama.cpp)
                elif re.match(r'\s*[a-z_]+"\s*,', content):
                    content = prefill + content   # continuation without echo
                # else: the server started a fresh message; parse it as is.
                if os.environ.get("TASUKE_DEBUG"):
                    with open(STATE_DIR / "debug.log", "a") as dbg:
                        dbg.write(f"\n=== {time.strftime('%T')} forced continuation ===\n{content}\n")
                prose, calls = extract_tool_calls(content, TOOL_NAMES)
                self._note_stats(result)
                return prose, calls
        result = None
        with self.ui.stream() as s:
            kind, payload = first
            if kind == "text":
                s.feed(payload)
            elif kind == "done":
                result = payload
            if result is None:
                for kind, payload in gen:
                    if kind == "text":
                        s.feed(payload)
                    else:
                        result = payload
            content = result.content if result else s.buf
            if os.environ.get("TASUKE_DEBUG"):
                with open(STATE_DIR / "debug.log", "a") as dbg:
                    dbg.write(f"\n=== {time.strftime('%T')} raw model output (finish={getattr(result, 'finish_reason', None)}) ===\n{content}\n")
            prose, calls = extract_tool_calls(content, TOOL_NAMES)
            if not calls and result and result.tool_calls:
                for tc in result.tool_calls:
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {"command": tc["arguments"]}
                    if tc["name"] in TOOL_NAMES:
                        calls.append(ToolCall(name=tc["name"], arguments=args))
            s.finish(prose)
        self._note_stats(result)
        return prose, calls

    def _call_model_with_retry(self) -> tuple[str, list[ToolCall]]:
        """If the server vanished mid-request, bring it back once and retry."""
        try:
            return self._call_model()
        except BackendError as e:
            if self.reconnect is None:
                raise
            self.ui.warn(f"{e}")
            self.ui.info("reconnecting to the model server…")
            self.client = self.reconnect()
            return self._call_model()

    def _note_stats(self, result):
        if result:
            u = result.usage or {}
            self.stats.prompt_tokens = u.get("prompt_tokens", self.stats.prompt_tokens)
            self.stats.gen_tokens += u.get("completion_tokens", 0)
            t = result.timings or {}
            if t.get("predicted_per_second"):
                self.stats.tps = t["predicted_per_second"]

    # ------------------------------------------------------------------ approvals
    def _gate_command(self, cmd: str) -> tuple[str | None, str, str]:
        """Decide whether to run `cmd`. Returns (command_to_run or None, level, note_for_model)."""
        v = safety.classify(cmd)
        if v.level == safety.BLOCKED:
            self.ui.show_blocked(cmd, v.reason)
            return None, v.level, f"not run: {v.reason}. Use a non-interactive alternative."
        head = cmd.split()[0] if cmd.split() else ""
        if self.approval == "dry" and v.level != safety.SAFE:
            self.ui.show_command(cmd, v.level, v.reason, dry=True)
            return None, v.level, ("DRY RUN: this command was not executed. Assume it succeeded and continue so the "
                                   "user can see the whole plan.")
        need = safety.needs_approval(v.level, self.approval)
        if need and v.level == safety.ASK and head in self.always_allow:
            need = False
        self.ui.show_command(cmd, v.level, v.reason)
        if not need:
            return cmd, v.level, ""
        current = cmd
        edited = False
        while True:
            decision, _ = self.ui.approve(v.level, what="run" if not edited else "run edited command")
            if decision == "quit":
                raise UserQuit()
            if decision == "no":
                return None, v.level, "The user declined to run this command."
            if decision == "always":
                self.always_allow.add(head)
                return current, v.level, ""
            if decision == "edit":
                new = self.ui.edit_command(current)
                if not new:
                    return None, v.level, "The user declined to run this command."
                current = new
                edited = True
                v = safety.classify(current)
                self.ui.show_command(current, v.level, v.reason, edited=True)
                if not safety.needs_approval(v.level, self.approval) or v.level == safety.SAFE:
                    return current, v.level, ""
                continue
            return current, v.level, ""

    # ------------------------------------------------------------------ tool dispatch
    def _run_tool(self, call: ToolCall) -> str:
        name, args = call.name, call.arguments
        try:
            if name == "run_shell":
                cmd = str(args.get("command", "")).strip()
                if not cmd:
                    return "error: empty command"
                run_cmd, level, note = self._gate_command(cmd)
                if run_cmd is None:
                    if note.startswith("DRY RUN"):
                        self.changes.append(f"would run: {cmd}")
                    return note
                timeout = args.get("timeout")
                label = "running"
                if re.search(r"\bsudo\b", run_cmd):
                    self.ui.info("(sudo may ask for your password)")
                    res = self.ex.run_shell(run_cmd, timeout=timeout)
                else:
                    with self.ui.status(label) as upd:
                        t0 = time.time()

                        def tick(proc, upd=upd, t0=t0):
                            import threading

                            def loop():
                                while proc.poll() is None:
                                    upd(f"running… {time.time() - t0:.0f}s")
                                    time.sleep(0.5)
                            threading.Thread(target=loop, daemon=True).start()
                        res = self.ex.run_shell(run_cmd, timeout=timeout, on_start=tick)
                self.stats.commands += 1
                if level in (safety.ASK, safety.DANGER):
                    self.changes.append(f"ran: {run_cmd}" + (f"  (exit {res.exit_code})" if res.exit_code else ""))
                self.ui.show_output(res.display, res.exit_code, res.elapsed, res.meta.get("timed_out", False))
                prefix = f"(ran edited command: {run_cmd})\n" if run_cmd != cmd else ""
                return prefix + res.text
            if name == "read_file":
                self.last_file_path = str(self.ex._resolve(str(args.get("path", ""))))
                self.ui.show_file_op("read", str(args.get("path", "")))
                res = self.ex.read_file(str(args.get("path", "")), args.get("start_line"), args.get("max_lines"))
                self.ui.print(f"[meta]  {res.display}[/]")
                return res.text
            if name == "write_file":
                path, content = str(args.get("path", "")), str(args.get("content", ""))
                p, diff, exists = self.ex.diff_for_write(path, content)
                self.last_file_path = str(p)
                self.ui.show_file_op("write" if exists else "create", str(p))
                self.ui.show_diff(diff)
                if self.approval == "dry":
                    self.changes.append(f"would {'write' if exists else 'create'}: {p}")
                    return "DRY RUN: file not written. Assume it was and continue."
                if self.approval != "yolo":
                    decision, _ = self.ui.approve("ask", what="write this file")
                    if decision == "quit":
                        raise UserQuit()
                    if decision == "no":
                        return "The user declined this write."
                res = self.ex.write_file(path, content)
                self.changes.append(f"{'wrote' if exists else 'created'}: {p}  ({_diff_stat(diff)})")
                self.ui.print(f"[ok]  {res.display}[/]")
                return res.text
            if name == "edit_file":
                path = str(args.get("path", ""))
                p, updated, diff = self.ex.prepare_edit(path, str(args.get("old", "")), str(args.get("new", "")))
                self.last_file_path = str(p)
                self.ui.show_file_op("edit", str(p))
                if updated is None:
                    self.ui.print(f"[err]  {diff}[/]")
                    return diff
                self.ui.show_diff(diff)
                if self.approval == "dry":
                    self.changes.append(f"would edit: {p}  ({_diff_stat(diff)})")
                    return "DRY RUN: edit not applied. Assume it was and continue."
                if self.approval != "yolo":
                    decision, _ = self.ui.approve("ask", what="apply this edit")
                    if decision == "quit":
                        raise UserQuit()
                    if decision == "no":
                        return "The user declined this edit."
                res = self.ex.apply_edit(p, updated)
                self.changes.append(f"edited: {p}  ({_diff_stat(diff)})")
                self.ui.print(f"[ok]  {res.display}[/]")
                return res.text
            if name == "ask_user":
                q = str(args.get("question", ""))
                if is_permission_question(q):
                    self.ui.info(f"(model asked: {q.strip()[:90]} — answered yes; the approval prompt is the real gate)")
                    return "Yes, go ahead. Commands that change state still go through the user's approval prompt, so just call the tool."
                ans = self.ui.ask_user(q)
                if ans is None:
                    return ("(No user is available to answer right now. Do not guess or invent a value. If you can find "
                            "the answer by running commands, do that; otherwise stop and state exactly what you need from the user.)")
                return f"User answered: {ans}"
        except UserQuit:
            raise
        except Exception as e:  # a tool bug must not kill the session
            return f"error: tool {name} failed: {e}"
        return f"error: unknown tool {name}"

    # ------------------------------------------------------------------ pre-flight
    def preflight_run(self, command: str) -> str | None:
        """Run a command the user already named (through the normal approval gate) and return the
        tool-style output, or None if it was declined/blocked. Saves the model its first round-trip."""
        call = ToolCall(name="run_shell", arguments={"command": command}, raw="preflight")
        out = self._run_tool(call)
        if out.startswith("not run:") or out.startswith("The user declined"):
            return None
        return out

    # ------------------------------------------------------------------ a full turn
    def run_turn(self, user_text: str) -> str:
        self.stats = TurnStats()
        self.messages.append({"role": "user", "content": user_text})
        last_cmd_sig = None
        repeat = 0
        seen: dict[str, int] = {}   # every tool-call signature this turn → how often
        nudges = 0
        final = ""
        for rnd in range(self.max_rounds):
            self.stats.rounds = rnd + 1
            try:
                prose, calls = self._call_model_with_retry()
            except BackendError as e:
                self.ui.error(str(e))
                self.messages.append({"role": "assistant", "content": "(no response: the model server was unreachable)"})
                return ""
            except KeyboardInterrupt:
                self.ui.console.print()
                self.ui.info("interrupted")
                self.messages.append({"role": "assistant", "content": "(interrupted by the user before finishing)"})
                return ""
            if not calls and looks_like_unfinished_intent(prose):
                # The model announced a step and stopped. Start the tool call for it and let it finish the JSON.
                try:
                    _, calls = self._call_model(prefill=prose.rstrip() + self.FORCE_PREFIX)
                except BackendError:
                    calls = []
                if not calls:
                    calls = self._infer_calls(prose)
                    if calls:
                        self.ui.info("(the model described this step instead of calling the tool; doing what it described)")
            if not calls:
                self.messages.append({"role": "assistant", "content": prose or "(no response)"})
                # Nudge at most twice in a row; the counter resets whenever a real tool call happens.
                if nudges < 2 and looks_like_unfinished_intent(prose):
                    nudges += 1
                    self.messages.append({"role": "user", "content": NUDGE})
                    continue
                final = prose
                break
            nudges = 0
            # Persist the assistant turn in canonical form so the model sees the format it should use.
            assistant_content = (prose.strip() + "\n" if prose.strip() else "") + "\n".join(c.canonical() for c in calls)
            self.messages.append({"role": "assistant", "content": assistant_content})
            # Execute (small models usually send one; honour several in order).
            responses = []
            stuck = False
            for c in calls:
                key = call_signature(c)
                repeat = repeat + 1 if key == last_cmd_sig else 0
                last_cmd_sig = key
                seen[key] = seen.get(key, 0) + 1
                out = self._run_tool(c)
                if self.ex.last_interrupted:
                    self.ex.last_interrupted = False
                    responses.append(f"<tool_response>\n{out}\n</tool_response>")
                    self.messages.append({"role": "user", "content": "\n".join(responses)})
                    self.ui.info("stopped; tell me how to proceed")
                    return ""
                if repeat >= 1 or seen[key] >= 2:
                    out += ("\n\nNOTE: you have made this exact call before in this task and it did not resolve the "
                            "problem. Do not repeat it. Step back: re-read the original error, reconsider the diagnosis, "
                            "and change approach, or give your final answer explaining what you found.")
                if repeat >= 2 or seen[key] >= 3:
                    stuck = True
                responses.append(f"<tool_response>\n{out}\n</tool_response>")
            self.messages.append({"role": "user", "content": "\n".join(responses)})
            if stuck:
                self.ui.warn("the model keeps repeating the same failing step; stopping here. Tell it what to do differently.")
                final = ""
                break
        else:
            self.ui.warn(f"stopped after {self.max_rounds} steps without a final answer. Say 'continue' to keep going.")
            final = ""
        self.last_blocks = shell_blocks(final) if final else []
        return final

    # ------------------------------------------------------------------ recovering narrated actions
    def _infer_calls(self, prose: str) -> list[ToolCall]:
        """A small model often *shows* the fix (a file in a code block, a command in another) instead of calling
        a tool. Turn the first such block into the tool call it obviously meant. Approval still applies."""
        fences = _FENCE_RE.findall(prose)
        if not fences:
            return []
        for lang, body in fences:
            lang = lang.lower()
            body_s = body.strip("\n")
            is_file = lang in _FILE_LANGS or (lang == "" and (body_s.lstrip()[:1] in "{[<" or body_s.startswith("#!")))
            if not is_file:
                continue
            target = None
            for m in _PATH_RE.finditer(prose):
                cand = self.ex._resolve(m.group(1))
                if cand.is_file():
                    target = str(cand)
                    break
            target = target or self.last_file_path
            if target:
                return [ToolCall(name="write_file", arguments={"path": target, "content": body_s + "\n"}, raw="inferred")]
        shells = shell_blocks(prose)
        if shells and looks_like_unfinished_intent(prose):
            return [ToolCall(name="run_shell", arguments={"command": shells[0]}, raw="inferred")]
        return []

    def run_block(self, idx: int) -> str:
        """User typed a number: run that suggested block as if the model had asked for it."""
        cmd = self.last_blocks[idx]
        call = ToolCall(name="run_shell", arguments={"command": cmd})
        self.messages.append({"role": "user", "content": f"Run suggested command #{idx + 1} for me."})
        self.messages.append({"role": "assistant", "content": call.canonical()})
        out = self._run_tool(call)
        self.messages.append({"role": "user", "content": f"<tool_response>\n{out}\n</tool_response>\nNow continue: interpret the result and finish the task."})
        # Let the model react to the result.
        self.stats = TurnStats()
        prose, calls = self._call_model()
        if calls:
            # It wants to keep going; hand back to the regular loop by replaying as a turn-less continuation.
            self.messages.append({"role": "assistant", "content": "\n".join(c.canonical() for c in calls)})
            responses = [f"<tool_response>\n{self._run_tool(c)}\n</tool_response>" for c in calls]
            self.messages.append({"role": "user", "content": "\n".join(responses)})
            return self.run_turn("Continue from the result above and finish; do not repeat work.")
        self.messages.append({"role": "assistant", "content": prose})
        self.last_blocks = shell_blocks(prose)
        return prose
