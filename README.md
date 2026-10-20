# tasuke

A terminal assistant that runs on a **local LLM** and fixes things by **actually running
commands on your machine**: it looks at processes, ports, logs and files, diagnoses from the
real output, applies the fix with your approval, and verifies it. No network, no API key.

```
$ ?? why can't I reach my dev server on 8080
$ lsof -nP -iTCP:8080 -sTCP:LISTEN
(no output)
$ ps aux | grep -i '[v]ite\|[n]ext\|[n]ode'
...
Nothing is listening on 8080. Your dev server crashed on start: `.env` is missing
`DATABASE_URL`. Added it from `.env.example` and restarted; it's up on 8080 now.
```

## How it's wired

| Piece | What |
|---|---|
| Model | any `.gguf` via **llama.cpp**'s `llama-server`; or an existing Ollama / LM Studio / vLLM endpoint |
| Agent loop | model → tool call → **risk-classified** execution → output → model, until it answers |
| Tools | `run_shell`, `read_file`, `edit_file`, `write_file`, `ask_user` |
| Shell | `??` alias, **Ctrl-G** natural-language → command, `tasuke fix` for the command that just failed, pipe support |

Tool calls are parsed tolerantly (small models wrap them in ```` ```json ````, `<tools>`, bare
JSON...), and a model that *narrates* a fix instead of doing it gets nudged into making the call.

## Install

```sh
brew install llama.cpp          # if you don't have llama-server yet
uv tool install -e ~/code/tasuke
tasuke init                     # writes ~/.config/tasuke/config.toml, installs the zsh integration
```

Drop a GGUF in `~/models` (or point `model_dirs` at where yours live). `tasuke models` shows what it
found and which one it will pick: the largest that fits in ~40% of RAM, so a 24 GB Mac gets
Qwen2.5-Coder-14B Q4_K_M (~20 tok/s on Apple silicon) and a 16 GB one gets the 7B. The 14B is
noticeably more careful about edits than the 7B; the 7B is fine for questions and read-only
diagnosis. The server starts on first use (about 10 s to load) and stays up between invocations.

## Use

```sh
?? what is eating my disk                   # investigates with du/df, reports, offers cleanup
?? the brew install of postgres failed      # reads logs, fixes, retries
tasuke fix                                  # after any command fails: re-runs it, diagnoses, fixes
make 2>&1 | ?? why does this fail           # pipe anything in
tasuke                                      # interactive; `!cmd` runs a command the model also sees
tasuke -c                                   # pick up the last session
```

Type something like `find files bigger than 1G modified this week` and hit **Ctrl-G**: the line
is replaced with the command. Nothing runs until you press Enter. About 1.1 s on the 14B here
(0.5 s on the 7B fast tier, see below).

```sh
tasuke -n "upgrade postgres and migrate my db"   # dry run: read-only commands run, changes are only shown
tasuke log                                       # past sessions: request · outcome · what changed
```

Every turn that changed something ends with a `changed:` block: commands that altered state and files
written or edited, with `+n -m` line counts.

**Pre-flight.** `tasuke fix` runs the failing command itself (through the same approval gate) and hands
the model the real error plus a directory listing before the first model call; `??` does the same when
your request names a script that "fails". That removes the exploratory first round-trips.

## Fast tier

`[fast]` in the config names a second, smaller model for `suggest` (Ctrl-G) and `explain`, with fixes
staying on the main model. It is `"auto"` by default: on only when both models fit in half of RAM.
On this 24 GB machine the 14B + 7B (13.7 GB) do not qualify, and running both pushed the machine into
swap, so it stays off here; set `enabled = true` to force it. Measured Ctrl-G wall clock, warm:

| model | suggest latency |
|---|---|
| 14B (default here) | 1.1 to 1.4 s |
| 7B fast tier | 0.5 to 0.75 s |

Shell integration exists for **zsh** (`tasuke.zsh`) and **bash** (`tasuke.bash`: PROMPT_COMMAND hook,
`??` function, Ctrl-G via `bind -x`); `tasuke init` picks the one matching `$SHELL`.

## Approval model

Every command is classified before it runs:

| level | examples | default behaviour |
|---|---|---|
| safe | `ls`, `lsof`, `git status`, `brew list`, `curl -I` | runs immediately |
| ask | `brew install`, `kill`, `git add`, `sed -i`, `>` redirects | `y`/`n`/`e`dit/`a`lways |
| danger | `rm -rf`, `dd`, `git push --force`, `curl … \| sh` | must type `yes` |
| blocked | `vim`, `top`, `tail -f`, `ssh host` | never runs; the model is told to use a non-interactive form |

`-y` confirms only dangerous commands; `--yolo` confirms nothing; `-n`/`--dry-run` runs only read-only
commands and shows the rest as a plan. File writes always show a diff.
Anything unrecognised is "ask", on purpose.

## Config

`~/.config/tasuke/config.toml` (see `tasuke config`). Notable keys: `[server].base_url`,
`[server].model`, `[server].model_dirs`, `[agent].approval`, `[agent].command_timeout`.

Using Ollama instead of llama.cpp:

```toml
[server]
base_url = "http://127.0.0.1:11434"
manage = false
model_name = "qwen2.5-coder:7b"
```

## Tests

```sh
uv run pytest            # unit tests: classifier, parser, executor, backend, agent fallbacks
```

## Known limits

- **It is a 14B model.** It is good at reading real output and taking the obvious next step, and
  fine at small edits. It will occasionally fix the wrong thing confidently; that is what the
  approval prompt and the diff are for. The 7B is noticeably weaker at edits.
- Commands run non-interactively with no TTY. Anything that needs a prompt (sudo password aside,
  which goes to the terminal directly) has to be given a non-interactive flag; the classifier
  blocks the common interactive programs and tells the model why.
- `cd` only persists when it is the whole command; otherwise chain with `&&`.
- Context is 16k tokens by default; long sessions have old command output trimmed, so ask again if it
  has forgotten something.
- Hooks cost ~0.01 ms per command and ~0.1 ms at shell startup (zsh; bash uses PROMPT_COMMAND + history).
- The first call after a cold start pays for loading the model (~10 s for the 14B) and for
  processing the ~1.4k-token system prompt; later calls reuse the KV cache.

## Layout

```
tasuke/
  cli.py       entry point & subcommands        agent.py    the loop, approvals, context trimming
  backend.py   OpenAI-compatible client, llama-server lifecycle, model discovery
  tools.py     executor (shell / files)         safety.py   command risk classifier
  parse.py     tolerant tool-call extraction    prompts.py  system prompts
  ui.py        rich rendering & prompts         session.py  persistence
  shell/tasuke.zsh                              zsh hooks, ??, Ctrl-G
```

Sessions live in `~/.local/share/tasuke/sessions`, the server log in
`~/.local/state/tasuke/llama-server.log`. `TASUKE_DEBUG=1` logs raw model output to
`~/.local/state/tasuke/debug.log`.
