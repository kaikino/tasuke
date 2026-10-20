# tasuke shell integration for bash. Installed by `tasuke init`; sourced from ~/.bashrc.
#   ?? <anything>   ask tasuke (globbing off, so ? and * in the question are literal)
#   Ctrl-G          replace the natural-language line you're typing with a shell command (does not run it)
#   tasuke fix      diagnose and fix the command that just failed

# ---- remember the last command and its exit status, for `tasuke fix` ----------------
# bash has no preexec; PROMPT_COMMAND runs before each prompt, so look at the previous command via
# history. HISTCONTROL=ignorespace etc. may hide some commands; then TASUKE_LAST_CMD is simply stale.
_tasuke_precmd() {
  local st=$?
  local last
  last=$(HISTTIMEFORMAT= builtin history 1 2>/dev/null)
  last=${last#*[0-9]  }          # strip the leading history number
  if [[ -n $last && $last != "$__tasuke_seen" ]]; then
    __tasuke_seen=$last
    export TASUKE_LAST_CMD=$last TASUKE_LAST_STATUS=$st
  fi
}
case ";$PROMPT_COMMAND;" in
  *";_tasuke_precmd;"*) ;;
  *) PROMPT_COMMAND="_tasuke_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;;
esac

# ---- ?? ---------------------------------------------------------------------------------
# The `function` keyword accepts names a plain `name()` definition rejects. `set -f` = noglob,
# scoped to the function by `local -`.
function ?? { local -; set -f; tasuke "$@"; }
command -v fix >/dev/null 2>&1 || alias fix='tasuke fix'

# ---- Ctrl-G: natural language -> command, in place -------------------------------------
_tasuke_suggest_widget() {
  [[ -z $READLINE_LINE ]] && return 0
  local out
  out=$(command tasuke suggest -- "$READLINE_LINE" 2>/dev/null)
  if [[ -n $out ]]; then
    READLINE_LINE=$out
    READLINE_POINT=${#READLINE_LINE}
  fi
}
if [[ $- == *i* ]]; then
  bind -x '"\C-g": _tasuke_suggest_widget'
fi
