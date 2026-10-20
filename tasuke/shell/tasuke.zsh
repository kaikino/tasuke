# tasuke shell integration. Installed by `tasuke init`; sourced from ~/.zshrc.
#
#   ?? <anything>     ask tasuke (no quoting needed: `?? why is port 8080 busy?`)
#   Ctrl-G            turn the natural-language line you're typing into a shell command (does not run it)
#   tasuke fix        diagnose and fix the command that just failed
#   cmd 2>&1 | ??     hand tasuke some output to diagnose

# ---- remember the last command and its exit status, for `tasuke fix` ----------------
autoload -Uz add-zsh-hook
_tasuke_preexec() { __tasuke_cmd="$1" }
_tasuke_precmd() {
  local st=$?
  if [[ -n $__tasuke_cmd ]]; then
    export TASUKE_LAST_CMD="$__tasuke_cmd" TASUKE_LAST_STATUS=$st
    __tasuke_cmd=""
  fi
}
add-zsh-hook preexec _tasuke_preexec
add-zsh-hook precmd  _tasuke_precmd

# ---- ?? ---------------------------------------------------------------------------------
# noglob: a trailing "?" or a "*" in your question must not be expanded as a pattern.
alias -- '??'='noglob tasuke'
# `fix` only if nothing else already owns the name.
(( $+commands[fix] )) || alias fix='tasuke fix'

# ---- Ctrl-G: natural language -> command, in place -------------------------------------
_tasuke_suggest_widget() {
  [[ -z $BUFFER ]] && return 0
  zle -M "tasuke: thinking…"
  local out
  out=$(command tasuke suggest -- "$BUFFER" 2>/dev/null)
  if [[ -n $out ]]; then
    BUFFER="$out"
    CURSOR=${#BUFFER}
    zle -M ""
  else
    zle -M "tasuke: no suggestion (is the model server up? try: tasuke status)"
  fi
  zle redisplay
}
zle -N _tasuke_suggest_widget
bindkey '^G' _tasuke_suggest_widget

# ---- completion for subcommands -----------------------------------------------------------
_tasuke() {
  local -a subs
  subs=('fix:diagnose the last failed command' 'suggest:one command for a request' 'explain:explain a command'
        'serve:start the model server' 'stop:stop it' 'status:is it up' 'models:list .gguf files'
        'config:show config' 'init:install shell integration' 'sessions:list sessions')
  if (( CURRENT == 2 )); then
    _describe 'subcommand' subs
  fi
}
compdef _tasuke tasuke 2>/dev/null
