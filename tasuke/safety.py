"""Command risk classification. Decides what runs without asking.

Levels:
  safe    read-only; runs automatically (unless approval=ask has been tightened).
  ask     changes state (installs, writes, kills a process); confirm first.
  danger  destructive or irreversible; needs an explicit typed "yes".
  blocked interactive / never-ending programs that can't run without a TTY.

Deliberately conservative: anything the classifier doesn't recognise is "ask".
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

SAFE, ASK, DANGER, BLOCKED = "safe", "ask", "danger", "blocked"
_ORDER = {SAFE: 0, ASK: 1, DANGER: 2, BLOCKED: 3}


@dataclass
class Verdict:
    level: str
    reason: str = ""

    def __lt__(self, other: "Verdict"):
        return _ORDER[self.level] < _ORDER[other.level]


# ---------------------------------------------------------------- destructive patterns
_DANGER = [
    (r"\brm\b.*\s-[a-zA-Z]*[rR][a-zA-Z]*\b", "recursive rm"),
    (r"\brm\b.*\s-[a-zA-Z]*f[a-zA-Z]*\b", "forced rm"),
    (r"\brm\s+(?:-\S+\s+)*(/|~|\$HOME|\*|\.\.?)(\s|$)", "rm of a top-level path"),
    (r"\brmdir\s+(/|~)(\s|$)", "rmdir of a top-level path"),
    (r"\bfind\b.*\s-(delete|exec\s+rm)\b", "find -delete / -exec rm"),
    (r"\bxargs\b.*\brm\b", "xargs rm"),
    (r"\bdd\b", "dd"),
    (r"\bmkfs(\.\w+)?\b", "mkfs"),
    (r"\bdiskutil\s+(erase\w*|partitionDisk|reformat|zeroDisk|secureErase|apfs\s+delete\w*|resetFusion)", "diskutil erase/partition"),
    (r"\bshred\b|\bwipefs\b", "wipe"),
    (r">\s*/dev/(sd|disk|nvme|hd|rdisk)", "write to a block device"),
    (r"\bchmod\s+(-R\s+)?[0-7]*[67]{3}\b", "chmod 777-ish"),
    (r"\bch(mod|own|grp)\s+-[a-zA-Z]*R\b.*\s/(\s|$)", "recursive chmod/chown on /"),
    (r":\s*\(\s*\)\s*\{", "fork bomb"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|da|k|fi)?sh\b", "pipe download into a shell"),
    (r"\bgit\s+push\b.*(--force\b|\s-f\b|--force-with-lease\b|\+\w)", "git force push"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+clean\b.*\s-[a-zA-Z]*[fdx]", "git clean -f"),
    (r"\bgit\s+checkout\s+(--\s+)?\.(\s|$)|\bgit\s+restore\s+(\.|--staged\s+\.|:/)(\s|$)", "discard all working changes"),
    (r"\bgit\s+branch\s+-D\b", "force-delete branch"),
    (r"\bgit\s+(stash\s+(drop|clear)|filter-branch|filter-repo)\b", "git history destruction"),
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b|\bTRUNCATE\s+TABLE\b|\bDELETE\s+FROM\b(?!.*\bWHERE\b)", "destructive SQL"),
    (r"\bkill\s+(-\w+\s+)?-1\b|\bkillall5\b|\bpkill\s+-9\s+-f\s+\.\*|\bkillall\s+(-9\s+)?(Finder|WindowServer|loginwindow|launchd|kernel_task)", "kill everything"),
    (r"\b(shutdown|reboot|halt|poweroff)\b|\binit\s+[06]\b", "shutdown/reboot"),
    (r"\blaunchctl\s+(unload|remove|bootout|disable)\b|\bsystemctl\s+(disable|mask|stop)\b.*\b(ssh|sshd|network|systemd-)", "disable system service"),
    (r"\bcsrutil\b|\bnvram\b|\bspctl\s+--master-disable|\bbless\b", "firmware/security settings"),
    (r"\bcrontab\s+-r\b", "wipe crontab"),
    (r"\bhistory\s+-c\b", "clear history"),
    (r"--no-preserve-root", "--no-preserve-root"),
    (r"\b(truncate|:)\s*>\s*\S+", "truncate file"),
    (r">\s*~?/?\S*\.(zshrc|bashrc|zprofile|bash_profile|profile|gitconfig|ssh/\S+)(\s|$)", "overwrite a dotfile"),
    (r"\bdefaults\s+delete\b", "defaults delete"),
    (r"\bbrew\s+(uninstall|remove|rm|zap)\b.*--force|\bbrew\s+cleanup\s+--prune=all", "forced brew uninstall"),
    (r"\b(docker|podman)\s+(system\s+prune|volume\s+(rm|prune)|rm\s+-f|rmi\s+-f|container\s+prune)", "docker destructive"),
    (r"\bkubectl\s+delete\b.*(--all\b|namespace|ns\b)", "kubectl delete all/namespace"),
    (r"\bterraform\s+(destroy|apply\s+-auto-approve)", "terraform destroy/auto-approve"),
    (r"\b(aws\s+s3\s+(rb|rm)|gcloud\s+.*delete|az\s+.*delete)\b", "cloud delete"),
    (r"\bmv\s+(\S+\s+)*(/|~|\$HOME)(\s|$)", "mv over a top-level path"),
    (r"\bsecurity\s+delete-", "keychain delete"),
    (r"\bpip\s+uninstall\s+-y\s+.*\b(pip|setuptools)\b", "uninstall pip"),
    (r"\bnpm\s+(cache\s+clean\s+--force|uninstall\s+-g\s+npm)\b", "npm self-destruct"),
]
_DANGER_RE = [(re.compile(p, re.IGNORECASE if "SQL" in why else 0), why) for p, why in _DANGER]

# ---------------------------------------------------------------- interactive / endless programs
_BLOCKED_HEADS = {
    "vim", "vi", "nvim", "nano", "emacs", "pico", "ed", "less", "more", "most", "htop", "btop", "atop",
    "glances", "nmon", "iotop", "iftop", "nethogs", "ranger", "mc", "tmux", "screen", "watch", "lazygit",
    "lazydocker", "k9s", "ncdu", "tig", "gdb", "lldb", "irb", "pry", "ghci", "ipython", "jshell",
    "mysql", "psql", "sqlite3", "redis-cli", "mongo", "mongosh", "telnet", "ftp", "sftp", "nc",
    "fzf", "sk", "dialog", "whiptail", "passwd", "visudo", "vipw", "crontab",
}
_BLOCKED_PATTERNS = [
    (r"^tail\s+.*-[a-zA-Z]*[fF]\b", "tail -f never exits; use `tail -n 100` instead"),
    (r"^top\b(?!.*-l\s*\d)", "top is interactive; use `top -l 1 -n 20` (macOS) or `top -bn1` (Linux)"),
    (r"^journalctl\b.*\s-f\b", "journalctl -f never exits; use `journalctl -n 200 --no-pager`"),
    (r"^log\s+stream\b", "log stream never exits; use `log show --last 5m`"),
    (r"^ping\b(?!.*\s-c\s*\d)", "ping without -c never exits (it is rewritten to -c 4)"),
    (r"^(python3?|node|ruby|irb|perl|lua|php|R|julia|ghci|erl|iex|clj|scala|sbt|bash|zsh|sh|fish)\s*$",
     "bare interpreter would wait for stdin; pass -c/-e or a script"),
    (r"^ssh\s+(?:-\S+\s+)*\S+\s*$", "ssh without a remote command opens an interactive session; append the command"),
    (r"^(sudo\s+)?(apt|apt-get|dnf|yum)\s+(install|remove|upgrade|dist-upgrade|autoremove)\b(?!.*\s-y\b)",
     "will prompt; add -y"),
    (r"^(git\s+)?(rebase|add)\s+(-i|--interactive|-p|--patch)\b", "interactive git mode"),
    (r"^(docker|podman)\s+(run|exec)\b.*\s-[a-zA-Z]*[it]{2}", "docker -it allocates a TTY"),
    (r"^man\b", "man pages page; use `man -P cat <topic> | head -80` or `<cmd> --help`"),
]
_BLOCKED_RE = [(re.compile(p), why) for p, why in _BLOCKED_PATTERNS]

# ---------------------------------------------------------------- read-only heads
_SAFE_HEADS = {
    # files & text
    "ls", "eza", "exa", "lsd", "tree", "cat", "bat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "ag",
    "ack", "find", "fd", "fdfind", "locate", "mdfind", "mdls", "which", "whereis", "whatis", "type", "command",
    "file", "stat", "du", "df", "sort", "uniq", "cut", "tr", "column", "jq", "yq", "diff", "cmp", "comm", "xxd",
    "hexdump", "strings", "od", "nl", "tac", "rev", "fold", "paste", "join", "expand", "unexpand", "seq", "bc",
    "expr", "basename", "dirname", "realpath", "readlink", "pwd", "echo", "printf", "true", "false", "test",
    "date", "cal", "env", "printenv", "md5", "md5sum", "shasum", "sha1sum", "sha256sum", "cksum", "b2sum",
    "tee",  # tee writes, but to a path we check for below
    # processes & system
    "ps", "pgrep", "lsof", "uptime", "top", "sleep", "uname", "hostname", "whoami", "id", "groups", "who", "w", "last", "nproc",
    "free", "vm_stat", "vmstat", "iostat", "sar", "sw_vers", "system_profiler", "sysctl", "sysinfo", "arch",
    "getconf", "ulimit", "umask", "pmset", "ioreg", "kextstat", "dmesg", "log", "last", "fs_usage",
    "launchctl", "systemctl", "journalctl", "service", "pstree", "procs", "nettop",
    # network (read)
    "ifconfig", "ip", "ipconfig", "route", "netstat", "ss", "arp", "ping", "ping6", "traceroute", "tracepath",
    "mtr", "dig", "nslookup", "host", "whois", "scutil", "networksetup", "nc", "curl", "openssl", "ssh-keygen",
    # package managers (queries only; mutations caught below)
    "brew", "port", "apt", "apt-cache", "apt-get", "dpkg", "dpkg-query", "rpm", "dnf", "yum", "pacman", "apk",
    "pip", "pip3", "uv", "pipx", "npm", "pnpm", "yarn", "bun", "cargo", "go", "gem", "bundle", "mise", "asdf",
    "nvm", "pyenv", "rbenv", "jenv", "rustup", "conda", "poetry",
    # dev tools (queries)
    "git", "gh", "docker", "podman", "kubectl", "helm", "terraform", "make", "cmake", "xcode-select", "xcodebuild",
    "swift", "swiftc", "clang", "gcc", "cc", "ld", "otool", "nm", "ldd", "lipo", "codesign", "spctl", "plutil",
    "defaults", "security", "python", "python3", "node", "ruby", "perl", "java", "javac", "rustc", "awk", "sed",
    "xargs", "timeout", "gtimeout", "time", "nice", "caffeinate", "mount", "diskutil", "hdiutil", "tmutil",
    "softwareupdate", "mdutil", "say", "open", "pbpaste", "osascript", "sqlite3", "lsblk", "blkid", "fdisk",
    "smartctl", "sensors", "lscpu", "lspci", "lsusb", "lshw", "hwinfo", "numfmt", "units", "zcat", "gzcat",
    "gunzip", "tar", "unzip", "zipinfo", "7z", "gpg",
}
# Sub-commands of multi-tool CLIs that mutate state (anything else from a safe head stays safe).
_MUTATING_SUBCMDS = {
    "git": {"add", "commit", "push", "pull", "merge", "rebase", "reset", "checkout", "switch", "restore", "rm",
            "mv", "clean", "stash", "cherry-pick", "revert", "am", "apply", "tag", "branch", "remote", "init",
            "clone", "submodule", "worktree", "gc", "prune", "filter-branch", "notes", "bisect", "config"},
    "gh": {"pr", "issue", "repo", "release", "api", "auth", "gist", "secret", "variable", "workflow", "run"},
    "brew": {"install", "uninstall", "remove", "rm", "reinstall", "upgrade", "update", "link", "unlink", "tap",
             "untap", "cleanup", "pin", "unpin", "services", "autoremove", "bundle", "cask", "postinstall"},
    "port": {"install", "uninstall", "upgrade", "selfupdate", "clean", "activate", "deactivate"},
    "apt": {"install", "remove", "purge", "upgrade", "update", "dist-upgrade", "autoremove", "full-upgrade",
            "edit-sources"},
    "apt-get": {"install", "remove", "purge", "upgrade", "update", "dist-upgrade", "autoremove", "clean", "autoclean"},
    "dpkg": {"-i", "--install", "-r", "--remove", "-P", "--purge", "--configure"},
    "dnf": {"install", "remove", "erase", "upgrade", "update", "autoremove", "clean", "reinstall", "swap"},
    "yum": {"install", "remove", "erase", "upgrade", "update", "autoremove", "clean", "reinstall"},
    "pacman": {"-S", "-R", "-U", "-Sy", "-Syu", "-Rs", "-Rns", "-Sc", "-Scc"},
    "apk": {"add", "del", "upgrade", "update", "fix"},
    "pip": {"install", "uninstall", "download", "cache"}, "pip3": {"install", "uninstall", "download", "cache"},
    "uv": {"add", "remove", "sync", "lock", "run", "tool", "python", "venv", "init", "build", "publish", "cache"},
    "pipx": {"install", "uninstall", "upgrade", "inject", "reinstall", "run", "ensurepath"},
    "npm": {"install", "i", "add", "uninstall", "remove", "rm", "un", "update", "up", "run", "start", "test",
            "publish", "link", "unlink", "ci", "init", "exec", "x", "cache", "rebuild", "prune", "dedupe", "audit", "set", "config"},
    "pnpm": {"install", "i", "add", "remove", "rm", "update", "up", "run", "start", "test", "publish", "link", "dlx", "exec", "store", "prune"},
    "yarn": {"install", "add", "remove", "upgrade", "run", "start", "test", "publish", "link", "cache", "init", "set", "dlx"},
    "bun": {"install", "i", "add", "remove", "rm", "update", "run", "x", "create", "init", "link", "publish", "pm"},
    "cargo": {"install", "uninstall", "build", "run", "test", "bench", "publish", "update", "add", "remove", "new",
              "init", "clean", "fix", "fmt", "clippy"},
    "go": {"build", "install", "run", "test", "get", "mod", "generate", "clean", "fmt", "fix", "work", "vet"},
    "gem": {"install", "uninstall", "update", "cleanup", "pristine", "push"},
    "bundle": {"install", "update", "exec", "add", "remove", "clean"},
    "mise": {"install", "use", "uninstall", "upgrade", "set", "unset", "exec", "x", "run", "r", "prune", "reshim", "self-update"},
    "asdf": {"install", "uninstall", "global", "local", "shell", "plugin", "update", "reshim"},
    "rustup": {"install", "update", "default", "toolchain", "target", "component", "override", "self"},
    "conda": {"install", "remove", "update", "create", "env", "clean"}, "poetry": {"install", "add", "remove", "update", "run", "build", "publish", "lock"},
    "docker": {"run", "exec", "start", "stop", "restart", "kill", "rm", "rmi", "pull", "push", "build", "create",
               "commit", "tag", "load", "save", "import", "cp", "pause", "unpause", "rename", "update", "attach",
               "login", "logout", "prune", "compose", "swarm", "stack", "service", "network", "volume", "system",
               "container", "image", "buildx", "context"},
    "podman": {"run", "exec", "start", "stop", "restart", "kill", "rm", "rmi", "pull", "push", "build", "create",
               "commit", "tag", "load", "save", "import", "cp", "pause", "unpause", "rename", "update", "attach",
               "login", "logout", "prune", "compose", "pod", "network", "volume", "system", "container", "image"},
    "kubectl": {"apply", "create", "delete", "edit", "patch", "replace", "scale", "rollout", "set", "label",
                "annotate", "taint", "cordon", "uncordon", "drain", "exec", "run", "expose", "cp", "attach",
                "port-forward", "proxy", "certificate", "debug"},
    "helm": {"install", "uninstall", "upgrade", "rollback", "delete", "repo", "push", "create", "package", "plugin"},
    "terraform": {"apply", "destroy", "import", "init", "taint", "untaint", "state", "workspace", "force-unlock", "login", "logout"},
    "make": None,   # None = any invocation mutates (builds write files)
    "cmake": None, "xcodebuild": None, "swift": None, "swiftc": None, "clang": None, "gcc": None, "cc": None, "ld": None,
    "launchctl": {"load", "unload", "start", "stop", "kickstart", "kill", "bootstrap", "bootout", "enable",
                  "disable", "remove", "submit", "setenv", "unsetenv", "limit", "reboot", "config"},
    "systemctl": {"start", "stop", "restart", "reload", "enable", "disable", "mask", "unmask", "edit", "set-property",
                  "daemon-reload", "daemon-reexec", "kill", "reset-failed", "isolate", "set-default", "link", "revert",
                  "poweroff", "reboot", "halt", "suspend", "hibernate"},
    "service": None,
    "defaults": {"write", "delete", "import", "rename"},
    "security": {"add-generic-password", "delete-generic-password", "add-internet-password", "delete-internet-password",
                 "import", "export", "unlock-keychain", "lock-keychain", "create-keychain", "delete-keychain",
                 "set-keychain-settings", "add-trusted-cert", "remove-trusted-cert", "authorizationdb"},
    "networksetup": None,  # has -list*/-get* reads, handled below by flag check
    "scutil": None,
    "sysctl": None,        # -w writes; handled below
    "diskutil": {"mount", "unmount", "umount", "eject", "rename", "enableJournal", "disableJournal", "repairDisk",
                 "repairVolume", "verifyDisk", "verifyVolume", "apfs", "coreStorage", "cs", "resizeVolume",
                 "splitPartition", "mergePartitions", "addPartition", "eraseVolume", "mountDisk", "unmountDisk"},
    "hdiutil": {"attach", "detach", "create", "convert", "resize", "burn", "eject", "mount", "unmount"},
    "tmutil": {"startbackup", "stopbackup", "enable", "disable", "setdestination", "removedestination", "delete",
               "deletelocalsnapshots", "localsnapshot", "thinlocalsnapshots", "restore", "setexclusion", "removeexclusion", "addexclusion"},
    "softwareupdate": {"-i", "--install", "-ia", "--install-rosetta", "-d", "--download", "--schedule"},
    "mdutil": {"-E", "-i", "-a"},
    "mount": None, "say": None, "open": None, "osascript": None, "caffeinate": None,
    "sqlite3": None, "openssl": None, "ssh-keygen": None, "gpg": None,
    "tar": {"-x", "x", "-c", "c", "--extract", "--create", "-xf", "-xvf", "-xzf", "-xjf", "-cf", "-czf", "-cvf", "-cvzf", "-xzvf", "-czvf"},
    "unzip": None, "7z": None, "gunzip": None,
    "fdisk": None, "mdfind": set(), "python": None, "python3": None, "node": None, "ruby": None, "perl": None,
    "java": None, "javac": None, "rustc": None, "awk": None, "sed": None, "xargs": None,
    "timeout": None, "gtimeout": None, "time": None, "nice": None,
    "nc": None, "curl": None, "mdls": set(), "log": {"erase", "config"}, "dmesg": {"-c", "-C", "--clear"},
    "tee": None, "sort": set(), "echo": set(), "printf": set(), "test": set(),
}
# git: the one tool everyone runs constantly — refined allowlist beats the generic set above.
_GIT_SAFE = {"status", "log", "diff", "show", "branch", "remote", "rev-parse", "ls-files", "ls-tree", "blame",
             "describe", "tag", "grep", "fetch", "stash", "reflog", "shortlog", "cat-file", "rev-list", "count-objects",
             "fsck", "var", "help", "version", "--version", "ls-remote", "check-ignore", "name-rev", "whatchanged",
             "config", "worktree", "symbolic-ref", "merge-base", "cherry", "range-diff", "difftool", "show-ref", "for-each-ref"}
_GIT_SAFE_ONLY_IF_READ = {
    "branch": re.compile(r"\s-[a-zA-Z]*[dDmMcCuU]\b|--(delete|move|copy|set-upstream-to|unset-upstream|edit-description)"),
    "remote": re.compile(r"\b(add|remove|rm|rename|set-url|set-head|prune|update)\b"),
    "tag": re.compile(r"\s-[a-zA-Z]*[adfms]\b|--(delete|annotate|force|sign)|\btag\s+[^-\s]"),
    "stash": re.compile(r"\b(push|pop|apply|drop|clear|save|branch)\b|\bstash\s*$"),
    "config": re.compile(r"\bconfig\b(?!.*\s(--get\S*|-l|--list|--show-origin|--show-scope|--get-regexp))"),
    "worktree": re.compile(r"\b(add|remove|move|prune|lock|unlock|repair)\b"),
    "reflog": re.compile(r"\b(expire|delete)\b"),
    "fetch": re.compile(r"--prune-tags|\s-[pP]\b"),
}
_WRITE_FLAGS = {
    "sed": re.compile(r"(^|\s)-[a-zA-Z]*i"),
    "sysctl": re.compile(r"(^|\s)-w\b|\w=\S"),
    "networksetup": re.compile(r"\s-(set|add|remove|create|delete|order|detect|enable|disable)\w*"),
    "scutil": re.compile(r"\s--set\b|\s-w\b"),
    "mount": re.compile(r"^mount\s+\S"),  # bare `mount` lists; with args it mounts
    "curl": re.compile(r"(^|\s)-[a-zA-Z]*[XdFTouO]|--(data|form|upload|output|remote-name|request|user)\b"),
    "nc": re.compile(r"(^|\s)-[a-zA-Z]*l"),  # listening is fine-ish but ask
    "openssl": re.compile(r"\b(req|genrsa|genpkey|ecparam|pkcs12|ca|enc|rand\s+-out)\b"),
    "ssh-keygen": re.compile(r"(^|\s)-[a-zA-Z]*[tRfA]|--"),
    "gpg": re.compile(r"--(import|delete|gen|edit|sign|encrypt|decrypt)|-[a-zA-Z]*[esc]\b"),
    "tar": re.compile(r"(^|\s)-?[a-zA-Z]*[xc]"),
    "awk": re.compile(r"\bsystem\s*\(|>\s*\""),
    "xargs": re.compile(r"\b(rm|mv|cp|chmod|chown|kill|sed\s+-i|tee|dd|install)\b"),
    "python": re.compile(r"\s-c\s|^\S+\s+\S+\.py"), "python3": re.compile(r"\s-c\s|^\S+\s+\S+\.py"),
    "node": re.compile(r"\s-e\s|\s-p\s|^\S+\s+\S+\.m?js"), "ruby": re.compile(r"\s-e\s|\.rb\b"),
    "perl": re.compile(r"\s-[a-zA-Z]*e\s|\.pl\b"), "java": re.compile(r"^\S+\s+(?!-version|--version)\S"),
    "javac": re.compile(r"."), "rustc": re.compile(r"."),
    "timeout": re.compile(r"."), "gtimeout": re.compile(r"."), "time": re.compile(r"."), "nice": re.compile(r"."),
    "tee": re.compile(r"."), "sqlite3": re.compile(r"\b(insert|update|delete|drop|create|alter|replace|vacuum|attach)\b", re.I),
    "osascript": re.compile(r"."), "open": re.compile(r"."), "say": re.compile(r"."), "caffeinate": re.compile(r"."),
    "unzip": re.compile(r"^unzip\s+(?!.*\s-[lLtz]\b)"), "7z": re.compile(r"^7z\s+[xea]\b"), "gunzip": re.compile(r"^gunzip\s+(?!.*-[ctl])"),
    "fdisk": re.compile(r"^fdisk\s+(?!-l)"), "log": re.compile(r"\b(erase|config)\b"), "dmesg": re.compile(r"\s-[cC]\b|--clear"),
    "softwareupdate": re.compile(r"\s(-i|--install|-ia|--install-rosetta|-d|--download|--schedule)\b"),
    "mdutil": re.compile(r"\s-[Eia]\b"), "service": re.compile(r"\b(start|stop|restart|reload|enable|disable)\b"),
    "diskutil": re.compile(r"\b(mount|unmount|umount|eject|rename|repair\w*|resize\w*|split\w*|merge\w*|add\w*|erase\w*|apfs\s+(?!list)|cs\s+(?!list)|mountDisk|unmountDisk|enableJournal|disableJournal)\b"),
    "hdiutil": re.compile(r"\b(attach|detach|create|convert|resize|burn|eject|mount|unmount)\b"),
    "tmutil": re.compile(r"\b(startbackup|stopbackup|enable|disable|set\w+|remove\w+|delete\w*|localsnapshot|thin\w+|restore|add\w+)\b", re.I),
}
_REDIRECT_RE = re.compile(r"(?<![<>&\d])>{1,2}(?!&)\s*(?!/dev/null|&\d)")  # > or >> to a file (not /dev/null, not 2>&1)
_PROCESS_SUB_RE = re.compile(r"[<>]\(")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+")
_SPLIT_RE = re.compile(r"\s*(\|\||&&|;|\|)\s*(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)")


def _strip_prefixes(seg: str) -> tuple[str, bool]:
    """Remove env assignments, sudo/doas, command/builtin/exec/time/nohup wrappers. Returns (seg, had_sudo)."""
    sudo = False
    changed = True
    while changed and seg:
        changed = False
        m = _ENV_ASSIGN_RE.match(seg)
        if m:
            seg = seg[m.end():]
            changed = True
            continue
        toks = seg.split(None, 1)
        if not toks:
            break
        head = toks[0]
        rest = toks[1] if len(toks) > 1 else ""
        if head in ("sudo", "doas", "pkexec"):
            sudo = True
            # drop sudo flags like -u user, -E, -H, -n, -S
            while rest.startswith("-"):
                parts = rest.split(None, 1)
                flag = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
                if flag in ("-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U") and rest:
                    parts = rest.split(None, 1)
                    rest = parts[1] if len(parts) > 1 else ""
            seg = rest
            changed = True
        elif head in ("command", "builtin", "exec", "nohup", "noglob", "env", "time", "\\time", "stdbuf", "unbuffer"):
            # `env` may carry assignments which the loop strips next iteration
            r = rest
            if head in ("stdbuf",):
                r = re.sub(r"^(-\S+\s+)+", "", r)
            seg = r
            changed = True
    return seg.strip(), sudo


def _head_and_args(seg: str) -> tuple[str, list[str]]:
    try:
        toks = shlex.split(seg, posix=True)
    except ValueError:
        toks = seg.split()
    if not toks:
        return "", []
    head = toks[0]
    if "/" in head:
        head = head.rsplit("/", 1)[-1]
    return head, toks[1:]


def _classify_segment(seg: str) -> Verdict:
    seg = seg.strip()
    if not seg:
        return Verdict(SAFE)
    if _PROCESS_SUB_RE.search(seg):
        return Verdict(ASK, "process substitution")
    for rx, why in _DANGER_RE:
        if rx.search(seg):
            return Verdict(DANGER, why)
    inner, sudo = _strip_prefixes(seg)
    for rx, why in _BLOCKED_RE:
        if rx.search(inner):
            if why.startswith("ping"):
                break  # rewritten by the executor, not blocked
            return Verdict(BLOCKED, why)
    head, args = _head_and_args(inner)
    if head in _BLOCKED_HEADS:
        if head == "crontab" and ("-l" in args):
            pass
        elif head == "nc" and ("-z" in args or "-zv" in args or "-vz" in args):
            pass
        elif head == "sqlite3" and any(a.lower().startswith(("select", ".tables", ".schema", "pragma")) for a in args):
            pass
        else:
            return Verdict(BLOCKED, f"{head} is interactive")
    if _REDIRECT_RE.search(inner):
        return Verdict(ASK, "writes to a file")
    if head not in _SAFE_HEADS:
        v = Verdict(ASK, f"`{head}` changes state or is unknown")
        return v if not sudo else Verdict(ASK, f"sudo {head}")
    # Safe head: refine by subcommand / flags.
    if head == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        if sub and sub not in _GIT_SAFE:
            return Verdict(ASK, f"git {sub} changes the repo")
        rx = _GIT_SAFE_ONLY_IF_READ.get(sub)
        if rx and rx.search(inner):
            return Verdict(ASK, f"git {sub} with a mutating flag")
        return Verdict(ASK, "sudo git") if sudo else Verdict(SAFE)
    rx = _WRITE_FLAGS.get(head)
    if rx and rx.search(inner):
        return Verdict(ASK, f"{head} with a writing flag/argument")
    subs = _MUTATING_SUBCMDS.get(head, set())
    if subs is None and head not in _WRITE_FLAGS:
        return Verdict(ASK, f"{head} changes state")
    if subs:
        for a in args:
            if a in subs:
                return Verdict(ASK, f"{head} {a}")
            if not a.startswith("-"):
                break  # first positional decides for most tools
        # flags that are mutating (dpkg -i, pacman -S ...)
        for a in args:
            if a.startswith("-") and a in subs:
                return Verdict(ASK, f"{head} {a}")
    if head in ("kill", "pkill", "killall"):
        return Verdict(ASK, "kills a process")
    if sudo:
        return Verdict(ASK, f"sudo {head}")
    return Verdict(SAFE)


def split_segments(command: str) -> list[str]:
    """Split on pipes/&&/;/|| outside quotes. Good enough for classification (not execution)."""
    parts = _SPLIT_RE.split(command)
    return [p for p in parts if p and p not in ("|", "||", "&&", ";")]


def classify(command: str) -> Verdict:
    command = command.strip()
    if not command:
        return Verdict(SAFE)
    # Strip a leading subshell / braces, and trailing &.
    command = command.rstrip("&").strip()
    if command.startswith("(") and command.endswith(")"):
        command = command[1:-1]
    if command.startswith("{") and command.endswith("}"):
        command = command[1:-1]
    # Cross-segment patterns (curl | sh, xargs rm) must see the whole line.
    for rx, why in _DANGER_RE:
        if rx.search(command):
            return Verdict(DANGER, why)
    # Heredocs / newlines: classify each line.
    verdict = Verdict(SAFE)
    for line in command.splitlines():
        for seg in split_segments(line):
            v = _classify_segment(seg)
            if verdict < v:
                verdict = v
    return verdict


def rewrite(command: str) -> str:
    """Small rewrites that turn never-ending commands into finite ones."""
    cmd = command
    cmd = re.sub(r"(^|[;&|]\s*)(sudo\s+)?ping6?\s+(?!(?:[^;&|]*?\s)?-c\s*\d)", lambda m: m.group(0) + "-c 4 ", cmd)
    return cmd


def needs_approval(level: str, mode: str) -> bool:
    if level == BLOCKED:
        return False  # never runs
    if mode == "yolo":
        return False
    if mode == "safe":
        return level == DANGER
    return level in (ASK, DANGER)
