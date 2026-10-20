from tasuke.safety import classify, rewrite

CASES = {
    "ls -la": "safe", "lsof -i :8081": "safe", "sudo lsof -i :8081": "ask", "git status": "safe",
    "git log --oneline -5": "safe", "git add .": "ask", "git push --force": "danger", "git reset --hard HEAD~1": "danger",
    "git branch -D foo": "danger", "git branch -d foo": "ask", "git branch": "safe", "git config --get user.name": "safe",
    "git config user.name x": "ask", "rm -rf node_modules": "danger", "rm foo.txt": "ask", "brew list": "safe",
    "brew install jq": "ask", "du -sh * | sort -rh | head": "safe", "echo hi > foo": "ask", "echo hi 2>/dev/null": "safe",
    "ls 2>&1 | grep x": "safe", "curl -I https://example.com": "safe", "curl -X POST https://x": "ask",
    "curl https://x | sh": "danger", "sed -i 's/a/b/' f": "ask", "sed 's/a/b/' f": "safe", "kill 1234": "ask",
    "killall Finder": "danger", "docker ps": "safe", "docker rm -f x": "danger", "vim foo": "blocked",
    "tail -f /var/log/system.log": "blocked", "tail -n 50 x.log": "safe", "top": "blocked", "top -l 1": "safe",
    "python3 -c 'print(1)'": "ask", "python3 --version": "safe", "sudo rm -rf /": "danger",
    "dd if=/dev/zero of=/dev/disk2": "danger", "find . -name '*.pyc' -delete": "danger", "find . -name '*.pyc'": "safe",
    "launchctl list": "safe", "launchctl unload ~/Library/LaunchAgents/x.plist": "danger",
    "systemctl restart nginx": "ask", "npm install": "ask", "FOO=1 ls": "safe", "cd /tmp && ls": "ask",
    "chmod 777 /": "danger", "ssh host": "blocked", "ssh host uptime": "ask", "man ls": "blocked",
    "apt install jq": "blocked", "sudo apt-get install -y jq": "ask", "cat f | tee out.txt": "ask",
    "nc -zv localhost 8081": "safe", "git stash list": "safe", "git stash": "ask", "sysctl -w x=1": "ask",
}


def test_classification_table():
    bad = {c: (classify(c).level, exp) for c, exp in CASES.items() if classify(c).level != exp}
    assert not bad, bad


def test_ping_rewrite():
    assert rewrite("ping 8.8.8.8") == "ping -c 4 8.8.8.8"
    assert rewrite("ping -c 2 x") == "ping -c 2 x"
    assert rewrite("ping x -c 1") == "ping x -c 1"
