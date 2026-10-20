"""Server-state handling against a tiny fake llama-server: 503 while loading, then 200."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tasuke.backend import BackendError, LLMClient


class _Fake(BaseHTTPRequestHandler):
    loading_left = 3

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path == "/health":
            if _Fake.loading_left > 0:
                _Fake.loading_left -= 1
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"Loading model"}}')
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/props":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"model_path": "/m/qwen-7b.gguf", "default_generation_settings": {"n_ctx": 4096}}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo <tool_call>"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
            "data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.fixture
def fake_server():
    _Fake.loading_left = 3
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_health_transitions_and_wait(fake_server):
    c = LLMClient(fake_server)
    assert c.health() == "loading"
    assert c.wait_until_up(timeout=5)
    assert c.health() == "up"
    assert c.describe() == "qwen-7b" and c.n_ctx(16384) == 4096


def test_stream_parsing(fake_server):
    c = LLMClient(fake_server)
    _Fake.loading_left = 0
    events = list(c.stream_chat([{"role": "user", "content": "hi"}]))
    text = "".join(p for k, p in events if k == "text")
    done = [p for k, p in events if k == "done"][0]
    assert text == "Hello <tool_call>" and done.usage["completion_tokens"] == 2 and done.finish_reason == "stop"


def test_down_server_is_a_clean_error():
    c = LLMClient("http://127.0.0.1:9")  # nothing listens on the discard port
    assert c.health() == "down"
    with pytest.raises(BackendError):
        list(c.stream_chat([{"role": "user", "content": "hi"}]))
