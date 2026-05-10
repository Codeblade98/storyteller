import json
import urllib.request

from story_engine.llm.client import OpenAICompatibleClient


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def test_openai_compatible_client_parses_response(monkeypatch):
    body = {"choices": [{"message": {"content": "{\"ok\": true, \"text\": \"hello\"}"}}]}
    fake_bytes = json.dumps(body).encode("utf-8")

    def fake_urlopen(request, timeout=60):
        return _FakeResponse(fake_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OpenAICompatibleClient(base_url="http://test", model="m", api_key="k")
    res = client.complete_json("prompt", temperature=0.0)

    assert isinstance(res, str)
    assert res == body["choices"][0]["message"]["content"]


def test_openai_compatible_client_makes_request(monkeypatch):
    body = {"choices": [{"message": {"content": "{\"ok\": true}"}}]}
    fake_bytes = json.dumps(body).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout=60):
        captured["request"] = request
        return _FakeResponse(fake_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OpenAICompatibleClient(base_url="http://example.com/", model="test-model", api_key="secret")
    res = client.complete_json("my prompt", temperature=0.7)

    # verify returned content
    assert res == body["choices"][0]["message"]["content"]

    req = captured.get("request")
    assert req is not None

    # URL should be base_url + /chat/completions (base_url rstrip('/') in client)
    assert req.full_url == "http://example.com/chat/completions"

    # payload should include model, temperature, and the prompt message
    assert hasattr(req, "data")
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.7
    assert payload["messages"][0]["content"] == "my prompt"

    # authorization header should include the API key
    auth = None
    if hasattr(req, "get_header"):
        auth = req.get_header("authorization")
    elif hasattr(req, "headers") and "authorization" in req.headers:
        auth = req.headers["authorization"]

    assert auth == "Bearer secret"
