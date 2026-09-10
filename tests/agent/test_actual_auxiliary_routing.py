"""Actual background tasks exercise the same configured route as foreground chat."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest
import yaml


@pytest.fixture
def actual_endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            content = (
                '{"title":"Actual background routing"}'
                if "response_format" in payload
                else "The task is complete."
            )
            response = {
                "id": "chatcmpl-background",
                "created": 1,
                "model": payload["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            if payload.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                body = f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("aux_provider", ["auto", "actual", "aci", "custom"])
@pytest.mark.parametrize("use_api_key", [False, True])
def test_actual_background_tasks_reach_chat_completions(
    tmp_path, monkeypatch, actual_endpoint, aux_provider, use_api_key
):
    from agent.auxiliary_client import async_call_llm
    from agent.context_compressor import ContextCompressor
    from agent.title_generator import generate_title
    from hermes_cli.runtime_provider import resolve_runtime_provider

    base_url, requests = actual_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if use_api_key:
        monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
        monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:1")
    aux_model = "override-model" if aux_provider == "custom" else "test-model"
    config = {
        "model": {
            "provider": "aci" if aux_provider == "aci" else "actual",
            "default": "test-model",
            "base_url": base_url,
        },
        "auxiliary": {
            task: {
                "provider": aux_provider,
                "model": aux_model,
                **({"base_url": base_url + "/v1"} if aux_provider == "custom" else {}),
            }
            for task in (
                "compression",
                "title_generation",
                "session_search",
            )
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = resolve_runtime_provider(requested="actual")
    runtime["model"] = "test-model"
    assert runtime["api_mode"] == "chat_completions"
    assert (
        generate_title(
            "Stale conversation", main_runtime=runtime, runtime_validator=lambda: False
        )
        is None
    )
    assert requests == []
    assert (
        generate_title("Check the background routing", timeout=5, main_runtime=runtime)
        == "Actual background routing"
    )
    compressor = ContextCompressor(
        model=runtime["model"],
        provider=runtime["provider"],
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        api_mode=runtime["api_mode"],
        config_context_length=32768,
        quiet_mode=True,
    )
    assert (
        compressor._call_summary_llm("Summarize this conversation.", time.monotonic())
        == "The task is complete."
    )
    response = asyncio.run(
        async_call_llm(
            task="session_search",
            messages=[{"role": "user", "content": "Summarize the results."}],
            main_runtime=runtime,
            timeout=5,
        )
    )
    assert response.choices[0].message.content == "The task is complete."
    assert len(requests) == 3
    assert all(
        path == "/v1/chat/completions" and payload["model"] == aux_model
        for path, payload in requests
    )


@pytest.mark.parametrize("override", ["", "http://127.0.0.1:8081", "invalid-url"])
def test_actual_setup_keeps_provider_settings_in_yaml(tmp_path, monkeypatch, override):
    from hermes_cli import config as config_module
    from hermes_cli import model_setup_flows as setup
    from hermes_cli.auth import resolve_api_key_provider_credentials
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from providers import get_provider_profile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8089")
    configured_url = "http://127.0.0.1:8080"
    raw = {
        "model": {
            "provider": "actual",
            "default": "old-model",
            "base_url": configured_url,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("ACTUAL_API_KEY=actual-test-key\n", encoding="utf-8")
    monkeypatch.setattr(
        setup,
        "_ensure_flow_api_key",
        lambda *_a, **_kw: ("actual-test-key", "actual-test-key", False),
    )
    monkeypatch.setattr(setup, "_ask", lambda *_a, **_kw: override)
    monkeypatch.setattr(setup, "_api_key_provider_model_list", lambda *_a: [])
    monkeypatch.setattr(setup, "_pick_model_or_prompt", lambda *_a, **_kw: "test-model")
    setup._model_flow_api_key_provider(config_module.load_config(), "actual")

    expected_url = override if override.startswith("http://") else configured_url
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["provider"] == "actual"
    assert saved["model"]["base_url"] == expected_url
    assert saved["model"]["default"] == "test-model"
    assert env_path.read_text(encoding="utf-8") == "ACTUAL_API_KEY=actual-test-key\n"
    assert get_provider_profile("actual").env_vars == ("ACTUAL_API_KEY",)
    assert "ACTUAL_BASE_URL" not in config_module.OPTIONAL_ENV_VARS
    assert "ACTUAL_API_MODE" not in config_module.OPTIONAL_ENV_VARS
    assert (
        resolve_api_key_provider_credentials("actual")["base_url"]
        == expected_url + "/v1"
    )
    for kwargs in ({}, {"explicit_api_key": "actual-test-key"}):
        runtime = resolve_runtime_provider(requested="actual", **kwargs)
        assert runtime["base_url"] == expected_url + "/v1"
        assert runtime["api_mode"] == "chat_completions"
