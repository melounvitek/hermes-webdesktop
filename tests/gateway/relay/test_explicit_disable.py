"""Explicit relay opt-out must dominate ambient deployment credentials.

Real profile files and production startup/standalone entrypoints; only external
I/O and unrelated gateway services are replaced. No connector is contacted.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import json
import os

import pytest
import yaml

import gateway.relay as relay
from gateway.config import Platform, load_gateway_config
from gateway.platform_registry import platform_registry
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor, CONTRACT_VERSION
from gateway.relay.ws_transport import WebSocketRelayTransport
from gateway.run_startup import GatewayStartupMixin

URL = "wss://connector.example/relay"
CREDENTIALS = {
    "GATEWAY_RELAY_ID": "shared-gateway",
    "GATEWAY_RELAY_INSTANCE_ID": "shared-instance",
    "GATEWAY_RELAY_DELIVERY_KEY": "inherited-delivery-key",
}


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Keep the runtime loader on the same real profile as load_gateway_config.
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    for key in list(os.environ):
        if key.startswith("GATEWAY_RELAY_"):
            monkeypatch.delenv(key)
    for key, value in CREDENTIALS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(GatewayStartupMixin, "_register_config_hooks", lambda *a, **k: None)
    platform_registry.unregister("relay")
    yield tmp_path
    platform_registry.unregister("relay")


def configure(home, state, source, monkeypatch):
    cfg = {"platforms": {"slack": {"enabled": True, "token": "native-test-token"}},
           "slack": {"require_mention": False}}
    if state is not None:
        cfg["platforms"]["relay"] = {"enabled": state, "extra": {"relay_url": URL}}
    if source == "managed-only":
        # No user YAML: the managed overlay must still own relay intent before
        # the injected URL can exclusively suppress this legacy native Slack.
        (home / "gateway.json").write_text(json.dumps({"platforms": {"slack": cfg["platforms"]["slack"]}}))
        managed = home / "managed"
        managed.mkdir()
        managed_cfg: dict = {"gateway": {"relay_url": URL}, "slack": {"require_mention": False}}
        if state is not None:
            managed_cfg["platforms"] = {"relay": {"enabled": state}}
        (managed / "config.yaml").write_text(yaml.safe_dump(managed_cfg))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        monkeypatch.setenv("GATEWAY_RELAY_URL", URL)
        assert not (home / "config.yaml").exists()
        return
    if source == "env":
        monkeypatch.setenv("GATEWAY_RELAY_URL", URL)
    elif source == "managed":
        managed = home / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(yaml.safe_dump({"gateway": {"relay_url": URL}}))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    else:
        cfg["gateway"] = {"relay_url": URL}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def descriptor():
    return CapabilityDescriptor(contract_version=CONTRACT_VERSION, platform="slack",
                                label="Relay", max_message_length=4096,
                                supports_draft_streaming=False, supports_edit=True,
                                supports_threads=False, markdown_dialect="plain", len_unit="chars")


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [False, True, None], ids=["disabled", "enabled", "legacy"])
@pytest.mark.parametrize("source", ["yaml", "env", "managed", "managed-only"])
@pytest.mark.parametrize("pinned", [False, True], ids=["unprovisioned", "inherited-secret"])
async def test_production_startup_respects_profile_opt_out(profile, monkeypatch, state, source, pinned):
    configure(profile, state, source, monkeypatch)
    if state is True and source != "managed-only":
        # The winning top-level enable must override a lower-priority disable.
        path = profile / "config.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg.setdefault("gateway", {})["platforms"] = {"relay": {"enabled": False}}
        path.write_text(yaml.safe_dump(cfg))
    if pinned:
        monkeypatch.setenv("GATEWAY_RELAY_SECRET", "a" * 64)
    before = {k: v for k, v in os.environ.items() if k.startswith("GATEWAY_RELAY_")}
    token = Mock(return_value="identity-token")
    provision = Mock(return_value={"gatewayId": "provisioned-gateway", "secret": "b" * 64,
                                   "deliveryKey": "new-delivery-key"})
    policy = Mock(return_value=200)
    monkeypatch.setattr(relay, "_resolve_relay_identity_token", token)
    monkeypatch.setattr(relay, "_post_provision", provision)
    monkeypatch.setattr(relay, "_post_policy", policy)
    relay_connect = AsyncMock(return_value=True)
    monkeypatch.setattr(WebSocketRelayTransport, "connect", relay_connect)
    monkeypatch.setattr(WebSocketRelayTransport, "handshake", AsyncMock(return_value=descriptor()))
    monkeypatch.setattr(RelayAdapter, "_start_revocation_monitor", lambda self: None)
    native = SimpleNamespace(connect=AsyncMock(return_value=True), send_path_degraded=False,
                             send=AsyncMock(return_value={"success": True}))

    # Execute start(), including real relay bootstrap, prefilter, connect and
    # aggregation. Suppress only unrelated recovery/watchers/status services.
    runner = GatewayStartupMixin()
    runner.config = load_gateway_config()
    runner.adapters = {}
    runner._failed_platforms = {}
    runner.delivery_router = SimpleNamespace(adapters={})
    for name in ("_start_install_faulthandler", "_start_log_startup_environment",
                 "_start_startup_warmup", "_wire_adapter_handlers", "_update_platform_runtime_status",
                 "_wire_teams_pipeline_runtime", "_install_plugin_message_injector",
                 "_update_runtime_status", "_start_spawn_background_watchers"):
        monkeypatch.setattr(runner, name, Mock(), raising=False)
    runner._start_check_access_policy = lambda: False
    runner._abort_startup_if_shutdown_requested = AsyncMock(return_value=False)
    runner._multiplex_on = lambda: False
    runner._startup_should_abort = lambda: False

    async def recover():
        runner._start_register_plugins_relay_hooks()

    async def connect(adapter, platform):
        return await adapter.connect()

    async def secondary(count, skipped):
        return False, count

    runner._start_recover_previous_run = recover
    runner._connect_initial_adapter_with_timeout = connect
    runner._start_secondary_profiles = secondary
    runner._start_finish_wiring = AsyncMock()
    runner._start_handle_no_connections = lambda *args: False
    runner._publish_primary_adapter = lambda p, a: runner.adapters.__setitem__(p, a)
    runner._create_adapter = lambda p, c: (platform_registry.create_adapter("relay", c)
                                         if p == Platform.RELAY else native)
    assert await runner.start() is True

    if state is False:
        token.assert_not_called()
        provision.assert_not_called()
        policy.assert_not_called()
        relay_connect.assert_not_awaited()
        assert not platform_registry.is_registered("relay")
        assert Platform.RELAY not in runner.adapters
        assert Platform.RELAY not in runner._failed_platforms
        assert {k: v for k, v in os.environ.items() if k.startswith("GATEWAY_RELAY_")} == before
        native.connect.assert_awaited_once()
        assert runner.adapters[Platform.SLACK] is native
        from cron.scheduler_delivery import _resolve_target_transport

        resolved, error = _resolve_target_transport(
            {"id": "native-job"}, Platform.SLACK, "slack", {"chat_id": "channel"}, {}, runner.config,
        )
        assert error is None
        assert resolved[1] is runner.config.platforms[Platform.SLACK]
        # Deliver through the native adapter published by real startup, not just
        # a helper predicate. Standalone routing above must also remain enabled.
        resolved, error = _resolve_target_transport(
            {"id": "native-job"}, Platform.SLACK, "slack", {"chat_id": "channel"},
            runner.adapters, runner.config,
        )
        assert error is None
        assert not resolved[0].is_relay
        assert await resolved[0].send(Platform.SLACK, "channel", "native receipt", {}) == {"success": True}
        native.send.assert_awaited_once_with("channel", "native receipt", metadata={})
    else:
        assert platform_registry.is_registered("relay")
        assert provision.call_count == (0 if pinned else 1)
        assert token.call_count == provision.call_count
        policy.assert_called_once()
        # Existing env-exclusive behavior is unchanged. YAML activation is additive.
        assert native.connect.await_count == (0 if source in {"env", "managed-only"} else 1)
        # The env path already auto-enables relay in the platform loader.
        if source in {"env", "managed-only"} or state is True:
            relay_connect.assert_awaited_once()
            assert Platform.RELAY in runner.adapters
        if source in {"env", "managed-only"}:
            from cron.scheduler_delivery import _resolve_target_transport

            resolved, error = _resolve_target_transport(
                {"id": "relay-job"}, Platform.SLACK, "slack", {"chat_id": "channel"}, {}, runner.config,
            )
            assert resolved is None
            assert "relay-fronted" in error


@pytest.mark.parametrize("yaml_state", ["absent", "empty", "sibling", "enabled"])
@pytest.mark.parametrize("disabled", [False, "false"])
def test_legacy_json_disable_is_advisory_like_other_platforms(profile, monkeypatch, yaml_state, disabled):
    (profile / "gateway.json").write_text(json.dumps({"platforms": {
        "relay": {"enabled": disabled},
        "slack": {"enabled": True, "token": "native-test-token"},
        "telegram": {"enabled": False},
    }}))
    cfg = {}
    if yaml_state == "sibling":
        cfg = {"platforms": {"relay": {"extra": {"relay_url": URL}}}}
    elif yaml_state == "enabled":
        cfg = {"platforms": {"relay": {"enabled": True}}}
    if yaml_state != "absent":
        (profile / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("GATEWAY_RELAY_URL", URL)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "native-telegram-test-token")
    monkeypatch.setenv("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS", "true")
    config = load_gateway_config()
    # A legacy gateway.json ``enabled`` is advisory for relay exactly as it is for telegram:
    # env presence (the URL / the token) re-enables both. Only YAML sets ``_enabled_explicit``.
    assert relay.relay_explicitly_disabled() is False
    assert config.platforms[Platform.RELAY].enabled
    assert config.platforms[Platform.SLACK].enabled
    assert config.platforms[Platform.TELEGRAM].enabled


@pytest.mark.parametrize("user_yaml", ["no-files", "absent", "empty", "null", "list", "malformed", "unreadable", "sibling", "enabled", "disabled"])
@pytest.mark.parametrize("managed_state", ["absent", "url-only", "enabled", "disabled"])
def test_managed_layer_agrees_with_native_config(profile, monkeypatch, user_yaml, managed_state):
    """Managed leaves override user leaves, not CLI defaults or a second merge policy."""
    path = profile / "config.yaml"
    user_docs = {
        "empty": "", "null": "null\n", "list": "- not-a-config\n",
        "malformed": "platforms: [\n",
        "sibling": "platforms:\n  relay:\n    extra:\n      user_sibling: kept\n",
        "enabled": "platforms:\n  relay:\n    enabled: true\n",
        "disabled": "platforms:\n  relay:\n    enabled: false\n",
    }
    if user_yaml == "unreadable":
        path.mkdir()  # A directory fails open even when tests run as root.
    elif user_yaml not in {"absent", "no-files"}:
        path.write_text(user_docs[user_yaml])
    managed = profile / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    if managed_state != "absent":
        block: dict = {"extra": {"relay_url": "${GATEWAY_RELAY_URL}", "managed_sibling": "pinned"}}
        if managed_state != "url-only":
            block["enabled"] = managed_state == "enabled"
        (managed / "config.yaml").write_text(yaml.safe_dump({
            "platforms": {"relay": block},
            "gateway": {"max_concurrent_sessions": 7},
        }))
    if user_yaml == "no-files":
        monkeypatch.setenv("SLACK_BOT_TOKEN", "native-test-token")
    else:
        (profile / "gateway.json").write_text(json.dumps({"platforms": {
            "slack": {"enabled": True, "token": "native-test-token"},
            "telegram": {"enabled": False},
        }}))
    monkeypatch.setenv("GATEWAY_RELAY_URL", URL)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "native-telegram-test-token")
    # An absent managed enabled leaf must not introduce DEFAULT_CONFIG's false.
    disabled = managed_state == "disabled" or (
        managed_state in {"absent", "url-only"} and user_yaml == "disabled"
    )
    config = load_gateway_config()
    assert relay.relay_explicitly_disabled() is disabled
    assert config.platforms[Platform.RELAY].enabled is not disabled
    assert config.platforms[Platform.SLACK].enabled is disabled
    assert config.platforms[Platform.TELEGRAM].enabled is disabled
    if managed_state != "absent":
        assert config.max_concurrent_sessions == 7
        assert config.platforms[Platform.RELAY].extra["managed_sibling"] == "pinned"
        assert config.platforms[Platform.RELAY].extra["relay_url"] == URL
    if user_yaml == "sibling":
        assert config.platforms[Platform.RELAY].extra["user_sibling"] == "kept"


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["url", "provision", "policy", "registration", "forced-registration", "fronted"])
@pytest.mark.parametrize("spelling", ["top", "nested", "normalized", "yaml-wins", "gateway-relay", "top-relay", "managed-disable", "sibling-keeps-disable"])
async def test_disabled_standalone_paths_have_no_relay_side_effects(profile, monkeypatch, entry, spelling):
    block = {"enabled": "false" if spelling == "normalized" else False}
    cfg = {"gateway": {"relay_url": URL}}
    if spelling == "nested":
        cfg["gateway"]["platforms"] = {"relay": block}
    elif spelling == "gateway-relay":
        cfg["gateway"]["relay"] = block
        cfg["platforms"] = {"relay": {"enabled": True}}
    elif spelling == "top-relay":
        cfg["relay"] = block
        cfg["gateway"]["relay"] = {"enabled": True}
    elif spelling == "managed-disable":
        cfg["platforms"] = {"relay": {"enabled": True}}
        managed = profile / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(yaml.safe_dump({"platforms": {"relay": block}}))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    elif spelling == "sibling-keeps-disable":
        cfg["gateway"]["platforms"] = {"relay": block}
        cfg["platforms"] = {"relay": {"extra": {"relay_url": URL}}}
    else:
        cfg["platforms"] = {"relay": block}
        if spelling == "yaml-wins":
            cfg["gateway"]["platforms"] = {"relay": {"enabled": True}}
    (profile / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("GATEWAY_RELAY_URL", URL)
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://identity.example/token")
    before = {k: v for k, v in os.environ.items() if k.startswith("GATEWAY_RELAY_")}
    http = Mock(side_effect=AssertionError("disabled relay attempted HTTP"))
    monkeypatch.setattr("urllib.request.urlopen", http)
    provision = Mock(return_value={"secret": "b" * 64, "gatewayId": "replaced"})
    monkeypatch.setattr(relay, "_post_provision", provision)

    if entry == "url":
        assert relay.relay_url() is None
    elif entry == "provision":
        assert relay.self_provision_relay() is False
    elif entry == "policy":
        monkeypatch.setenv("GATEWAY_RELAY_SECRET", "a" * 64)
        before["GATEWAY_RELAY_SECRET"] = "a" * 64
        monkeypatch.setenv("SLACK_ALLOW_BOTS", "all")
        assert relay.send_relay_policy() is False
    elif entry in {"registration", "forced-registration"}:
        assert relay.register_relay_adapter(url=URL, force=entry == "forced-registration") is False
        assert not platform_registry.is_registered("relay")
    elif entry == "fronted":
        assert relay.relay_fronted_platforms() == set()

    http.assert_not_called()
    provision.assert_not_called()
    assert {k: v for k, v in os.environ.items() if k.startswith("GATEWAY_RELAY_")} == before
