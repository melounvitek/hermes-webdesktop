"""Legacy disk caching must never become a cached relay policy decision."""
import builtins
import json
import logging
import os
from pathlib import Path

import pytest

from gateway.config_loader import load_legacy_gateway_json
from gateway.relay import relay_explicitly_disabled
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_legacy_reads_are_reused_without_stale_policy(tmp_path, monkeypatch, caplog):
    home = tmp_path / "primary"
    other = tmp_path / "secondary"
    managed = tmp_path / "managed"
    for directory in (home, other, managed):
        directory.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    legacy = home / "gateway.json"
    disabled = json.dumps({"platforms": {"relay": {"enabled": False}}})
    enabled = json.dumps({"platforms": {"relay": {"enabled": True}}})
    actual_open = builtins.open
    opens = []

    def count_open(file, *args, **kwargs):
        if Path(file).name == "gateway.json":
            opens.append(Path(file))
        return actual_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", count_open)
    caplog.set_level(logging.INFO, logger="gateway.config")
    assert not relay_explicitly_disabled()  # An absent file must not be cached.
    legacy.write_text(disabled, encoding="utf-8")
    for _ in range(10):
        assert relay_explicitly_disabled()
    assert opens == [legacy]
    assert len([r for r in caplog.records if "Loaded legacy" in r.message]) == 1

    # Both the marker-bearing parse and nested data belong to each caller.
    loaded = load_legacy_gateway_json(home)
    loaded["platforms"]["relay"]["enabled"] = True
    loaded["platforms"]["relay"]["extra"]["_enabled_explicit"] = False
    assert relay_explicitly_disabled()
    assert load_legacy_gateway_json(home)["platforms"]["relay"]["extra"]["_enabled_explicit"]

    (other / "gateway.json").write_text(enabled, encoding="utf-8")
    token = set_hermes_home_override(other)
    try:
        assert not relay_explicitly_disabled()
    finally:
        reset_hermes_home_override(token)
    assert relay_explicitly_disabled()
    assert opens == [legacy, other / "gateway.json"]

    # User and managed policy are recomposed even on a legacy cache hit.
    user = home / "config.yaml"
    user.write_text("platforms:\n  relay:\n    enabled: true\n", encoding="utf-8")
    assert not relay_explicitly_disabled()
    overlay = managed / "config.yaml"
    overlay.write_text("platforms:\n  relay:\n    enabled: ${CACHE_TEST_RELAY}\n", encoding="utf-8")
    monkeypatch.setenv("CACHE_TEST_RELAY", "false")
    assert relay_explicitly_disabled()
    monkeypatch.setenv("CACHE_TEST_RELAY", "true")
    assert not relay_explicitly_disabled()
    overlay.write_text("platforms:\n  relay:\n    enabled: false\n", encoding="utf-8")
    assert relay_explicitly_disabled()
    overlay.unlink()
    assert not relay_explicitly_disabled()
    user.unlink()
    assert relay_explicitly_disabled()
    assert opens == [legacy, other / "gateway.json"]

    legacy.write_text(enabled, encoding="utf-8")
    assert not relay_explicitly_disabled()
    # Atomic replacement must invalidate even with equal mtime and byte length.
    replacement = home / "replacement.json"
    replacement.write_text(disabled, encoding="utf-8")
    replacement.replace(legacy)
    assert relay_explicitly_disabled()
    stat = legacy.stat()
    replacement.write_text(enabled + " ", encoding="utf-8")
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert replacement.stat().st_size == stat.st_size
    replacement.replace(legacy)
    assert not relay_explicitly_disabled()
    legacy.unlink()
    assert not relay_explicitly_disabled()
    legacy.write_text(disabled, encoding="utf-8")
    assert relay_explicitly_disabled()


@pytest.mark.parametrize("invalid", ["malformed", "unreadable"])
def test_failed_legacy_reads_warn_and_recover(tmp_path, monkeypatch, caplog, invalid):
    legacy = tmp_path / "gateway.json"
    valid = json.dumps({"platforms": {"relay": {"enabled": False}}})
    legacy.write_text(valid, encoding="utf-8")
    assert load_legacy_gateway_json(tmp_path)["platforms"]["relay"]["enabled"] is False
    caplog.set_level(logging.WARNING, logger="gateway.config")
    actual_open = builtins.open
    unreadable = invalid == "unreadable"
    attempts = []

    def read(file, *args, **kwargs):
        if Path(file) == legacy:
            attempts.append(file)
            if unreadable:
                raise PermissionError("synthetic unreadable legacy file")
        return actual_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read)
    legacy.write_text("{" if invalid == "malformed" else valid + " ", encoding="utf-8")
    for _ in range(2):
        assert load_legacy_gateway_json(tmp_path) == {}
    assert len(attempts) == 2
    assert len([r for r in caplog.records if "Failed to load" in r.message]) == 2
    unreadable = False
    # A different recovered value proves we did not resurrect the old parse.
    legacy.write_text(json.dumps({"platforms": {"relay": {"enabled": True}}}), encoding="utf-8")
    assert load_legacy_gateway_json(tmp_path)["platforms"]["relay"]["enabled"] is True
    assert load_legacy_gateway_json(tmp_path)["platforms"]["relay"]["enabled"] is True
    assert len(attempts) == 3
