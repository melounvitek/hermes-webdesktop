"""Bounded, certificate-verified downloads from the packaged issuer only."""

from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urljoin
from urllib.request import (
    build_opener,
    HTTPSHandler,
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
)

from browser_setup import E


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Download redirects are refused; use the packaged issuer URL")


def source_url(value):
    parts = urlsplit(value)
    E.require(
        parts.scheme == "https"
        and parts.hostname
        and not parts.username
        and not parts.password
        and not parts.query
        and not parts.fragment
        and parts.path.endswith("/CURRENT.json"),
        "Source must be a fixed HTTPS CURRENT.json URL",
    )
    return value


def fetch(url, limit, expected_size=None):
    E.require(urlsplit(url).scheme == "https", "HTTPS is required")
    opener = build_opener(ProxyHandler({}), HTTPSHandler(), NoRedirects())
    deadline = time.monotonic() + 120
    with opener.open(
        Request(url, headers={"Accept-Encoding": "identity"}), timeout=15
    ) as response:
        E.require(response.status == 200, "Expected complete HTTP 200 download")
        E.require(
            response.headers.get("Content-Encoding", "identity") == "identity",
            "Encoded downloads refused",
        )
        length = response.headers.get("Content-Length")
        if length is not None:
            E.require(
                length.isdecimal() and int(length) <= limit,
                "Download exceeds size limit",
            )
            if expected_size is not None:
                E.require(
                    int(length) == expected_size,
                    "Download length differs from descriptor",
                )
        result = bytearray()
        while True:
            E.require(time.monotonic() < deadline, "Download deadline exceeded")
            # read1 returns after one socket read, so a slow trickle cannot keep
            # one large read alive past every deadline check.
            block = response.read1(min(65536, limit + 1 - len(result)))
            if not block:
                break
            result.extend(block)
            E.require(len(result) <= limit, "Download exceeds size limit")
        if length is not None:
            E.require(len(result) == int(length), "Truncated download")
        if expected_size is not None:
            E.require(
                len(result) == expected_size, "Download size differs from descriptor"
            )
        return bytes(result)


def current(source, destination):
    source_url(source)
    descriptor = E.load_json(fetch(source, 16384))
    E.keys(descriptor, "schema archive launcher")
    E.require(
        type(descriptor["schema"]) is int and descriptor["schema"] == 1,
        "Unsupported distribution schema",
    )
    for key, limit in (("archive", E.MAX_ARCHIVE), ("launcher", 1024 * 1024)):
        item = descriptor[key]
        E.keys(item, "name size sha256")
        E.require(
            isinstance(item["name"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", item["name"]),
            "Invalid distribution filename",
        )
        E.hex_value(item["sha256"])
        E.require(
            type(item["size"]) is int and 0 < item["size"] <= limit,
            "Invalid distribution size",
        )
    E.require(
        descriptor["archive"]["name"] != descriptor["launcher"]["name"],
        "Duplicate distribution filename",
    )
    for key in ("archive", "launcher"):
        item = descriptor[key]
        raw = fetch(urljoin(source, item["name"]), item["size"], item["size"])
        E.require(E.digest(raw) == item["sha256"], f"{key} SHA-256 mismatch")
        (destination / key).write_bytes(raw)
    launcher = E.read_regular(destination / "launcher")
    manifest, _ = E.archive_payload(
        str(destination / "archive"), descriptor["archive"]["sha256"], launcher
    )
    return descriptor, manifest


def packaged_source():
    value = E.load_json(E.read_regular(Path(__file__).with_name("source.json"), 16384))
    E.keys(value, "source")
    return source_url(value["source"])
