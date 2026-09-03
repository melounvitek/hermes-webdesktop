"""Regex-based secret redaction for logs and tool output.

Masks API keys, tokens, and credentials before they reach log files, verbose
output, or gateway logs. Short tokens (< 18 chars) are fully masked; longer
tokens keep the first 6 and last 4 characters for debuggability.
"""

import logging
import os
import re
import shlex
import threading
from urllib.parse import unquote_plus

# Shared with agent/file_safety's read-block list so the two defenses can't
# drift: a blocked file_tools read that falls back to ``cat`` is still caught.
from agent.file_safety import _BLOCKED_PROJECT_ENV_BASENAMES as _ENV_FILE_BASENAMES

logger = logging.getLogger(__name__)

# Sensitive query-string param names (case-insensitive): opaque tokens / OAuth
# codes / pre-signed signatures with no vendor prefix.
_SENSITIVE_QUERY_PARAMS = frozenset({
    "access_token", "refresh_token", "id_token", "token", "api_key", "apikey",
    "client_secret", "password", "auth", "jwt", "session", "secret", "key",
    "code", "signature", "x-amz-signature",
})

# Snapshot at import time so runtime env mutations (e.g. an LLM-generated
# `export HERMES_REDACT_SECRETS=false`) cannot disable redaction mid-session.
# ON by default; `security.redact_secrets: false` bridges to this env var.
_REDACT_ENABLED = os.getenv("HERMES_REDACT_SECRETS", "true").lower() in {"1", "true", "yes", "on"}

# Known API key prefixes -- match the prefix + contiguous token chars.
# Every pattern MUST start with a literal prefix: _PREFIX_SUBSTRINGS (the cheap
# pre-screen gate) is derived from these literals and must stay false-negative-free.
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",            # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",            # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",            # GitHub refresh token
    r"xapp-\d+-[A-Za-z0-9-]{10,}",      # Slack app-Level token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack bot/app/user tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
    r"fal_[A-Za-z0-9_-]{10,}",          # Fal.ai
    r"fc-[A-Za-z0-9]{10,}",             # Firecrawl
    r"bb_live_[A-Za-z0-9_-]{10,}",      # BrowserBase
    r"gAAAA[A-Za-z0-9_=-]{20,}",        # Codex encrypted tokens
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",        # Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",             # Replicate API token
    r"npm_[A-Za-z0-9]{10,}",            # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI API token
    r"dop_v1_[A-Za-z0-9]{10,}",         # DigitalOcean PAT
    r"doo_v1_[A-Za-z0-9]{10,}",         # DigitalOcean OAuth
    r"am_[A-Za-z0-9_-]{10,}",           # AgentMail API key
    r"sk_[A-Za-z0-9_]{10,}",            # ElevenLabs TTS key (sk_ underscore, not sk- dash)
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily search API key
    r"exa_[A-Za-z0-9]{10,}",            # Exa search API key
    r"gsk_[A-Za-z0-9]{10,}",            # Groq Cloud API key
    r"syt_[A-Za-z0-9]{10,}",            # Matrix access token
    r"retaindb_[A-Za-z0-9]{10,}",       # RetainDB API key
    r"hsk-[A-Za-z0-9]{10,}",            # Hindsight API key
    r"mem0_[A-Za-z0-9]{10,}",           # Mem0 Platform API key
    r"brv_[A-Za-z0-9]{10,}",            # ByteRover API key
    r"xai-[A-Za-z0-9]{30,}",            # xAI (Grok) API key
    r"ntn_[A-Za-z0-9]{10,}",            # Notion internal integration token
    r"fw-[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"fw_[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"fpk_[A-Za-z0-9]{30,}",            # Fireworks AI project key
    # GitLab token families (each keeps a full literal prefix for the pre-screen).
    r"glpat-[A-Za-z0-9_\-]{10,}",       # GitLab personal access token
    r"gloas-[A-Za-z0-9_\-]{10,}",       # GitLab OAuth application secret
    r"gldt-[A-Za-z0-9_\-]{10,}",        # GitLab deploy token
    r"glrt-[A-Za-z0-9_.\-]{10,}",       # GitLab runner authentication token (routable tokens are dotted)
    r"glrtr-[A-Za-z0-9_.\-]{10,}",      # GitLab runner registration token (routable)
    r"glcbt-[A-Za-z0-9_\-]{10,}",       # GitLab CI/CD job token
    r"glptt-[A-Za-z0-9_\-]{10,}",       # GitLab pipeline trigger token
    r"glft-[A-Za-z0-9_\-]{10,}",        # GitLab feed token
    r"glimt-[A-Za-z0-9_\-]{10,}",       # GitLab incoming mail token
    r"glagent-[A-Za-z0-9_\-]{10,}",     # GitLab agent (KAS) token
    r"glsoat-[A-Za-z0-9_\-]{10,}",      # GitLab service-account access token
    r"glffct-[A-Za-z0-9_\-]{10,}",      # GitLab feature-flags client token
    r"glwt-[A-Za-z0-9_\-]{10,}",        # GitLab workspace token
    r"GR1348941[A-Za-z0-9_\-]{10,}",    # GitLab legacy runner registration token
    r"pk-lf-[A-Za-z0-9\-]{8,}",         # Langfuse public key (sk-lf- already covered by sk- pattern)
]

# ENV assignment: KEY=value where KEY carries a secret-like name. Uppercase keys
# tolerate spaces around "=" and allow the keyword embedded anywhere
# (``MYTOKEN=…``) — an all-caps key is almost never prose. Bare ``KEY``/``PASS``/
# ``PW`` suffixes are included; _key_has_secret_keyword rejects ``KEYBOARD=``.
_SECRET_ENV_NAMES = r"(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PW|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2")
# Lowercase env names: only underscore-boundary forms (``openai_key=``) — NOT
# bare ``password=``/``token=``, which appear in prose, URLs, and form bodies.
# The lookbehind anchors each attempt to the start of an identifier run; without
# it re.sub retries the greedy prefix at every byte of a long opaque payload.
_ENV_ASSIGN_LOWER_RE = re.compile(
    rf"(?<![a-z0-9_])([a-z0-9_]+(?:_|^)(?:key|pass|pw|token|secret|password|passwd|credential|auth)(?=[^a-z0-9_]|$))\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)

# Lowercase / dotted config-file keys (``spring.datasource.password=x``,
# line-start ``password=x``). Carve-outs vs prose/code/URLs: values stop at
# whitespace AND ``&`` (form bodies go pair-by-pair via _redact_form_body);
# _CFG_DOTTED_RE needs a NAMESPACED key; _CFG_ANCHORED_RE needs line start
# (optionally after ``export``). The ``://`` URL guard lives at the call site.
_SECRET_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential|auth)"
_CFG_VALUE = r"(['\"]?)([^\s&]+?)\2(?=[\s&]|$)"
# Linear pre-gate for the _CFG_*_RE subs: no secret keyword => neither can match.
_CFG_SECRET_WORD_RE = re.compile(_SECRET_CFG_NAMES, re.IGNORECASE)

# Programmatic env lookups (``os.getenv(...)``, ``process.env.X``, ``$ENV{X}``)
# as the VALUE of a KEY=... match name a variable; they are not a leaked secret.
_ENV_LOOKUP_VALUE_RE = re.compile(r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)")
# Namespaced key: the secret word may sit anywhere in a dotted path.
# NOTE(perf): possessive quantifiers (nested ``(?:[...]+\.)+`` backtracked
# exponentially); the ``*`` runs bordering {_SECRET_CFG_NAMES} must stay
# backtrackable (``app.api.key=``). The lookbehind anchors each attempt to a key
# run start so re.sub is not quadratic; the match set is unchanged.
_CFG_DOTTED_RE = re.compile(
    rf"(?<![A-Za-z0-9_.\-])"
    rf"([A-Za-z0-9_\-]++\.[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*+"
    rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]++)"
    rf"={_CFG_VALUE}",
    re.IGNORECASE,
)
# Line-anchored bare key: ``password=…`` / ``export api_key=…`` at start of line.
_CFG_ANCHORED_RE = re.compile(
    rf"(^[ \t]*(?:export[ \t]+)?[A-Za-z0-9_\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_\-]*)={_CFG_VALUE}",
    re.IGNORECASE | re.MULTILINE,
)

# Unquoted YAML / colon config (``password: secret``): keyword in the KEY
# (anchored to line start) and a single whitespace-free value, so ``note:
# secret meeting`` is left alone. Bare ``auth`` excluded so ``Authorization:``
# (masked by _AUTH_HEADER_RE) / ``author:`` don't match; ``auth_token`` still
# matches via ``token``. Quoted values defer to _JSON_FIELD_RE (lookahead).
# NOTE(perf): possessive where the successor is disjoint; the leading class
# stays backtrackable (see _CFG_DOTTED_RE).
_YAML_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential)"
_YAML_ASSIGN_RE = re.compile(
    rf"(^[ \t]*+[A-Za-z0-9_.\-]*{_YAML_CFG_NAMES}[A-Za-z0-9_.\-]*+)(:[ \t]*+)(?!['\"])([^\s&]++)",
    re.IGNORECASE | re.MULTILINE,
)

# Word-boundary validation for the key patterns above: their classes allow
# arbitrary affixes (``client_secret``, ``s3.secret-key``), which also matched
# prose CONTAINING a keyword (``Secretary:``, ``tokenizer:``). A keyword counts
# only at a word boundary: key edge, next to a non-letter, or a camelCase
# transition (``clientSecret``, ``APIToken``); trailing plural ``s`` is part of
# it. Concatenations match via explicit alternatives (``authtoken``, ``apikey``).
_KEY_KEYWORD_RE = re.compile(
    r"(?:api|auth|access|refresh|session|secret)[ _.\\-]?(?:key|token)"
    r"|token|secret|passwd|password|pass|pw|credential|auth|key",
    re.IGNORECASE,
)

# Key names that are credential-specific even when their values are short or
# human-readable. Bare ``token`` / ``key`` are intentionally absent: they also
# describe model limits, tensor names, and cache keys, so those assignments
# are gated on value shape (_looks_like_opaque_credential).
_STRONG_KEY_KEYWORD_RE = re.compile(
    r"(?:api|auth|access|refresh|session|id|bearer)[ _.\\-]?(?:key|token)"
    r"|key[ _.\\-]?material|secret|passwd|password|pass|pw|credential|auth|bearer",
    re.IGNORECASE,
)


def _is_word_start(s: str, i: int) -> bool:
    """True if position ``i`` in ``s`` begins a word (not mid-word)."""
    if i == 0:
        return True
    prev, cur = s[i - 1], s[i]
    if not prev.isalpha() or (cur.isupper() and prev.islower()):  # camelCase: clientSecret
        return True
    # Acronym run ending (APIToken): 'T' starts a word when followed by lowercase.
    return cur.isupper() and prev.isupper() and i + 1 < len(s) and s[i + 1].islower()


def _is_word_end(s: str, j: int, *, allow_plural: bool = True) -> bool:
    """True if position ``j`` (exclusive end) in ``s`` ends a word."""
    if j >= len(s):
        return True
    cur = s[j]
    if not cur.isalpha() or (cur.isupper() and s[j - 1].islower()):  # camelCase: secretKey
        return True
    if allow_plural and cur in "sS":
        return _is_word_end(s, j + 1, allow_plural=False)
    return False


def _has_word_bounded_keyword(key: str, keyword_re: "re.Pattern[str]") -> bool:
    """True if ``keyword_re`` matches ``key`` at a word boundary (see _KEY_KEYWORD_RE)."""
    return any(
        _is_word_start(key, m.start()) and _is_word_end(key, m.end())
        for m in keyword_re.finditer(key)
    )


def _key_has_secret_keyword(key: str) -> bool:
    """Post-match key validator: ``API_KEY``/``DB_PW`` count, ``KEYBOARD``/``secretary`` do not."""
    return _has_word_bounded_keyword(key, _KEY_KEYWORD_RE)


def _looks_like_opaque_credential(value: str) -> bool:
    """Credential-like shape test for ambiguous ``token``/``key`` values.

    Vendor prefixes and JWTs have dedicated redactors; this catches the remaining
    opaque family without treating short technical scalars (``CPU``, ``local``)
    as secrets merely because their key contains ``token`` or ``key``.
    """
    if value == "***" or value.startswith("«redacted:"):
        return True
    if len(value) >= 16 and re.fullmatch(r"[A-Fa-f0-9]+", value):
        return True
    if len(value) >= 20 and re.fullmatch(r"[A-Za-z0-9_./+=-]+", value):
        return True
    if len(value) < 12:
        return False
    return sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]")) >= 2


def _should_redact_assignment(key: str, value: str, *, check_keyword: bool) -> bool:
    """Shared gate for the ENV / JSON / YAML assignment passes.

    Skips programmatic env lookups used as values (code snippets, not secrets),
    optionally requires a word-bounded secret keyword in the key, then redacts
    when the key is unambiguously credential-bearing or the value looks opaque.
    """
    if _ENV_LOOKUP_VALUE_RE.match(value):
        return False
    if check_keyword and not _key_has_secret_keyword(key):
        return False
    return (_has_word_bounded_keyword(key, _STRONG_KEY_KEYWORD_RE)
            or _looks_like_opaque_credential(value))


# JSON field patterns: "apiKey": "value", "token": "value", etc.
_JSON_KEY_NAMES = r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
_JSON_FIELD_RE = re.compile(rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"', re.IGNORECASE)

# Authorization / Proxy-Authorization, any scheme or bare credential; header
# name and scheme word preserved. The credential class excludes quotes: pulling
# a closing quote into the mask turns value corruption into SYNTAX corruption
# (unterminated quote → shell EOF / SyntaxError).
_AUTH_HEADER_RE = re.compile(r"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?([^\s\"']+)", re.IGNORECASE)

# API-key style headers (single opaque value, no scheme word): non-vendor-prefix
# values would otherwise leak when a curl command is echoed into tool output.
_SECRET_HEADER_NAMES = r"(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)"
_SECRET_HEADER_RE = re.compile(rf"({_SECRET_HEADER_NAMES}\s*:\s*)(\S+)", re.IGNORECASE)

# Telegram bot tokens: bot<digits>:<token> or <digits>:<token>, token >= 30 chars.
_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")

# Private key blocks: -----BEGIN RSA PRIVATE KEY----- ... -----END RSA PRIVATE KEY-----
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----")

# Database connection strings: protocol://user:PASSWORD@host. The userinfo and
# password groups forbid whitespace so a match can never span a line break (a
# greedy ``[^@]+`` once ran to a decorator's ``@`` on the next code line).
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)

# Bare-token URL credential ``scheme://TOKEN@host`` (``git remote set-url
# https://PASSWORD@github.com/...``): unambiguously a secret, since round-trip
# URLs (OAuth callbacks, magic links) carry tokens in the QUERY STRING, never
# bare userinfo. ``user:pass@`` passes through (class forbids ``:``); DB schemes
# belong to _DB_CONNSTR_RE. 8+ char floor skips short usernames; the class
# forbids ``/`` so an ``@`` in a path/query (``?q=user@example.com``) never counts.
_URL_BARE_TOKEN_RE = re.compile(
    r"((?:https?|wss?|git|ssh|ftp|ftps|sftp)://)"  # scheme
    r"([^\s:@/]{8,})"                               # bare token (no colon/slash/@), 8+ chars
    r"(@[^\s]+)",                                   # @host...
    re.IGNORECASE,
)

# JWT tokens: header.payload[.signature] — always start with "eyJ" (base64 "{").
# Matches 1-part (header only), 2-part, and full 3-part JWTs.
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}")

# E.164 phone numbers: +<country><number>, 7-15 digits. The negative lookahead
# prevents matching hex strings or identifiers.
_SIGNAL_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")

# URLs containing query strings — `scheme://authority path ?query [#fragment]` (CDP-URL path).
_URL_WITH_QUERY_RE = re.compile(r"(https?|wss?|ftp)://([^\s/?#]+)([^\s?#]*)\?([^\s#]+)(#\S*)?")

# URLs containing userinfo — `scheme://user:password@host` for ANY web scheme
# (DB protocols are covered by _DB_CONNSTR_RE). CDP-URL path.
_URL_USERINFO_RE = re.compile(r"(https?|wss?|ftp)://([^/\s:@]+):([^/\s@]+)@")

# Strict provider-egress URL redaction: delimiters stay in capture groups so the
# query/fragment layout is preserved byte-for-byte; the key is decoded
# separately for classification. Values stop at ``&``/``;`` (both valid).
_STRICT_URL_PARAM_RE = re.compile(r"([?#&;])([A-Za-z0-9_.~+%\-]+)=([^#&;\s\"'<>]*)")

# Userinfo in absolute and network-path (``//user:pass@host``) references; the
# authority stops at path/query/fragment delimiters. Anchored on the mandatory
# ``//`` — an optional-scheme prefix backtracked O(n²) on long alphanumeric runs
# (~55s per sub() on a 320KB compaction payload).
_STRICT_URL_USERINFO_RE = re.compile(r"(//)([^/\s?#@]+)@")

# Form-urlencoded body detection: conservative — only applies when the entire
# text looks like a query string (k=v&k=v pattern with no newlines).
_FORM_BODY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$")

# Control / zero-width characters that can split a token body (``sk-abc\x1bdef``,
# ``ghp_abc\n123``) and escape the contiguous prefix regexes.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f\u200b-\u200f\u2028-\u202f\u2060\ufeff]")

# Union of every _PREFIX_PATTERNS body class: a control-stripped match may only
# span token-body or control chars. ``=`` is excluded so a KEY=value separator
# never lets a match span unrelated text.
_TOKEN_BODY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.")


def _compile_prefix_matcher(patterns: list) -> "re.Pattern[str]":
    return re.compile(r"(?<![A-Za-z0-9_-])(" + "|".join(patterns) + r")(?![A-Za-z0-9_-])")


_PREFIX_RE = _compile_prefix_matcher(_PREFIX_PATTERNS)


def _mask_control_split_tokens(text: str, mask_fn) -> str:
    """Mask tokens whose body is split by control/zero-width characters.

    Match on a control-stripped copy, then mask the corresponding span in the
    ORIGINAL — only when that span holds solely token-body and control chars, so
    a match can never cross into another line's unrelated text.
    """
    stripped = _CONTROL_CHARS_RE.sub("", text)
    if stripped == text:
        return text
    orig_idx = [i for i, c in enumerate(text) if not _CONTROL_CHARS_RE.match(c)]
    out, matches = list(text), []
    for m in _PREFIX_RE.finditer(stripped):
        start_orig = orig_idx[m.start(1)]
        end_orig = orig_idx[m.end(1) - 1] + 1
        span = text[start_orig:end_orig]
        # Self-matching fragment AND a span crossing a LINE boundary: do NOT join
        # (``ghp_<tok>\nbutton`` would mask ``button``; the prefix pass handles the
        # fragment). Non-newline controls (ESC, ZWSP) never legitimately sit
        # between a token and prose, so there the join proceeds regardless.
        if ("\n" in span or "\r" in span) and _PREFIX_RE.search(span):
            continue
        # Reject spans containing a non-token char (``sk_abc…\nTAVILY_API_KEY=``
        # matched across lines) and matches running into a ``KEY=`` name.
        if (all(c in _TOKEN_BODY_CHARS or _CONTROL_CHARS_RE.match(c) for c in span)
                and (end_orig >= len(text) or text[end_orig] != "=")):
            matches.append((start_orig, end_orig, mask_fn(m.group(1))))
    for start_orig, end_orig, replacement in reversed(matches):
        out[start_orig:end_orig] = list(replacement)
    return "".join(out)


# mask_secret strips EVERY control char (incl. \n/\t, C1, DEL, zero-width) so a
# masked secret never emits multiline or invisible bytes into display output.
_DISPLAY_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f\x80-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064]")


def mask_secret(value: str, *, head: int = 4, tail: int = 4, floor: int = 12,
                placeholder: str = "***", empty: str = "") -> str:
    """Mask a secret for display (``hermes config`` / ``status`` / ``dump``).

    Values shorter than ``floor`` (after control-byte stripping) return
    ``placeholder``; falsy input returns ``empty``.

    >>> mask_secret("sk-proj-abcdef1234567890")
    'sk-p...7890'
    >>> mask_secret("short")
    '***'
    """
    value = _DISPLAY_CONTROL_RE.sub("", value) if value else value
    if not value:
        return empty
    return placeholder if len(value) < floor else f"{value[:head]}...{value[-tail:]}"


def _mask_token(token: str) -> str:
    """Mask a log token — 18-char floor, preserves 6 prefix / 4 suffix; empty → ``***``."""
    if not token:
        return "***"
    return mask_secret(token, head=6, tail=4, floor=18)


def _redact_query_string(query: str) -> str:
    """Replace values of sensitive ``k=v&k=v`` params with ``***``; others pass through."""
    if not query:
        return query
    parts = []
    for pair in query.split("&"):
        key, sep, _value = pair.partition("=")
        parts.append(f"{key}=***" if sep and key.lower() in _SENSITIVE_QUERY_PARAMS else pair)
    return "&".join(parts)


def _canonical_url_param_name(name: str) -> str:
    """Decode a URL parameter name (up to 3 unquote rounds) for case-insensitive matching."""
    decoded = name
    for _ in range(3):
        next_value = unquote_plus(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded.casefold().replace("-", "_")


def _redact_strict_url_credentials(text: str) -> str:
    """Redact credentials from absolute, relative, and network URL references.

    Stricter than display/log redaction; used only at explicit secret-egress
    boundaries. Preserves keys, separators, public params, hosts, and paths.
    """
    def _redact_param(match: re.Match) -> str:
        if _canonical_url_param_name(match.group(2)) not in _SENSITIVE_QUERY_PARAMS:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}=***"

    def _redact_userinfo(match: re.Match) -> str:
        userinfo = match.group(2)
        if ":" in userinfo:
            return f"{match.group(1)}{userinfo.partition(':')[0]}:***@"
        return f"{match.group(1)}***@"

    text = _STRICT_URL_PARAM_RE.sub(_redact_param, text)
    return _STRICT_URL_USERINFO_RE.sub(_redact_userinfo, text)


def redact_cdp_url(value: object) -> str:
    """Mask secrets in a CDP/browser endpoint URL before it is logged.

    ``redact_sensitive_text`` deliberately passes web-URL query params and
    ``user:pass@`` through (OAuth callbacks, magic links the agent must follow).
    CDP discovery tokens are pure credentials, so this opts INTO both URL
    redactors. Single source of truth for CDP URLs passed to a log/error.
    """
    text = redact_sensitive_text("" if value is None else str(value))
    if not text:
        return text
    text = _URL_WITH_QUERY_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}{m.group(3)}?{_redact_query_string(m.group(4))}{m.group(5) or ''}",
        text,
    )
    return _URL_USERINFO_RE.sub(lambda m: f"{m.group(1)}://{m.group(2)}:***@", text)


def _redact_form_body(text: str) -> str:
    """Redact sensitive values when the ENTIRE text is a clean ``k=v&k=v`` body."""
    if not text or "\n" in text or "&" not in text or not _FORM_BODY_RE.match(text.strip()):
        return text
    return _redact_query_string(text.strip())


def _mask_token_nonreusable(token: str) -> str:
    """Redact a prefix-matched credential to a NON-REUSABLE sentinel.

    No head/tail chars (an agent once wrote a truncated-looking mask back into a
    config file, corrupting the credential); only the vendor prefix label is kept
    so the credential KIND stays visible.
    """
    label = next((sub for sub in _PREFIX_SUBSTRINGS if token.startswith(sub)), "") if token else ""
    return f"«redacted:{label}…»" if label else "«redacted-secret»"


def _assignment_sub(render, *, check_keyword: bool):
    """re.sub callback: keep the match unless the key/value pair (groups[0], groups[-1]) needs redaction."""
    def _sub(m):
        groups = m.groups()
        if not _should_redact_assignment(groups[0], groups[-1], check_keyword=check_keyword):
            return m.group(0)
        return render(groups)
    return _sub


def _redact_assignments(text: str) -> str:
    """ENV / config / JSON / YAML assignment passes (skipped for code files).

    Passes that would match ``token=``/``key=`` URL params skip ``://`` text:
    web-URL query params are intentionally passed through (see redact_sensitive_text).
    """
    if "=" in text:
        _redact_env = _assignment_sub(lambda g: f"{g[0]}={g[1]}{_mask_token(g[2])}{g[1]}", check_keyword=True)
        text = _ENV_ASSIGN_RE.sub(_redact_env, text)
        # Lowercase env names; unlike the all-caps regex this one would match URL params.
        if "://" not in text:
            text = _ENV_ASSIGN_LOWER_RE.sub(_redact_env, text)
        # Lowercase/dotted config keys. The keyword pre-gate is exact (every
        # _CFG_*_RE match needs a secret keyword) and matters: _CFG_DOTTED_RE
        # backtracks quadratically on long unbroken [A-Za-z0-9_.\-] runs.
        if "://" not in text and _CFG_SECRET_WORD_RE.search(text):
            text = _CFG_DOTTED_RE.sub(_redact_env, text)
            text = _CFG_ANCHORED_RE.sub(_redact_env, text)

    if ":" in text and '"' in text:
        text = _JSON_FIELD_RE.sub(
            _assignment_sub(lambda g: f'{g[0]}: "{_mask_token(g[1])}"', check_keyword=False), text)

    # Unquoted YAML / colon config, after JSON so quoted values are handled
    # there (_YAML_ASSIGN_RE's lookahead skips quotes).
    if ":" in text and "://" not in text:
        text = _YAML_ASSIGN_RE.sub(
            _assignment_sub(lambda g: f"{g[0]}{g[1]}{_mask_token(g[2])}", check_keyword=True), text)
    return text


def _redact_url_credentials(text: str, code_file: bool) -> str:
    """DB connection-string passwords and bare-token URL userinfo (``://`` text only)."""
    def _redact_db(m):
        # With code_file=True a pure ``{...}`` password group is an f-string
        # template reference (f"postgresql://{user}:{pass}@{host}"), not a
        # literal credential — preserve it. The regex forbids whitespace in the
        # password group, so a single-line template's group(2) is exactly the
        # brace expression.
        pw = m.group(2)
        if code_file and pw.startswith("{") and pw.endswith("}"):
            return m.group(0)
        return f"{m.group(1)}***{m.group(3)}"
    text = _DB_CONNSTR_RE.sub(_redact_db, text)
    # ``scheme://TOKEN@host`` — only the colon-less bare-token form; ``user:pass@``
    # and query-string tokens pass through (see the web-URL note below).
    return _URL_BARE_TOKEN_RE.sub(lambda m: f"{m.group(1)}{_mask_token(m.group(2))}{m.group(3)}", text)


def _redact_phone(m):
    phone = m.group(1)
    keep = 2 if len(phone) <= 8 else 4
    return phone[:keep] + "****" + phone[-keep:]


def redact_sensitive_text(text: str, *, force: bool = False, code_file: bool = False,
                          file_read: bool = False, redact_url_credentials: bool = False) -> str:
    """Apply all redaction patterns to a block of text.

    Safe on any string. Enabled by default (``security.redact_secrets: false``
    disables); ``force=True`` is for safety boundaries that must never return
    raw secrets regardless.

    ``redact_url_credentials=True``: also redact credential-named query params
    and ``user:pass@`` userinfo — off by default because OAuth-callback /
    magic-link / pre-signed URLs must survive ordinary tool flows unchanged.
    ``code_file=True``: skip the ENV/JSON assignment passes for known source
    code (``MAX_TOKENS=***``, ``"apiKey": "test"`` fixtures). ``file_read=True``
    (implies code_file): prefix-matched credentials become a non-reusable
    sentinel (``«redacted:ghp_…»``) instead of a head/tail mask an agent could
    write back into config.yaml as a dead credential.

    Every regex sits behind a cheap substring gate that its pattern requires,
    so the gates are never false-negative.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text or not (force or _REDACT_ENABLED):
        return text
    code_file = code_file or file_read

    # Known prefixes (sk-, ghp_, etc.). Control/zero-width chars can split a
    # token body so _PREFIX_RE alone misses it — mask those runs first.
    if _has_known_prefix_substring(text):
        _prefix_sub = _mask_token_nonreusable if file_read else _mask_token
        text = _mask_control_split_tokens(text, _prefix_sub)
        text = _PREFIX_RE.sub(lambda m: _prefix_sub(m.group(1)), text)

    if not code_file:
        text = _redact_assignments(text)

    # Case-insensitive regex, so "uthorization" is the cheapest substring gate
    # covering every casing without a casefold().
    if "uthorization" in text or "UTHORIZATION" in text:
        text = _AUTH_HEADER_RE.sub(
            lambda m: m.group(1) + (m.group(2) or "") + _mask_token(m.group(3)), text,
        )

    # API-key style headers and Telegram bot tokens both require ":".
    if ":" in text:
        text = _SECRET_HEADER_RE.sub(lambda m: m.group(1) + _mask_token(m.group(2)), text)
        text = _TELEGRAM_RE.sub(lambda m: f"{m.group(1) or ''}{m.group(2)}:***", text)

    if "BEGIN" in text and "-----" in text:
        text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    if "://" in text:
        text = _redact_url_credentials(text, code_file)

    if "eyJ" in text:
        text = _JWT_RE.sub(lambda m: _mask_token(m.group(0)), text)

    # Web-URL query/userinfo redaction is opt-in (see docstring); known
    # credential shapes inside URLs are still caught by the passes above.
    if redact_url_credentials:
        text = _redact_strict_url_credentials(text)

    # Form-urlencoded bodies (only triggers on clean k=v&k=v inputs).
    if "&" in text and "=" in text:
        text = _redact_form_body(text)

    # E.164 phone numbers (Signal, WhatsApp)
    if "+" in text:
        text = _SIGNAL_PHONE_RE.sub(_redact_phone, text)

    return text


# Commands whose stdout is an env-var dump: terminal redaction runs the
# ENV-assignment pass (code_file=False) for these so opaque tokens with no vendor
# prefix are masked; everything else uses code_file=True (``MAX_TOKENS=100``).
_ENV_DUMP_COMMANDS = frozenset({"env", "printenv", "set", "export", "declare"})

# Commands that read file contents to stdout. A ``.env`` target is a credential
# dump (per AGENTS.md ``.env`` holds only secrets), so the ENV pass must run.
_FILE_READ_COMMANDS = frozenset({
    "cat", "head", "tail", "type", "bat", "less", "more", "nl",
    "zcat", "tac", "view", "batcat",
})


def _command_segments(command: str) -> list[str]:
    """Pipeline/sequence segments of a shell command, stripped, empties dropped."""
    return [seg.strip() for seg in re.split(r"[|;&]+", command) if seg.strip()]


def _command_reads_env_file(command: str | None) -> bool:
    """True if ``command`` reads a ``.env``-style file (by basename) to stdout.

    Template files (``.env.example``) are not in the basename list. Defense-in-
    depth, not a boundary: indirect reads (``sudo cat .env``, ``$(cat .env)``,
    ``sed``/``awk``) are not detected, matching ``is_env_dump_command``.
    """
    if not command:
        return False
    for seg in _command_segments(command):
        # Plain split() rather than shlex: shlex treats backslashes as escapes
        # and mangles Windows paths (``C:\Users\...\.env``).
        tokens = seg.split()
        if not tokens or tokens[0] not in _FILE_READ_COMMANDS:
            continue
        for arg in tokens[1:]:
            if arg.startswith("-"):
                continue
            # Strip quotes split() leaves attached, then any / or \ path prefix.
            basename = arg.strip("\"'").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if basename.lower() in _ENV_FILE_BASENAMES:
                return True
    return False


def is_env_dump_command(command: str | None) -> bool:
    """True if ``command`` dumps environment variables to stdout.

    First token of any pipeline/sequence segment in _ENV_DUMP_COMMANDS.
    Conservative: unrecognized → False (callers fall back to code_file=True).
    """
    if not command or not isinstance(command, str):
        return False
    for seg in _command_segments(command):
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if tokens and tokens[0] in _ENV_DUMP_COMMANDS:
            return True
    return False


def redact_terminal_output(output: str, command: str | None = None, *, force: bool = False) -> str:
    """Redact terminal/process stdout — the single policy for ALL terminal-output surfaces.

    ``code_file`` is False (ENV-assignment pass runs) only when ``command`` is an
    env dump or reads a ``.env`` file; otherwise True to avoid false positives
    on source/config dumps. ``force=True`` bypasses the global opt-out.
    """
    if not output:
        return output
    code_file = not (is_env_dump_command(command) or _command_reads_env_file(command))
    return redact_sensitive_text(output, force=force, code_file=code_file)


# --- Prefix pre-screen ------------------------------------------------------
# Derived from _PREFIX_PATTERNS at load time so a new prefix can't silently
# break the gate. No false negatives: every pattern has its literal prefix as a
# substring of any match.

def _extract_literal_prefix(pattern: str) -> str:
    """Leading literal chars of a regex (up to the first metacharacter)."""
    meta = "[(\\.?*+|{^$"
    for i, ch in enumerate(pattern):
        if ch in meta:
            return pattern[:i]
    return pattern


def _skip_char_class(pattern: str, i: int) -> int:
    """Given ``pattern[i] == "["``, return the index just past the closing ``]``."""
    i += 1
    if i < len(pattern) and pattern[i] == "]":
        i += 1
    while i < len(pattern) and pattern[i] != "]":
        if pattern[i] == "\\":
            i += 1
        i += 1
    return i


def _unbounded_quantifier_follows(pattern: str, j: int) -> bool:
    """True if an open-ended quantifier (``*``, ``+``, ``{m,}``) starts at ``pattern[j]``."""
    if j >= len(pattern):
        return False
    if pattern[j] in "*+":
        return True
    if pattern[j] == "{":
        k = pattern.find("}", j)
        body = pattern[j + 1:k] if k != -1 else ""
        return body[:-1].isdigit() and body.endswith(",")  # {m,} is open-ended; {m} / {m,n} bounded
    return False


def _pattern_structure(pattern: str) -> tuple[bool, bool]:
    """One scan → ``(has_top_level_alternation, has_nested_unbounded_repeat)``.

    Top-level ``|`` (outside any group/class) defeats the literal-prefix
    guarantee: for ``ab|.*`` the prefix ``ab`` binds only the first branch;
    grouped alternation (``ab(?:x|y)``) stays allowed. An unbounded quantifier
    applied to a group containing one — ``(a+)+`` / ``(?:x*)*`` / ``(a{2,})+`` —
    is the canonical ReDoS shape, and registered patterns run on every log line.
    Structural nesting only; overlapping branches (``(a|aa)+``) are the plugin
    author's responsibility.
    """
    top_level_alt = nested = False
    contains_unbounded = [False]  # per open group: does it contain an unbounded repeat?
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            i = _skip_char_class(pattern, i)
        elif ch == "(":
            contains_unbounded.append(False)
        elif ch == ")":
            inner = contains_unbounded.pop() if len(contains_unbounded) > 1 else False
            if inner and _unbounded_quantifier_follows(pattern, i + 1):
                nested = True
            contains_unbounded[-1] = contains_unbounded[-1] or inner
        elif ch == "|" and len(contains_unbounded) == 1:
            top_level_alt = True
        elif _unbounded_quantifier_follows(pattern, i):
            contains_unbounded[-1] = True
            if ch == "{":
                i = pattern.find("}", i)  # skip the {m,} body
        i += 1
    return top_level_alt, nested


def _has_top_level_alternation(pattern: str) -> bool:
    return _pattern_structure(pattern)[0]


def _has_nested_unbounded_repeat(pattern: str) -> bool:
    return _pattern_structure(pattern)[1]


_PREFIX_SUBSTRINGS = tuple(_extract_literal_prefix(p) for p in _PREFIX_PATTERNS)


def _has_known_prefix_substring(text: str) -> bool:
    """Cheap pre-check before the expensive ``_PREFIX_RE``."""
    return any(p in text for p in _PREFIX_SUBSTRINGS)


# --- Plugin-registered redaction patterns -----------------------------------
# ADDITIVE-ONLY: a plugin can extend what gets masked but cannot weaken a
# built-in, so it can only over-redact. Keyed by registration source so plugin
# unload has a clean seam to drop ONE plugin's patterns.
_PLUGIN_PREFIX_PATTERNS: dict = {}
_registry_lock = threading.Lock()


def _plugin_patterns() -> list:
    """All plugin-registered patterns in registration order."""
    return [p for patterns in _PLUGIN_PREFIX_PATTERNS.values() for p in patterns]


def _rebuild_prefix_matcher() -> None:
    """Recompile the prefix alternation and pre-screen substrings.

    Callers look these globals up at call time, so swapping the module
    attributes (atomic under the GIL) propagates immediately.
    """
    global _PREFIX_RE, _PREFIX_SUBSTRINGS
    combined = _PREFIX_PATTERNS + _plugin_patterns()
    _PREFIX_RE = _compile_prefix_matcher(combined)
    _PREFIX_SUBSTRINGS = tuple(_extract_literal_prefix(p) for p in combined)


# Structural validators for register_redaction_patterns, in check order:
# (predicate -> reject when True, warning message with (source, pattern) args).
_PATTERN_REJECT_RULES = (
    (_has_top_level_alternation,
     "%s: skipping redaction pattern %r — top-level alternation escapes the literal-prefix "
     "guarantee (in 'ab|.*' the prefix binds only the first branch); wrap alternation in "
     "a group after the prefix, e.g. 'ab(?:x|y)'"),
    (_has_nested_unbounded_repeat,
     "%s: skipping redaction pattern %r — nested unbounded quantifiers (e.g. '(a+)+') can "
     "backtrack catastrophically, and registered patterns run on every log line and tool output"),
    (lambda pattern: len(_extract_literal_prefix(pattern)) < 2,
     "%s: skipping redaction pattern %r — must start with at least 2 literal characters "
     "(needed for the pre-screen substring gate)"),
)


def register_redaction_patterns(patterns, source: str = "plugin") -> int:
    """Additively register credential-token regexes with the redaction engine.

    Accepted patterns join the vendor-prefix alternation everywhere built-ins
    apply. Invalid entries (non-compiling, top-level alternation, nested
    unbounded quantifiers, < 2 literal prefix chars) are warned and skipped,
    never raised — a broken plugin must not break startup. Duplicates are
    skipped. Returns the number of patterns accepted.
    """
    accepted = []
    for pattern in patterns or []:
        if not isinstance(pattern, str) or not pattern.strip():
            logger.warning("%s: skipping empty/non-string redaction pattern", source)
            continue
        pattern = pattern.strip()
        try:
            re.compile(pattern)
        except re.error as exc:
            logger.warning("%s: skipping invalid redaction pattern %r (%s)", source, pattern, exc)
            continue
        rejected = next((message for reject, message in _PATTERN_REJECT_RULES if reject(pattern)), None)
        if rejected:
            logger.warning(rejected, source, pattern)
            continue
        if pattern in _PREFIX_PATTERNS or pattern in _plugin_patterns() or pattern in accepted:
            logger.debug("%s: redaction pattern %r already registered", source, pattern)
            continue
        accepted.append(pattern)

    if accepted:
        with _registry_lock:
            _PLUGIN_PREFIX_PATTERNS.setdefault(source, []).extend(accepted)
            _rebuild_prefix_matcher()
        logger.info("%s: registered %d redaction pattern(s)", source, len(accepted))
    return len(accepted)


def _reset_plugin_redaction_patterns() -> None:
    """Drop all plugin-registered patterns (tests/teardown only)."""
    with _registry_lock:
        _PLUGIN_PREFIX_PATTERNS.clear()
        _rebuild_prefix_matcher()


class RedactingFormatter(logging.Formatter):
    """Log formatter that redacts secrets from all log messages."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(super().format(record))
