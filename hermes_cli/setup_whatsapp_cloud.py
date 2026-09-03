"""Interactive setup wizard for the WhatsApp Cloud API adapter.

Walks the user through the 6 credentials Meta requires + recipient allowlist, auto-generates the
verify token, and prints exact follow-up instructions for the parts that can't happen inside the
wizard process (cloudflared, gateway, Meta's webhook dashboard, recipient list).

The wizard intentionally does NOT smoke-test the webhook: the gateway and the tunnel both run in
separate processes the user starts AFTER this wizard exits, so any in-wizard probe would fail.
"""

from __future__ import annotations
from hermes_cli.cli_output import line_input

import re
import secrets
import sys
from typing import Optional


# --- Field-shape validators: each returns (ok, reason_if_not_ok) so obviously-malformed input is
# rejected before saving, sparing a round trip with Meta's 401 / 400 errors.

def _validate_phone_number_id(value: str) -> tuple[bool, Optional[str]]:
    """Phone Number ID is a 15-17 digit numeric ID assigned by Meta — NOT a phone number.

    The #1 setup mistake is pasting the actual phone number (10-11 digits), which Graph rejects
    with "Object with ID does not exist."
    """
    if not value:
        return False, "Phone Number ID is required"
    s = value.strip()
    if not s.isdigit():
        return False, "Phone Number ID must be numeric (no '+', spaces, or dashes)"
    if 10 <= len(s) <= 12:  # phone-number-sized: almost certainly the number itself
        return False, (
            "That looks like a phone number — but this field needs the "
            "Phone Number ID (Meta's internal ID, 15-17 digits, e.g. "
            "'7794189252778687'). Look just BELOW the 'From' dropdown in "
            "API Setup → it's labelled 'Phone number ID'.")
    if len(s) < 13:
        return False, "Phone Number ID looks too short (expected 13-18 digits)"
    if len(s) > 20:
        return False, "Phone Number ID looks too long (expected 13-18 digits)"
    return True, None


def _numeric_id_validator(label: str, lo: int, hi: int, expected: str):
    """Validator for a numeric Meta ID whose digit count must fall in [lo, hi]."""
    def validate(value: str) -> tuple[bool, Optional[str]]:
        if not value:
            return False, f"{label} is required"
        s = value.strip()
        if not s.isdigit():
            return False, f"{label} must be numeric"
        if len(s) < lo or len(s) > hi:
            return False, f"{label} looks wrong (expected {expected})"
        return True, None
    return validate


# WABA ID: similar length range as Phone Number ID. App ID: typically 15-16 digits.
_validate_waba_id = _numeric_id_validator("WABA ID", 10, 25, "10-25 digits")
_validate_app_id = _numeric_id_validator("App ID", 13, 20, "15-16 digits")

# Common paste mistakes for the access-token field: (prefixes, what it actually is).
_FOREIGN_TOKEN_PREFIXES = (
    (("sk-",), "That's an OpenAI key (starts with 'sk-'), not a Meta "
               "WhatsApp access token. Meta tokens start with 'EAA'."),
    (("xoxb-", "xoxp-"), "That's a Slack token, not a Meta WhatsApp access token. "
                         "Meta tokens start with 'EAA'."),
    (("ghp_", "gho_"), "That's a GitHub token, not a Meta WhatsApp access "
                       "token. Meta tokens start with 'EAA'."),
)


def _validate_app_secret(value: str) -> tuple[bool, Optional[str]]:
    """App Secret is a 32-character lowercase hex string."""
    if not value:
        return False, "App Secret is required"
    s = value.strip()
    if not re.fullmatch(r"[0-9a-f]+", s.lower()):
        return False, (
            "App Secret should be a hex string (only digits 0-9 and "
            "letters a-f). Make sure you copied the 'App secret' from "
            "Settings → Basic, not some other token.")
    if len(s) != 32:
        return False, f"App Secret should be exactly 32 hex characters (got {len(s)})"
    return True, None


def _validate_access_token(value: str) -> tuple[bool, Optional[str]]:
    """Meta access tokens (temp and System User alike) start with ``EAA``, 100-300+ chars."""
    if not value:
        return False, "Access token is required"
    s = value.strip()
    if not s.startswith("EAA"):
        for prefixes, reason in _FOREIGN_TOKEN_PREFIXES:
            if s.startswith(prefixes):
                return False, reason
        return False, (
            "Meta WhatsApp access tokens start with 'EAA'. Check that "
            "you're copying from the right place (API Setup → 'Generate "
            "access token', or Business Settings → System Users → "
            "'Generate token' for a permanent one).")
    if len(s) < 100:
        return False, f"Access token looks too short ({len(s)} chars, expected 100+)"
    return True, None


# --- Prompt helpers

def _prompt(message: str, default: Optional[str] = None, secret: bool = False) -> str:
    """Read one line of input. Returns "" on EOF / Ctrl+C / empty input.

    ``default`` is shown but NOT auto-applied on empty input: callers handle "kept existing"
    explicitly so a real value is distinguishable from a display preview (masked secrets).
    ``secret=True`` reads via ``getpass`` so credentials are not echoed or left in scrollback.
    """
    try:
        suffix = f" [{default}]" if default else ""
        if secret and sys.stdin.isatty():
            import getpass

            return getpass.getpass(f"{message}{suffix} (input hidden): ").strip()
        return line_input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _prompt_validated(
    message: str, validator, *, current: Optional[str] = None, help_text: Optional[str] = None,
    secret: bool = False) -> Optional[str]:
    """Repeat the prompt until the user enters a valid value or aborts.

    Returns the validated value, or None if the user gave up (empty response after an error, or
    Ctrl+C). ``current`` is shown as a default for re-runs of the wizard with existing config.
    """
    if help_text:
        for line in help_text.strip().splitlines():
            print(f"  {line}")
    attempts = 0
    while True:
        attempts += 1
        value = _prompt(f"  → {message}", default=current, secret=secret)
        if not value:
            return None
        ok, reason = validator(value)
        if ok:
            return value.strip()
        print(f"    ✗ {reason}")
        if attempts >= 3:
            try:
                cont = input("    Try again, or press Enter to skip: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not cont:
                return None
            attempts = 0


# --- Wizard

def _header(title: str) -> None:
    print("─" * 50)
    print(title)
    print("─" * 50)


def _lines(*lines: str) -> None:
    """print() each line; ``""`` yields a blank line."""
    for line in lines:
        print(line)


def _save_optional(key: str, value: Optional[str], current: Optional[str]) -> None:
    from hermes_cli.config import save_env_value
    if value:
        save_env_value(key, value)
        print(f"  ✓ Saved: {value}")
    elif current:
        print(f"  ✓ Keeping existing: {current}")


# Credential steps 1-3: (step title, env var, prompt label, validator, secret, preview chars of
# the existing value shown as default (0 = full), "saved" line, "kept" line, lines printed when
# nothing is configured, abort-when-missing). ``{v}`` in the saved/kept lines is the value.
_CREDENTIAL_STEPS = (
    ("STEP 1 — Phone Number ID", "WHATSAPP_CLOUD_PHONE_NUMBER_ID", "Phone Number ID",
     _validate_phone_number_id, False, 0, "  ✓ Saved: {v}", "  ✓ Keeping existing: {v}",
     ("\n✗ Phone Number ID is required. Aborting.",), True,
     "Found in: App Dashboard → WhatsApp → API Setup, in the\n"
     "'Send and receive messages' section.\n"
     "Look BELOW the 'From' dropdown — there's a 'Phone number ID'\n"
     "line with the value (15-17 digits, e.g. '7794189252778687').\n"
     "It is NOT the phone number itself (+1 555-...). That's the\n"
     "single most common setup mistake."),
    ("STEP 2 — Access Token", "WHATSAPP_CLOUD_ACCESS_TOKEN", "Access Token",
     _validate_access_token, True, 15, "  ✓ Saved (token hidden)", "  ✓ Keeping existing token",
     ("\n✗ Access Token is required. Aborting.",), True,
     "Two options for getting one:\n\n"
     "  (a) TEMP — App Dashboard → WhatsApp → API Setup →\n"
     "      'Generate access token' button. Lasts 24 hours.\n"
     "      Fine for testing today; you'll have to regenerate\n"
     "      tomorrow.\n\n"
     "  (b) PERMANENT (production) — System User token. One-time\n"
     "      setup, never expires:\n"
     "      • business.facebook.com → Settings → System users →\n"
     "        Add → Admin role\n"
     "      • Assign Assets → your app (Manage app), your\n"
     "        WhatsApp account (Manage WABAs)\n"
     "      • Generate token → expiration: Never → permissions:\n"
     "        business_management, whatsapp_business_messaging,\n"
     "        whatsapp_business_management\n\n"
     "Tokens start with 'EAA'."),
    ("STEP 3 — App Secret (required for webhook signature verification)", "WHATSAPP_CLOUD_APP_SECRET",
     "App Secret", _validate_app_secret, True, 8, "  ✓ Saved (secret hidden)",
     "  ✓ Keeping existing App Secret",
     ("\n⚠ Skipping App Secret — inbound webhooks will be refused",
      "   until you set WHATSAPP_CLOUD_APP_SECRET manually."), False,
     "Found in: App Dashboard → Settings → Basic →\n"
     "'App secret' field (click 'Show', enter your Facebook password).\n\n"
     "If 'Show' doesn't appear, you may need Admin role on the app.\n"
     "It's a 32-character lowercase hex string.\n\n"
     "Without the App Secret, inbound webhook POSTs are refused\n"
     "with HTTP 503 (we can't verify they actually came from Meta)."),
)

# Optional step-4 IDs: (prompt label, env var, validator, help text).
_OPTIONAL_ID_STEPS = (
    ("App ID (optional, press Enter to skip)", "WHATSAPP_CLOUD_APP_ID", _validate_app_id,
     "Found in: App Dashboard → Settings → Basic → 'App ID' at the\n"
     "top of the page. Numeric, ~15-16 digits.\n"
     "Not required for messaging — useful only for analytics later."),
    ("WABA ID (optional, press Enter to skip)", "WHATSAPP_CLOUD_WABA_ID", _validate_waba_id,
     "WhatsApp Business Account ID. Found in: App Dashboard →\n"
     "WhatsApp → API Setup, near the top — 'WhatsApp Business\n"
     "Account ID'. Numeric, ~15+ digits.\n"
     "Not required for messaging — useful for analytics."),
)


def _credential_step(step) -> tuple[Optional[str], bool]:
    """Run one _CREDENTIAL_STEPS entry. Returns (effective value, abort)."""
    from hermes_cli.config import get_env_value, save_env_value
    title, env_var, label, validator, secret, preview, saved, kept, missing, required, help_text = step
    _header(title)
    current = get_env_value(env_var) or None
    shown = (current[:preview] + "...") if (preview and current) else current
    value = _prompt_validated(label, validator, current=shown, secret=secret, help_text=help_text)
    if value:
        save_env_value(env_var, value)
        print(saved.format(v=value))
    elif current:
        value = current
        print(kept.format(v=value))
    else:
        _lines(*missing)
        if required:
            return None, True
    print()
    return value, False


def run_whatsapp_cloud_setup() -> int:
    """Interactive wizard for the WhatsApp Cloud API adapter.

    Returns 0 on full success, 1 on user abort, 2 on partial completion (some fields written but the
    user bailed before finishing).
    """
    from hermes_cli.config import get_env_value, save_env_value
    _lines(
        "", "⚕ WhatsApp Business Cloud API Setup", "=" * 50, "",
        "This wizard configures Hermes to talk to WhatsApp via Meta's",
        "official Cloud API. It's the production-grade path:", "",
        "  • No QR codes, no Node.js bridge subprocess",
        "  • Stable connection — no account-ban risk",
        "  • Business account required (not personal WhatsApp)",
        "  • Public webhook URL required (Cloudflare Tunnel, ngrok,",
        "    or your own reverse proxy with TLS)", "",
        "If you don't have a Meta app set up yet, follow these steps",
        "FIRST, then come back and re-run this wizard:", "",
        "  1. https://developers.facebook.com/apps → Create App",
        "     → 'Connect with customers through WhatsApp'",
        "  2. App Dashboard → WhatsApp → API Setup",
        "  3. Click 'Generate access token' (temp 24h token is fine to",
        "     start; switch to a System User permanent token later)", "")
    try:
        input("Press Enter to continue, or Ctrl+C to abort... ")
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return 1

    print()
    for step in _CREDENTIAL_STEPS:
        _, abort = _credential_step(step)
        if abort:
            return 1

    _header("STEP 4 — App ID & WABA ID (optional, for analytics)")
    ids = {}
    for label, env_var, validator, help_text in _OPTIONAL_ID_STEPS:
        current = get_env_value(env_var) or None
        value = _prompt_validated(
            label, lambda v, _val=validator: (True, None) if not v else _val(v),
            current=current, help_text=help_text)
        _save_optional(env_var, value, current)
        ids[env_var] = value or current
    print()

    _header("STEP 5 — Verify Token (auto-generated)")
    verify_token = get_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN") or None
    regen = "y"
    if verify_token:
        print(f"  An existing verify token is already set ({verify_token[:8]}...).")
        try:
            regen = input("  Generate a new one? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            regen = "n"
    if regen in {"y", "yes"}:
        label = "New verify token" if verify_token else "Generated"
        verify_token = secrets.token_urlsafe(32)
        save_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN", verify_token)
        print(f"  ✓ {label}: {verify_token}")
    else:
        print("  ✓ Keeping existing verify token")
    _lines("", "  → COPY THIS TOKEN NOW. You'll paste it into Meta's webhook",
           "    configuration dialog (next step).", "")

    _header("STEP 6 — Recipient Allowlist")
    _lines(
        "", "  Who is allowed to message the bot? (Comma-separated phone",
        "  numbers with country code, no '+' / spaces / dashes. Use '*'",
        "  to allow anyone — only safe if you've also configured Meta's",
        "  recipient whitelist for app-development mode.)", "")
    allow_default = get_env_value("WHATSAPP_CLOUD_ALLOWED_USERS") or None
    try:
        allowed = line_input(
            f"  → Allowed users{' [' + allow_default + ']' if allow_default else ''}: "
        ).strip() or (allow_default or "")
    except (EOFError, KeyboardInterrupt):
        allowed = ""
    if allowed:
        # Light normalization — strip spaces and dashes from each entry.
        allowed = ",".join(re.sub(r"[\s\-+]", "", part) for part in allowed.split(",") if part.strip())
        save_env_value("WHATSAPP_CLOUD_ALLOWED_USERS", allowed)
        print(f"  ✓ Saved: {allowed}")
    else:
        _lines("  ⚠ No allowlist — every inbound message will be denied.",
               "    Re-run this wizard or set WHATSAPP_CLOUD_ALLOWED_USERS manually.")
    print()

    _header("SETUP COMPLETE — Next steps")
    _lines(
        "", "  Hermes needs a public HTTPS URL to receive WhatsApp messages.",
        "  The recommended path is Cloudflare Tunnel (free, no port",
        "  forwarding, no DNS setup).", "",
        "    1. Install cloudflared (one-time, if you don't have it):",
        "         Windows:  winget install Cloudflare.cloudflared",
        "         macOS:    brew install cloudflared",
        "         Linux:    https://github.com/cloudflare/cloudflared/releases", "",
        "       Alternatives: ngrok, or your own domain + reverse proxy",
        "       with TLS.", "",
        "    2. Start the tunnel in a separate terminal:",
        "         cloudflared tunnel --url http://localhost:8090",
        "       Note the printed https://<random>.trycloudflare.com URL.", "",
        "    3. Start the Hermes gateway in another terminal:",
        "         hermes gateway", "",
        "    4. Verify your local config is reachable. From a third",
        "       terminal, with the tunnel URL substituted:", "",
        "         curl 'https://YOUR-TUNNEL.trycloudflare.com/whatsapp/webhook?\\",
        f"               hub.mode=subscribe&hub.verify_token={verify_token}&\\",
        "               hub.challenge=hello'", "",
        "       Expected: HTTP 200 with body 'hello'.",
        "       Also try: curl https://YOUR-TUNNEL.trycloudflare.com/health",
        "       (should return JSON with verify_token_configured: true).", "",
        "    5. Configure Meta to point at your tunnel:",
        "         App Dashboard → WhatsApp → Configuration → Edit webhook",
        "         Callback URL: <tunnel-url>/whatsapp/webhook",
        f"         Verify Token: {verify_token}",
        "         → Click 'Verify and save'",
        "         → Then 'Manage' webhook fields → subscribe to 'messages'", "",
        "    6. Add your phone to Meta's recipient list:",
        "         App Dashboard → WhatsApp → API Setup → 'To' →",
        "         'Manage phone number list'", "",
        "    7. DM the bot's test number from your phone.", "")
    _header("Optional: polish your bot's WhatsApp profile")
    effective_waba = ids["WHATSAPP_CLOUD_WABA_ID"]
    _lines(
        "", "  WhatsApp shows a display name and profile picture for your bot",
        "  in every chat header and contact list. These are set in Meta's",
        "  Business Manager, not via this wizard — but here's where to do",
        "  it once you're up and running:", "",
        "    • Display name + profile picture:",
        "        https://business.facebook.com/wa/manage/phone-numbers/"
        + (f"?waba_id={effective_waba}" if effective_waba else ""))
    if not effective_waba:
        print("        (select your WhatsApp Business Account on that page)")
    _lines(
        "        Display-name changes go through a ~24-48h Meta review.", "",
        "    • About, description, website, hours, business category:",
        "        Same page → click your phone number → 'Edit profile'.", "",
        "    • Verified badge (the green check):",
        "        Requires Meta's business verification process —",
        "        Business Manager → Security Center → Start Verification.", "",
        "  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/",
        "        messaging/whatsapp-cloud", "")
    return 0
