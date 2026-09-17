"""Per-user encrypted browser login persistence.

A browser session's storage_state (Playwright format: {cookies, origins},
covering cookies + localStorage) is Fernet-encrypted and saved per (user_id,
domain) when a session ends, and loaded back to seed the next session on that
domain — so a user doesn't have to log in again on every task.

Encryption follows the same lazy-cipher, Infisical-key pattern as
app/services/mcp/mcp_token_store.py: a Fernet key from
settings.BROWSER_STATE_ENCRYPTION_KEY, a clear error if it's missing or
invalid. storage_state contents (cookies, tokens, localStorage values) are
never logged — only counts and the domain.
"""

import json
from urllib.parse import urlparse

from cryptography.fernet import Fernet
from playwright.sync_api import StorageState, StorageStateCookie

from app.config.settings import settings
from app.constants.browser import BrowserLoginSource
from app.constants.log_tags import LogTag
from app.db.repositories.browser_profiles import browser_profile_repository
from app.models.browser_models import BrowserLoginProvenance
from app.services.browser.storage_state_types import OriginState
from shared.py.wide_events import log

_cipher: Fernet | None = None


def domain_of(url: str | None) -> str | None:
    """Lowercased hostname of a URL, used as the profile key. None if not a URL."""
    if not url:
        return None
    try:
        host = urlparse(url if "://" in url else f"https://{url}").hostname
    except ValueError:
        return None
    return host.lower() if host else None


def _get_cipher() -> Fernet:
    """Get the Fernet cipher for storage_state encryption (lazy init)."""
    global _cipher
    if _cipher is None:
        key: str | None = settings.BROWSER_STATE_ENCRYPTION_KEY
        if not key:
            raise ValueError("BROWSER_STATE_ENCRYPTION_KEY not configured in Infisical")
        try:
            # Fernet expects a URL-safe base64-encoded 32-byte key.
            _cipher = Fernet(key.encode())
        except Exception as e:
            raise ValueError(
                "BROWSER_STATE_ENCRYPTION_KEY is not a valid Fernet key "
                f"(must be 32 url-safe base64-encoded bytes): {e}"
            ) from e
    return _cipher


def _cookie_count(state: StorageState) -> int:
    return len(state.get("cookies", []))


def _origin_count(state: StorageState) -> int:
    return len(state.get("origins", []))


def _encrypt_state(state: StorageState) -> str:
    return _get_cipher().encrypt(json.dumps(state).encode()).decode()


def _decrypt_state(blob: str) -> StorageState:
    decrypted: StorageState = json.loads(_get_cipher().decrypt(blob.encode()).decode())
    return decrypted


async def load_storage_state(user_id: str, domain: str | None) -> StorageState | None:
    """Load and decrypt the saved storage_state for user_id+domain.

    Returns None when there's nothing to seed with (no user, no domain, or
    no saved record) rather than an empty dict, so callers can distinguish
    "seed with this" from "start fresh".
    """
    if not user_id or not domain:
        return None
    record = await browser_profile_repository.get_for_domain(user_id, domain)
    if record is None:
        return None
    state: StorageState = _decrypt_state(record.storage_state_blob)
    log.info(
        f"{LogTag.BROWSER} Loaded saved browser login",
        domain=domain,
        cookie_count=_cookie_count(state),
        origin_count=_origin_count(state),
    )
    return state


async def save_storage_state(
    user_id: str,
    domain: str | None,
    state: StorageState,
    provenance: BrowserLoginProvenance | None = None,
) -> None:
    """Encrypt and persist state for user_id+domain (upsert).

    No-op when there's no user/domain to key on, or when the user has opted
    out of login persistence (settings.BROWSER_PERSIST_LOGINS). provenance
    is recorded only on the import path; the task-end save leaves it None.
    """
    if not user_id or not domain:
        return
    persist_logins: bool = settings.BROWSER_PERSIST_LOGINS
    if not persist_logins:
        return
    blob = _encrypt_state(state)
    await browser_profile_repository.upsert_storage_state_blob(user_id, domain, blob, provenance)
    log.info(
        f"{LogTag.BROWSER} Saved browser login",
        domain=domain,
        cookie_count=_cookie_count(state),
        origin_count=_origin_count(state),
    )


async def forget_browser_logins(user_id: str, domain: str | None = None) -> int:
    """Delete saved logins for user_id, optionally scoped to one domain.

    Returns the number of records deleted. This is the storage-layer primitive
    exercised by the contract test; the settings-UI path
    (profiles.forget_saved_login) delegates here so there is one canonical
    implementation.
    """
    if not user_id:
        return 0
    deleted = await browser_profile_repository.delete_for_user(user_id, domain)
    log.info(f"{LogTag.BROWSER} Forgot browser logins", domain=domain, deleted_count=deleted)
    return deleted


def _cookie_applies_to_host(cookie_domain: str, host: str) -> bool:
    """Return whether cookie_domain covers host, per browser cookie-domain semantics.

    A leading-dot domain (.google.com) applies to that registrable host and
    every subdomain; a host-only domain applies only to the exact host.
    """
    cookie_domain = cookie_domain.lower()
    host = host.lower()
    if cookie_domain.startswith("."):
        suffix = cookie_domain[1:]
        return host == suffix or host.endswith(f".{suffix}")
    return cookie_domain == host


def _cookie_host(cookie: StorageStateCookie) -> str | None:
    """Return the registrable host a cookie is scoped to (leading dot stripped, lowercased), or None if it has none."""
    domain = cookie.get("domain")
    if not domain:
        return None
    return domain.lower().removeprefix(".") or None


def _origin_host(origin: OriginState) -> str | None:
    """Lowercased host of an origin entry, or None when it has no usable URL."""
    return domain_of(origin.get("origin"))


def _cookie_scopes_to(cookie: StorageStateCookie, host: str) -> bool:
    domain = cookie.get("domain")
    return bool(domain) and _cookie_applies_to_host(domain, host)


def split_storage_state_by_host(state: StorageState) -> dict[str, StorageState]:
    """Split one browser export into per-host slices keyed the way reuse loads them.

    Keys are the exact hostname a task starts at (domain_of). A leading-dot
    cookie lands in each host it applies to; a host with no cookies or
    origins is dropped rather than saved empty.
    """
    cookies = state.get("cookies", [])
    origins = state.get("origins", [])

    hosts: set[str] = set()
    for origin in origins:
        if origin_host := _origin_host(origin):
            hosts.add(origin_host)
    for cookie in cookies:
        if cookie_host := _cookie_host(cookie):
            hosts.add(cookie_host)

    slices: dict[str, StorageState] = {}
    for host in hosts:
        host_cookies = [c for c in cookies if _cookie_scopes_to(c, host)]
        host_origins = [o for o in origins if _origin_host(o) == host]
        if host_cookies or host_origins:
            slices[host] = StorageState(cookies=host_cookies, origins=host_origins)
    return slices


async def import_browser_profile(
    user_id: str,
    state: StorageState,
    source_browser: str | None = None,
    source_ip: str | None = None,
) -> list[tuple[str, int]]:
    """Split an uploaded profile per host and persist each slice as a saved login.

    Returns (host, cookie_count) for every host actually stored, so the caller
    can report what landed. Records provenance (source "import" plus the browser
    and client IP) on each per-host doc. Honours the same BROWSER_PERSIST_LOGINS
    opt-out as save_storage_state (each call no-ops when it is off)."""
    provenance = BrowserLoginProvenance(
        source=BrowserLoginSource.IMPORT,
        source_browser=source_browser,
        source_ip=source_ip,
    )
    slices = split_storage_state_by_host(state)
    imported: list[tuple[str, int]] = []
    for host, host_state in slices.items():
        await save_storage_state(user_id, host, host_state, provenance)
        imported.append((host, _cookie_count(host_state)))
    log.info(
        f"{LogTag.BROWSER} Imported browser profile",
        host_count=len(imported),
        cookie_count=_cookie_count(state),
    )
    return imported
