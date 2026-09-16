"""Unit tests for the timezone FastAPI dependencies.

The pure precedence rule (resolve_home_timezone) is covered in test_timezone.py.
Here we test the dependency wrappers — specifically the heal SIDE EFFECT: a
stored "UTC" being corrected (backfilled) from a valid browser header, which is
the root-cause fix for workflows/reminders running in UTC.
"""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, status
from fastapi.exceptions import WebSocketException
import pytest

from app.api.v1.dependencies import oauth_dependencies
from app.api.v1.dependencies.oauth_dependencies import (
    get_current_user,
    get_current_user_ws,
    get_user_timezone,
    get_user_timezone_from_preferences,
)
from app.constants.error_codes import NOT_AUTHENTICATED

_DEPS = "app.api.v1.dependencies.oauth_dependencies"
_BACKFILL = f"{_DEPS}._backfill_user_timezone"
_WS_AUTH = f"{_DEPS}.authenticate_workos_session"
_WS_DEV_BYPASS = f"{_DEPS}.resolve_dev_bypass_user"


class TestGetUserTimezoneFromPreferences:
    async def test_real_stored_zone_returned_without_backfill(self) -> None:
        with patch(_BACKFILL, new_callable=AsyncMock) as backfill:
            result = await get_user_timezone_from_preferences(
                user={"user_id": "u1", "timezone": "America/New_York"},
                x_timezone="Asia/Kolkata",
            )
            await asyncio.sleep(0)  # let any fire-and-forget task run
        assert result == "America/New_York"
        backfill.assert_not_awaited()

    async def test_stored_utc_is_healed_and_backfilled_from_header(self) -> None:
        with patch(_BACKFILL, new_callable=AsyncMock) as backfill:
            result = await get_user_timezone_from_preferences(
                user={"user_id": "u1", "timezone": "UTC"},
                x_timezone="Asia/Kolkata",
            )
            await asyncio.sleep(0)
        assert result == "Asia/Kolkata"
        backfill.assert_awaited_once_with("u1", "Asia/Kolkata")

    async def test_empty_stored_is_filled_and_backfilled(self) -> None:
        with patch(_BACKFILL, new_callable=AsyncMock) as backfill:
            result = await get_user_timezone_from_preferences(
                user={"user_id": "u1"},
                x_timezone="Asia/Kolkata",
            )
            await asyncio.sleep(0)
        assert result == "Asia/Kolkata"
        backfill.assert_awaited_once_with("u1", "Asia/Kolkata")

    async def test_genuine_utc_is_not_healed(self) -> None:
        with patch(_BACKFILL, new_callable=AsyncMock) as backfill:
            result = await get_user_timezone_from_preferences(
                user={"user_id": "u1", "timezone": "UTC"},
                x_timezone="UTC",
            )
            await asyncio.sleep(0)
        assert result == "UTC"
        backfill.assert_not_awaited()

    async def test_no_signal_falls_back_to_utc_without_backfill(self) -> None:
        with patch(_BACKFILL, new_callable=AsyncMock) as backfill:
            result = await get_user_timezone_from_preferences(
                user={"user_id": "u1"},
                x_timezone="",
            )
            await asyncio.sleep(0)
        assert result == "UTC"
        backfill.assert_not_awaited()

    async def test_garbage_header_does_not_heal(self) -> None:
        with patch(_BACKFILL, new_callable=AsyncMock) as backfill:
            result = await get_user_timezone_from_preferences(
                user={"user_id": "u1", "timezone": "UTC"},
                x_timezone="Not/A_Zone",
            )
            await asyncio.sleep(0)
        assert result == "UTC"
        backfill.assert_not_awaited()


class TestGetUserTimezoneHeaderDependency:
    def test_returns_canonical_zone_and_now_in_zone(self) -> None:
        tz_str, now = get_user_timezone(x_timezone="Asia/Kolkata")
        assert tz_str == "Asia/Kolkata"
        assert now.utcoffset() == timedelta(hours=5, minutes=30)

    def test_bad_header_falls_back_to_utc_without_raising(self) -> None:
        # The old ZoneInfo(header) raised on garbage; Timezone.parse must not.
        tz_str, now = get_user_timezone(x_timezone="Not/A_Zone")
        assert tz_str == "UTC"
        assert now.utcoffset() == timedelta(0)

    def test_offset_header_is_accepted(self) -> None:
        # ZoneInfo("+05:30") would have raised; the value object handles offsets.
        tz_str, now = get_user_timezone(x_timezone="+05:30")
        assert tz_str == "+05:30"
        assert now.utcoffset() == timedelta(hours=5, minutes=30)


class TestGetCurrentUser:
    async def test_an_unauthenticated_request_is_a_401_carrying_the_code(self) -> None:
        request = MagicMock()
        request.state = MagicMock(spec=[])
        with pytest.raises(HTTPException) as exc:
            await get_current_user(request)
        assert exc.value.status_code == 401
        assert exc.value.detail == {"code": NOT_AUTHENTICATED, "message": "Authentication required"}

    async def test_authenticated_without_user_data_is_a_401_carrying_the_code(self) -> None:
        request = MagicMock()
        request.state.authenticated = True
        request.state.user = None
        with pytest.raises(HTTPException) as exc:
            await get_current_user(request)
        assert exc.value.status_code == 401
        assert exc.value.detail == {"code": NOT_AUTHENTICATED, "message": "User data missing"}


def _websocket(cookies: dict[str, str] | None = None, protocol: str = "") -> MagicMock:
    websocket = MagicMock()
    websocket.cookies = cookies or {}
    websocket.headers = {"sec-websocket-protocol": protocol}
    return websocket


class TestGetCurrentUserWs:
    """The dependency used to close the socket and return ``{}`` typed as an
    AuthenticatedUser, so every caller held a value whose type was a lie."""

    async def test_a_session_cookie_yields_the_authenticated_user(self) -> None:
        with patch(_WS_AUTH, new_callable=AsyncMock, return_value=({"user_id": "u1"}, None)):
            user = await get_current_user_ws(_websocket({"wos_session": "cookie"}))

        assert user == {"user_id": "u1"}

    async def test_a_mobile_client_may_carry_the_token_as_a_subprotocol(self) -> None:
        """Cookies are unavailable there, so the token rides the handshake header."""
        auth = AsyncMock(return_value=({"user_id": "u1"}, None))
        with patch(_WS_AUTH, auth):
            user = await get_current_user_ws(_websocket(protocol="Bearer, mobile.jwt"))

        assert user == {"user_id": "u1"}
        auth.assert_awaited_once_with(session_token="mobile.jwt")

    async def test_no_credential_at_all_is_a_policy_violation(self) -> None:
        with patch(f"{_DEPS}.log") as mock_log:
            with pytest.raises(WebSocketException) as exc:
                await get_current_user_ws(_websocket())

        assert exc.value.code == status.WS_1008_POLICY_VIOLATION
        assert exc.value.reason == "no session cookie or protocol token"
        # The connection's wide event is where a refused dial is counted.
        mock_log.set.assert_called_once_with(disconnect_reason="auth_failure")

    async def test_a_rejected_session_is_a_policy_violation(self) -> None:
        with patch(_WS_AUTH, new_callable=AsyncMock, return_value=(None, None)):
            with pytest.raises(WebSocketException) as exc:
                await get_current_user_ws(_websocket({"wos_session": "stale"}))

        assert exc.value.code == status.WS_1008_POLICY_VIOLATION
        assert exc.value.reason == "session token rejected"

    @pytest.mark.parametrize("user_info", [{"user_id": 12345}, {"email": "a@b.c"}])
    async def test_a_user_without_a_string_id_never_yields_a_socket(self, user_info) -> None:
        """Callers key connection bookkeeping on the id; handing one back
        without it is what made the old empty-dict return look survivable."""
        with (
            patch(_WS_AUTH, new_callable=AsyncMock, return_value=(user_info, None)),
            patch(f"{_DEPS}.log") as mock_log,
        ):
            with pytest.raises(WebSocketException) as exc:
                await get_current_user_ws(_websocket({"wos_session": "cookie"}))

        assert exc.value.reason == "authenticated user has no id"
        assert mock_log.warning.call_args.args[0].endswith(
            "WebSocket user context carries no string user_id"
        )

    async def test_the_dev_bypass_resolves_its_target_without_a_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(oauth_dependencies.settings, "ENV", "development")
        monkeypatch.setattr(oauth_dependencies.settings, "DEV_AUTH_BYPASS_EMAIL", "dev@gaia.local")
        user_doc = MagicMock()
        with (
            patch(
                _WS_DEV_BYPASS, new_callable=AsyncMock, return_value=("dev@gaia.local", user_doc)
            ),
            patch(f"{_DEPS}.user_to_legacy_dict", return_value={"_id": "u1"}) as to_legacy,
            patch(
                f"{_DEPS}.build_user_context", return_value={"user_id": "u1", "dev_bypass": True}
            ) as build_context,
        ):
            user = await get_current_user_ws(_websocket())

        assert user == {"user_id": "u1", "dev_bypass": True}
        to_legacy.assert_called_once_with(user_doc)
        # The flags are the whole point of the bypass context: anything that
        # reads them must be able to tell this session from a real login.
        build_context.assert_called_once_with(
            {"_id": "u1"}, auth_provider="workos", dev_bypass=True
        )

    async def test_a_bypass_target_with_no_user_is_refused_not_impersonated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(oauth_dependencies.settings, "ENV", "development")
        monkeypatch.setattr(oauth_dependencies.settings, "DEV_AUTH_BYPASS_EMAIL", "dev@gaia.local")
        with patch(_WS_DEV_BYPASS, new_callable=AsyncMock, return_value=("ghost@x", None)):
            with pytest.raises(WebSocketException) as exc:
                await get_current_user_ws(_websocket())

        assert exc.value.reason == "dev bypass target has no Mongo user"
