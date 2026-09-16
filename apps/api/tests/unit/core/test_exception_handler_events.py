"""What each exception handler puts on the request's wide event.

The response body has its own tests; this is the other half of the contract.
Every field here is queried in Loki when someone asks why a request failed —
``status_code``, ``path`` and ``method`` are what makes one findable at all,
and the 500-vs-4xx split is what ``level="ERROR"`` searches rely on. None of it
was asserted anywhere, so any of it could be renamed or dropped silently.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.exceptions import RequestValidationError
import pytest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from app.core.exception_handlers import (
    app_error_handler,
    http_exception_handler,
    validation_error_handler,
)
from app.utils.errors import AppError, create_error

pytestmark = pytest.mark.unit

HANDLERS_LOG = "app.core.exception_handlers.wide_log"


def _request(method: str = "POST", path: str = "/api/v1/todos") -> Request:
    return Request(
        {"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}
    )


async def test_an_app_error_names_the_failure_and_where_it_happened() -> None:
    exc = create_error(
        message="Payment failed",
        why="Card declined",
        status_code=402,
        code="card_declined",
        public={"retry_allowed": True},
        provider="stripe",
    )

    with patch(HANDLERS_LOG) as mock_log:
        response = await app_error_handler(_request(), exc)

    assert response.status_code == 402
    mock_log.error.assert_called_once_with(
        "app_error",
        error={
            "message": "Payment failed",
            "why": "Card declined",
            "code": "card_declined",
            "retry_allowed": True,
            "provider": "stripe",
        },
        status_code=402,
        path="/api/v1/todos",
        method="POST",
    )


async def test_validation_failures_are_counted_and_listed() -> None:
    exc = RequestValidationError(
        [{"loc": ("body", "count"), "msg": "must be an int", "type": "int_parsing"}]
    )

    with patch(HANDLERS_LOG) as mock_log:
        response = await validation_error_handler(_request(), exc)

    assert response.status_code == 422
    mock_log.warning.assert_called_once_with(
        "validation_failed",
        validation_errors=[
            {"loc": ["body", "count"], "msg": "must be an int", "type": "int_parsing"}
        ],
        error_count=1,
    )


async def test_a_4xx_http_exception_is_a_warning_with_its_detail() -> None:
    with patch(HANDLERS_LOG) as mock_log:
        await http_exception_handler(
            _request(method="GET"), StarletteHTTPException(status_code=404, detail="Todo not found")
        )

    mock_log.error.assert_not_called()
    mock_log.warning.assert_called_once_with(
        "http_exception",
        status_code=404,
        detail="Todo not found",
        path="/api/v1/todos",
        method="GET",
    )


async def test_a_5xx_http_exception_is_an_error_so_level_searches_find_it() -> None:
    with patch(HANDLERS_LOG) as mock_log:
        await http_exception_handler(
            _request(), StarletteHTTPException(status_code=500, detail="boom")
        )

    mock_log.warning.assert_not_called()
    assert mock_log.error.call_args.args[0] == "http_exception"
    assert mock_log.error.call_args.kwargs["status_code"] == 500


async def test_an_explicit_cause_is_named_on_the_event() -> None:
    """``raise HTTPException(...) from e`` is the only way the real failure is
    recorded; without this the event says 500 and nothing about what broke."""
    exc = StarletteHTTPException(status_code=502, detail="upstream")
    exc.__cause__ = ValueError("connection reset")

    with patch(HANDLERS_LOG) as mock_log:
        await http_exception_handler(_request(), exc)

    kwargs: dict[str, Any] = mock_log.error.call_args.kwargs
    assert kwargs["error_type"] == "ValueError"
    assert kwargs["error"] == "connection reset"


async def test_an_incidental_context_is_not_reported_as_the_cause() -> None:
    """``__context__`` is set by any raise inside an except block and is usually
    unrelated; naming it would send investigators after the wrong exception."""
    exc = StarletteHTTPException(status_code=502, detail="upstream")
    exc.__context__ = ValueError("unrelated")

    with patch(HANDLERS_LOG) as mock_log:
        await http_exception_handler(_request(), exc)

    assert "error_type" not in mock_log.error.call_args.kwargs
    assert "error" not in mock_log.error.call_args.kwargs


async def test_a_bodiless_status_keeps_its_headers_and_is_still_recorded() -> None:
    exc = StarletteHTTPException(status_code=304, headers={"ETag": '"v1"'})

    with patch(HANDLERS_LOG) as mock_log:
        response = await http_exception_handler(_request(method="GET"), exc)

    assert response.status_code == 304
    assert response.headers["etag"] == '"v1"'
    assert response.body == b""
    mock_log.warning.assert_called_once()


async def test_the_registration_covers_every_kind_of_failure() -> None:
    """A handler left unregistered silently reverts that path to Starlette's
    own body, which is the shape this whole module exists to replace."""
    from app.core.exception_handlers import register_exception_handlers

    app = MagicMock()
    register_exception_handlers(app)

    registered = [call.args[0] for call in app.exception_handler.call_args_list]
    assert registered == [AppError, RequestValidationError, StarletteHTTPException, Exception]
