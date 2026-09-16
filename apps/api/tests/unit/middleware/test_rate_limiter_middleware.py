"""The default rate limit has to reach routes that live behind a lazy router.

FastAPI 0.139 defers router inclusion: ``app.routes`` holds ``_IncludedRouter``
wrappers with no ``.endpoint``, so slowapi's own handler lookup returned None
for every included route and ``_should_exempt`` then exempted it — the default
limit applied to nothing at all.
"""

from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel
import pytest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1.middleware.rate_limiter import (
    RouterAwareSlowAPIMiddleware,
    find_route_handler,
)
from app.core.middleware import rate_limit_handler

pytestmark = pytest.mark.unit


class _Payload(BaseModel):
    count: int


def _app(limit: str = "1/minute", *, enabled: bool = True) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RouterAwareSlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.state.limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[limit],
        storage_uri="memory://",
        enabled=enabled,
    )

    router = APIRouter(prefix="/included")

    @router.get("/thing")
    async def thing() -> _Payload:
        return _Payload(count=1)

    app.include_router(router)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    )


def test_the_handler_of_an_included_route_is_found() -> None:
    app = _app()
    scope = {"type": "http", "path": "/included/thing", "method": "GET", "headers": []}

    handler = find_route_handler(app, scope)

    assert handler is not None
    assert handler.__name__ == "thing"


def test_an_unmatched_path_has_no_handler() -> None:
    assert (
        find_route_handler(
            _app(), {"type": "http", "path": "/nope", "method": "GET", "headers": []}
        )
        is None
    )


async def test_the_default_limit_applies_to_an_included_route() -> None:
    async with _client(_app()) as client:
        assert (await client.get("/included/thing")).status_code == 200
        second = await client.get("/included/thing")

    assert second.status_code == 429
    assert second.json()["code"] == "rate_limit_exceeded"
    assert int(second.headers["retry-after"]) > 0


async def test_a_disabled_limiter_never_refuses() -> None:
    async with _client(_app(enabled=False)) as client:
        for _ in range(3):
            assert (await client.get("/included/thing")).status_code == 200


async def test_an_unrouted_path_is_exempt_rather_than_counted() -> None:
    """With no handler there is no route to attribute a limit to; slowapi's own
    contract is to skip, and a 404 must not be spent from the caller's budget."""
    async with _client(_app()) as client:
        for _ in range(3):
            assert (await client.get("/nope")).status_code == 404
        assert (await client.get("/included/thing")).status_code == 200
