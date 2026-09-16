"""API v1 middleware package.

Deliberately empty. The convenience re-exports that used to live here made
`import app.api.v1.middleware.tiered_rate_limiter` execute `.auth` as well,
so every importer of the rate limiter — the `@tiered_rate_limit` decorator,
and through it most of the app and the test suite — paid workos and the whole
middleware stack (131 modules, ~0.2 s). Import each middleware from its own
module instead.

`tiered_rate_limit` was never re-exported here either: it lives in
app/decorators/rate_limiting.py. A second copy in this package drifted and
silently skipped rate limiting.
"""
