"""
tests/test_auth_bypass.py
───────────────────────────
Confirms the dev AUTH_BYPASS path in app/core/security.py still works,
independent of the get_current_user override used by the other tests
(conftest.py overrides it for convenience — this test calls the real
function directly to make sure AUTH_BYPASS itself wasn't touched/broken).

Relies on the project's .env having ENVIRONMENT=development and
AUTH_BYPASS=true, same as the rest of local development.
"""
from app.core.config import settings
from app.core.security import get_current_user


async def test_auth_bypass_returns_dev_user():
    if not (settings.ENVIRONMENT == "development" and settings.AUTH_BYPASS):
        import pytest
        pytest.skip("AUTH_BYPASS is not enabled in this environment's .env")

    payload = await get_current_user(credentials=None)

    assert payload["sub"] == "dev-user"
    assert payload["dev_auth_bypass"] is True
