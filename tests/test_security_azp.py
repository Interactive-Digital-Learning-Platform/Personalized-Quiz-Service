"""
tests/test_security_azp.py
──────────────────────────
_verify_token's optional `azp` (authorized party) check.
"""
import pytest
from fastapi import HTTPException

from app.core import security
from app.core.config import settings


class _Key:
    key = "k"


class _FakeJWKS:
    def get_signing_key_from_jwt(self, token):
        return _Key()


@pytest.fixture
def decode_returns(monkeypatch):
    monkeypatch.setattr(security, "_jwks_client", _FakeJWKS())

    def set_payload(payload: dict):
        monkeypatch.setattr(security.jwt, "decode", lambda *a, **k: payload)

    return set_payload


async def test_any_azp_is_accepted_when_not_configured(decode_returns, monkeypatch):
    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "")
    decode_returns({"sub": "u", "azp": "https://evil.example"})

    assert (await security._verify_token("t"))["sub"] == "u"


async def test_foreign_azp_is_rejected_when_configured(decode_returns, monkeypatch):
    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "https://app.example, http://localhost:8081")
    decode_returns({"sub": "u", "azp": "https://evil.example"})

    with pytest.raises(HTTPException) as exc:
        await security._verify_token("t")
    assert exc.value.status_code == 401


async def test_listed_or_missing_azp_is_accepted_when_configured(decode_returns, monkeypatch):
    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "https://app.example,http://localhost:8081")

    decode_returns({"sub": "u", "azp": "http://localhost:8081"})
    assert (await security._verify_token("t"))["sub"] == "u"

    decode_returns({"sub": "u"})  # native app tokens may carry no azp
    assert (await security._verify_token("t"))["sub"] == "u"
