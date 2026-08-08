"""
tests/test_groq_retry.py
─────────────────────────
Covers app.services.groq_service._create_chat_completion_with_retry():
retries transient Groq errors (rate limit, timeout, connection, 5xx) with
backoff, but fails immediately on non-transient errors (bad request, auth,
etc). No real network calls or real sleeps — _groq_client and asyncio.sleep
are both mocked.

Context: added alongside a fix for "quiz generation fails most of the time"
— previously a single transient Groq error (e.g. a rate-limit blip) made
generate_questions() fail immediately with zero retries, forcing a fallback
to the (often-empty) DB cache. See app/services/quiz_service.generate_quiz().
"""
import httpx
import pytest
from groq import APIConnectionError, BadRequestError, GroqError, InternalServerError, RateLimitError
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.services import groq_service


def _fake_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status_code, request=request)


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


@pytest.mark.asyncio
async def test_retries_transient_error_then_succeeds():
    """A rate limit on the first attempt should not fail the whole call —
    it should retry and return the eventual successful response."""
    success_response = object()
    mock_create = AsyncMock(
        side_effect=[
            RateLimitError("rate limited", response=_fake_response(429), body=None),
            success_response,
        ]
    )

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.groq_service.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        result = await groq_service._create_chat_completion_with_retry(
            messages=[{"role": "user", "content": "hi"}]
        )

    assert result is success_response
    assert mock_create.call_count == 2
    mock_sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_exhausts_retries_and_raises_last_transient_error():
    """After GROQ_MAX_RETRIES retries are all exhausted, the last transient
    error should propagate (generate_questions() then converts it to a 502)."""
    mock_create = AsyncMock(
        side_effect=InternalServerError("groq is down", response=_fake_response(503), body=None)
    )

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.groq_service.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(InternalServerError):
            await groq_service._create_chat_completion_with_retry(
                messages=[{"role": "user", "content": "hi"}]
            )

    assert mock_create.call_count == settings.GROQ_MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_connection_error_is_retried():
    success_response = object()
    mock_create = AsyncMock(
        side_effect=[
            APIConnectionError(request=_fake_request()),
            success_response,
        ]
    )

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.groq_service.asyncio.sleep", new=AsyncMock()),
    ):
        result = await groq_service._create_chat_completion_with_retry(
            messages=[{"role": "user", "content": "hi"}]
        )

    assert result is success_response
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_non_transient_error_is_not_retried():
    """A bad-request-style error is never going to succeed on retry, so it
    should propagate immediately without consuming any retry attempts."""
    mock_create = AsyncMock(
        side_effect=BadRequestError("bad request", response=_fake_response(400), body=None)
    )

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.groq_service.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        with pytest.raises(BadRequestError):
            await groq_service._create_chat_completion_with_retry(
                messages=[{"role": "user", "content": "hi"}]
            )

    assert mock_create.call_count == 1
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_generate_questions_surfaces_502_after_retries_exhausted():
    """End-to-end: generate_questions() itself should still raise a clean
    HTTPException(502) once retries are exhausted, same contract as before."""
    from fastapi import HTTPException

    mock_create = AsyncMock(
        side_effect=RateLimitError("rate limited", response=_fake_response(429), body=None)
    )

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.groq_service.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await groq_service.generate_questions(
                grade=10, subject="Mathematics", difficulty="easy", question_count=5,
            )

    assert exc_info.value.status_code == 502
    assert mock_create.call_count == settings.GROQ_MAX_RETRIES + 1
