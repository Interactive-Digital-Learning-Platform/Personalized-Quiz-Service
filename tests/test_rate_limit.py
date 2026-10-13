"""
tests/test_rate_limit.py
────────────────────────
The Groq-backed endpoints are rate limited per user, and AI feedback is cached
until the analytics behind it change.
"""
import pytest
from fastapi import HTTPException

from app.core import rate_limit
from app.core.config import settings
from tests.test_quiz_integrity import _answer, _setup, _submit

QUOTE_BODY = {
    "subject": "Mathematics", "difficulty": "easy", "accuracy": 80.0,
    "correct_count": 4, "total_questions": 5, "is_timeout": False,
}


def test_sliding_window_frees_slots_as_old_hits_expire():
    for t in range(3):
        rate_limit._check("b", "u", limit=3, window=60, now=float(t))

    with pytest.raises(HTTPException) as exc:
        rate_limit._check("b", "u", limit=3, window=60, now=30.0)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "30"

    rate_limit._check("b", "u", limit=3, window=60, now=60.0)  # the hit at t=0 has aged out
    rate_limit._check("b", "other-user", limit=3, window=60, now=30.0)  # limits are per user


async def test_motivational_quote_is_rate_limited(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_MOTIVATIONAL_QUOTE", 2)

    async def fake_quote(**_):
        return "Keep going!"

    monkeypatch.setattr("app.api.routes.quiz.generate_motivational_quote", fake_quote)

    codes = [(await client.post("/api/v1/quiz/motivational-quote", json=QUOTE_BODY)).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


async def test_quiz_generate_is_rate_limited_before_any_work(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_QUIZ_GENERATE", 0)

    resp = await client.post("/api/v1/quiz/generate", json={"subject": "Mathematics", "lesson": "Algebra"})

    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


async def test_ai_feedback_is_cached_until_analytics_change(client, db_session, monkeypatch):
    calls = 0

    async def fake_feedback(summary):
        nonlocal calls
        calls += 1
        return {"weak_areas": ["Algebra"], "strong_areas": [], "suggestions": [f"tip {calls}"],
                "motivational_note": "Nice"}

    monkeypatch.setattr("app.api.routes.analytics.generate_feedback", fake_feedback)

    session_id, qids = await _setup(db_session)
    await _submit(client, session_id, [_answer(q) for q in qids])

    first = await client.get("/api/v1/analytics/feedback")
    second = await client.post("/api/v1/analytics/feedback")
    assert first.status_code == second.status_code == 200
    assert second.json()["suggestions"] == ["tip 1"]
    assert calls == 1

    # A new quiz changes the analytics, so the next request regenerates.
    session_id, qids = await _setup(db_session)
    await _submit(client, session_id, [_answer(q, "B") for q in qids])

    third = await client.get("/api/v1/analytics/feedback")
    assert third.json()["suggestions"] == ["tip 2"]
    assert calls == 2


async def test_ai_feedback_fallback_is_not_cached(client, db_session, monkeypatch):
    calls = 0

    async def fallback_feedback(summary):
        nonlocal calls
        calls += 1
        return {"weak_areas": [], "strong_areas": [], "suggestions": ["Keep practising regularly!"],
                "motivational_note": "Great effort!"}

    monkeypatch.setattr("app.api.routes.analytics.generate_feedback", fallback_feedback)
    session_id, qids = await _setup(db_session)
    await _submit(client, session_id, [_answer(q) for q in qids])

    await client.get("/api/v1/analytics/feedback")
    await client.get("/api/v1/analytics/feedback")

    assert calls == 2
