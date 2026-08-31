"""
tests/test_ai_generation_telemetry.py
────────────────────────────────────────
Tests for internal AI-generation telemetry:
- one AIGenerationEvent row per POST /quiz/generate request (success,
  failure-with-cache-fallback, and total failure)
- telemetry write failures never break the actual quiz-generation response
- the admin-only GET /analytics/system/ai-generation endpoint, including
  its authorization rule and date-range filtering

Groq itself is never called here — app.services.quiz_service.generate_questions
is mocked for deterministic, network-free control over success/failure, the
same way this project's other tests avoid the real Groq/full-submission
pipeline for anything unrelated to what's under test.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, status
from sqlalchemy import select

from app.core.config import settings
from app.models.ai_generation_event import AIGenerationEvent
from app.models.question import Question
from app.models.subject_mastery import SubjectMastery
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

GENERATE_URL = "/api/v1/quiz/generate"
TELEMETRY_URL = "/api/v1/analytics/system/ai-generation"


_DISTINCT_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _fake_question(i: int, lesson: str = "Algebra") -> dict:
    # The dedup logic (_is_near_duplicate) compares significant ALPHA words
    # only — texts differing solely by a digit (e.g. "Question 0?" vs
    # "Question 1?") look identical to it, since digits aren't alpha and get
    # stripped from the comparison. Use a genuinely distinct word per index
    # instead, so a batch of these doesn't get collapsed by dedup.
    word = _DISTINCT_WORDS[i % len(_DISTINCT_WORDS)]
    return {
        "question": f"What is the {word} concept called in this lesson?",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A",
        "explanation": "because",
        "lesson": lesson,
    }


def _generate_questions_mock() -> AsyncMock:
    # A challenge-zone quiz (no explicit difficulty/lesson) calls
    # generate_questions() once per difficulty tier, each asking for a
    # different `question_count` — so the mock must actually honor that
    # kwarg (and hand back distinct questions across calls) rather than
    # always returning one fixed-size batch, or the totals below won't add
    # up to what was requested.
    counter = {"i": 0}

    async def _side_effect(*, grade, subject, difficulty, question_count, **_kwargs):
        batch = []
        for _ in range(question_count):
            batch.append(_fake_question(counter["i"]))
            counter["i"] += 1
        return batch

    return AsyncMock(side_effect=_side_effect)


async def _latest_event(db) -> AIGenerationEvent:
    stmt = select(AIGenerationEvent).order_by(AIGenerationEvent.id.desc()).limit(1)
    return (await db.execute(stmt)).scalar_one()


async def _seed_cached_questions(
    db, subject: str, difficulty: str, lesson: str, count: int, grade: int | None = None
) -> None:
    for i in range(count):
        db.add(Question(
            question=f"Cached question {i}?", options=["A", "B", "C", "D"], correct_answer="A",
            subject=subject, lesson=lesson, difficulty=difficulty, grade=grade,
        ))
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Successful Groq generation
# ─────────────────────────────────────────────────────────────────────────────

async def test_successful_generation_records_telemetry(client, db_session):
    with patch("app.services.quiz_service.generate_questions", new=_generate_questions_mock()):
        resp = await client.post(GENERATE_URL, json={"subject": "Mathematics", "question_count": 5})

    assert resp.status_code == 201
    body = resp.json()
    assert body["cache_hit"] is False

    event = await _latest_event(db_session)
    assert event.subject == "Mathematics"
    assert event.requested_question_count == 5
    assert event.generated_question_count == 5
    assert event.success is True
    assert event.used_cache_fallback is False
    assert event.error_category is None
    assert event.provider == "groq"
    assert event.model_name == settings.GROQ_MODEL
    assert event.latency_ms >= 0.0
    assert event.session_id == body["session_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Failed generation with cache fallback
# ─────────────────────────────────────────────────────────────────────────────

async def test_failed_generation_falls_back_to_cache_and_records_telemetry(client, db_session):
    # Grade 7 has no curriculum data (see app/data/curriculum/) -- deliberate
    # here so this test's generic "Physics" lesson stand-in isn't filtered
    # out by the grade-10/11 curriculum-lesson matching the cache fallback
    # now applies (see test_quiz_generation_pool.py for dedicated
    # curriculum-filtering tests).
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _seed_cached_questions(db_session, "Science", "easy", "Physics", count=5, grade=7)

    with patch(
        "app.services.quiz_service.generate_questions",
        new=AsyncMock(side_effect=HTTPException(status.HTTP_502_BAD_GATEWAY, detail="AI service error: boom")),
    ):
        resp = await client.post(
            GENERATE_URL, json={"grade": 7, "subject": "Science", "difficulty": "easy", "question_count": 5},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["cache_hit"] is True

    event = await _latest_event(db_session)
    assert event.success is False
    assert event.used_cache_fallback is True
    assert event.error_category == "provider_error"
    assert event.generated_question_count == 5


# ─────────────────────────────────────────────────────────────────────────────
# 3. Failed generation WITHOUT a cache fallback
# ─────────────────────────────────────────────────────────────────────────────

async def test_failed_generation_without_fallback_still_records_telemetry(client, db_session):
    # No cached questions exist for this subject/difficulty at all. Grade 7
    # has no curriculum data, so the made-up subject name is still accepted
    # (legacy free-text behavior) rather than rejected by curriculum validation.
    with patch(
        "app.services.quiz_service.generate_questions",
        new=AsyncMock(side_effect=HTTPException(status.HTTP_502_BAD_GATEWAY, detail="AI returned malformed JSON: x")),
    ):
        resp = await client.post(
            GENERATE_URL,
            json={"grade": 7, "subject": "NoCacheSubject", "difficulty": "hard", "question_count": 5},
        )

    assert resp.status_code == 502

    event = await _latest_event(db_session)
    assert event.success is False
    assert event.used_cache_fallback is True
    assert event.error_category == "invalid_json"
    assert event.generated_question_count == 0
    assert event.session_id is None  # no QuizSession was ever created


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Challenge-zone quizzes must respect a shared Groq call budget
# ─────────────────────────────────────────────────────────────────────────────

async def test_challenge_zone_never_exceeds_generation_call_budget(client, db_session):
    # A challenge-zone quiz (no explicit difficulty/lesson) splits into
    # multiple difficulty tiers, and each tier independently runs a
    # dedup-retry loop -- without a shared cap, a mock that keeps returning
    # too few questions (forcing every tier to exhaust its own retries)
    # could balloon into far more Groq calls than one "generate quiz" tap
    # should ever cost. This is a regression test for exactly that: before
    # the shared _CallBudget existed, this scenario made ~12 calls instead
    # of being capped at GROQ_MAX_GENERATION_CALLS_PER_REQUEST.
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Seed mastery in the medium band so the challenge-zone profile spreads
    # question_count=10 across all three tiers (easy/medium/hard all > 0),
    # maximizing the number of independent dedup-retry loops in play.
    db_session.add(SubjectMastery(
        user_id=user.id, subject="Mathematics", difficulty="medium",
        last_accuracy=70.0, consecutive_strong=0, consecutive_weak=0,
        mastery_score=70.0, fluency_score=50.0, confidence_score=50.0, evidence_count=20,
    ))
    # Seed cached questions at every tier so that once the shared budget
    # runs out for a later tier, it can still fall back to cache and the
    # request succeeds overall -- the point of this test is the call count,
    # not whether cache happens to be empty. Grade 7 has no curriculum data
    # (see app/data/curriculum/), deliberate here so the generic "Algebra"
    # lesson stand-in isn't filtered out by the grade-10/11 curriculum-lesson
    # matching the pool/cache fallback now applies.
    await _seed_cached_questions(db_session, "Mathematics", "easy", "Algebra", count=10, grade=7)
    await _seed_cached_questions(db_session, "Mathematics", "medium", "Algebra", count=10, grade=7)
    await _seed_cached_questions(db_session, "Mathematics", "hard", "Algebra", count=10, grade=7)

    call_count = {"n": 0}

    async def _always_one_question(*, grade, subject, difficulty, question_count, **_kwargs):
        # Always returns exactly 1 question, no matter how many were asked
        # for -- guarantees every tier's dedup loop exhausts all 3 attempts
        # (each attempt still short of `question_count`) unless the shared
        # budget cuts it off first.
        call_count["n"] += 1
        return [_fake_question(call_count["n"])]

    with patch(
        "app.services.quiz_service.generate_questions",
        new=AsyncMock(side_effect=_always_one_question),
    ):
        resp = await client.post(GENERATE_URL, json={"grade": 7, "subject": "Mathematics", "question_count": 10})

    assert resp.status_code == 201
    assert call_count["n"] <= settings.GROQ_MAX_GENERATION_CALLS_PER_REQUEST


# ─────────────────────────────────────────────────────────────────────────────
# 4. Telemetry write failure must not break quiz generation
# ─────────────────────────────────────────────────────────────────────────────

async def test_telemetry_write_failure_does_not_break_generation(client, db_session):
    with (
        patch("app.services.quiz_service.generate_questions", new=_generate_questions_mock()),
        patch(
            "app.services.quiz_service.telemetry_service.record_generation_event",
            new=AsyncMock(side_effect=RuntimeError("simulated telemetry failure")),
        ),
    ):
        resp = await client.post(GENERATE_URL, json={"subject": "Mathematics", "question_count": 3})

    # The quiz itself must still be generated successfully even though
    # telemetry recording raised (bypassing its own internal safety net,
    # simulating a genuinely unexpected failure) — see the try/except around
    # the call site in quiz_service.generate_quiz()'s `finally` block.
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["questions"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 5. Endpoint authorization
# ─────────────────────────────────────────────────────────────────────────────

async def test_telemetry_endpoint_allowed_in_development(client, db_session):
    assert settings.ENVIRONMENT == "development"  # default assumed here
    resp = await client.get(TELEMETRY_URL)
    assert resp.status_code == 200


async def test_telemetry_endpoint_forbidden_in_production_for_non_admin(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ADMIN_CLERK_IDS", "")
    resp = await client.get(TELEMETRY_URL)
    assert resp.status_code == 403


async def test_telemetry_endpoint_allowed_in_production_for_admin(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ADMIN_CLERK_IDS", TEST_CLERK_ID)
    resp = await client.get(TELEMETRY_URL)
    assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 6. Date-range filtering
# ─────────────────────────────────────────────────────────────────────────────

async def test_telemetry_endpoint_date_range_filtering(client, db_session):
    now = datetime.now(timezone.utc)
    old_event = AIGenerationEvent(
        subject="OldSubject", requested_question_count=5, generated_question_count=5,
        provider="groq", model_name="test-model", success=True, used_cache_fallback=False,
        retry_count=0, duplicate_count=0, invalid_question_count=0, latency_ms=100.0,
        created_at=now - timedelta(days=10),
    )
    recent_event = AIGenerationEvent(
        subject="RecentSubject", requested_question_count=5, generated_question_count=5,
        provider="groq", model_name="test-model", success=True, used_cache_fallback=False,
        retry_count=0, duplicate_count=0, invalid_question_count=0, latency_ms=200.0,
        created_at=now - timedelta(hours=1),
    )
    db_session.add(old_event)
    db_session.add(recent_event)
    await db_session.commit()

    # Unfiltered: both events counted.
    resp_all = await client.get(TELEMETRY_URL)
    assert resp_all.json()["total_generation_requests"] == 2

    # Filtered to the last 2 days: only the recent event.
    start_date = (now - timedelta(days=2)).isoformat()
    resp_recent = await client.get(TELEMETRY_URL, params={"start_date": start_date})
    data_recent = resp_recent.json()
    assert data_recent["total_generation_requests"] == 1
    assert data_recent["average_generation_latency_ms"] == 200.0

    # Filtered to before 2 days ago: only the old event.
    end_date = (now - timedelta(days=2)).isoformat()
    resp_old = await client.get(TELEMETRY_URL, params={"end_date": end_date})
    data_old = resp_old.json()
    assert data_old["total_generation_requests"] == 1
    assert data_old["average_generation_latency_ms"] == 100.0
