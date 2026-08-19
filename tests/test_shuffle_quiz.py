"""
tests/test_shuffle_quiz.py
────────────────────────────
Covers the Shuffle/Remix quiz mode (GenerateQuizRequest.shuffle=True):

- Generation: quiz_service.generate_quiz() builds one quiz spanning multiple
  subjects via a SINGLE combined Groq call (see
  groq_service.generate_shuffle_quiz_questions) instead of one call per
  subject, and sets QuizSession.subject to "Mixed" when the resulting
  questions actually span more than one subject. Any shortfall (a subject
  the AI response didn't cover, or a total AI failure) falls back to the DB
  question cache rather than making more Groq calls.
- Submission: quiz_service.submit_quiz() must attribute each subject's
  mastery evidence to THAT subject's own SubjectMastery/LessonMastery rows,
  not to a single subject (e.g. the "Mixed" session subject literal) — a
  naive implementation would silently corrupt mastery data for shuffle
  quizzes, since one submission's graded_answers span several subjects.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException
from groq import BadRequestError
from sqlalchemy import select

from app.models.lesson_mastery import LessonMastery
from app.models.question import Question
from app.models.quiz_session import QuizSession
from app.models.subject_mastery import SubjectMastery
from app.schemas.quiz import GenerateQuizRequest
from app.services import groq_service
from app.services.quiz_service import generate_quiz, get_or_create_user
from tests.conftest import TEST_CLERK_ID


def _fake_chat_completion(questions: list[dict]):
    content = json.dumps({"questions": questions})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _canned_shuffle_questions(specs: list[tuple[str, int, str]]) -> list[dict]:
    """specs = [(subject, count, tag), ...] -> a flat list of canned
    question dicts tagged with "subject", the shape
    generate_shuffle_quiz_questions() expects back from one combined Groq
    call covering every subject at once."""
    out = []
    for subject, n, tag in specs:
        for i in range(n):
            out.append({
                "subject": subject,
                "lesson": f"{tag} Lesson{i}",
                "question": f"{tag} Question {i}?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "explanation": "because",
            })
    return out


def _bad_request(detail: str = "bad request"):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(detail, response=response, body=None)


# ── groq_service.generate_shuffle_quiz_questions() unit tests ────────────

@pytest.mark.asyncio
async def test_generate_shuffle_quiz_questions_makes_exactly_one_call():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion(
            _canned_shuffle_questions([("Mathematics", 2, "Math"), ("Science", 2, "Sci")])
        )
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        result = await groq_service.generate_shuffle_quiz_questions(
            grade=10,
            subject_plan=[("Mathematics", "easy", 2), ("Science", "medium", 2)],
        )

    assert mock_create.call_count == 1
    assert len(result) == 4
    assert {q["subject"] for q in result} == {"Mathematics", "Science"}
    # difficulty always comes from the plan, never trusted from the AI.
    assert all(q["difficulty"] == "easy" for q in result if q["subject"] == "Mathematics")
    assert all(q["difficulty"] == "medium" for q in result if q["subject"] == "Science")


@pytest.mark.asyncio
async def test_generate_shuffle_quiz_questions_drops_unrecognized_subject():
    """A question tagged with a subject that wasn't in subject_plan (e.g.
    the model invents/renames one) must be dropped rather than kept under
    a difficulty we never resolved for it."""
    mock_create = AsyncMock(
        return_value=_fake_chat_completion(
            _canned_shuffle_questions([("Mathematics", 1, "Math"), ("History", 1, "Hist")])
        )
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        result = await groq_service.generate_shuffle_quiz_questions(
            grade=10, subject_plan=[("Mathematics", "easy", 1)],
        )

    assert len(result) == 1
    assert result[0]["subject"] == "Mathematics"


# ── quiz_service.generate_quiz() shuffle integration tests ───────────────

@pytest.mark.asyncio
async def test_shuffle_generation_spans_subjects_and_sets_mixed(db_session):
    mock_create = AsyncMock(
        return_value=_fake_chat_completion(
            _canned_shuffle_questions([("Mathematics", 2, "Math"), ("Science", 2, "Sci")])
        )
    )
    payload = GenerateQuizRequest(shuffle=True, subjects=["Mathematics", "Science"], question_count=4)

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.quiz_service._is_near_duplicate", return_value=False),
    ):
        session, questions, cache_hit, difficulty, lesson = await generate_quiz(
            db=db_session, clerk_id=TEST_CLERK_ID, payload=payload,
        )

    assert session.subject == "Mixed"
    assert {q.subject for q in questions} == {"Mathematics", "Science"}
    assert len(questions) == 4
    # The whole shuffle AI phase is ONE combined Groq call, regardless of
    # how many subjects it covers.
    assert mock_create.call_count == 1
    assert cache_hit is False


@pytest.mark.asyncio
async def test_shuffle_generation_single_effective_subject_is_not_mixed(db_session):
    """allocate_question_counts can give every question to a single subject
    (e.g. question_count=1 spread across 6 subjects) — the session should
    then reflect that one subject, not "Mixed". _allocate_shuffle_subject_
    counts() is patched to a fixed single-subject result so which subject
    gets picked doesn't depend on randomness here -- that randomization
    itself is covered separately below."""
    all_six = ["Mathematics", "Science", "History", "English", "Geography", "Programming"]
    mock_create = AsyncMock(
        return_value=_fake_chat_completion(_canned_shuffle_questions([("Programming", 1, "Prog")]))
    )
    payload = GenerateQuizRequest(shuffle=True, subjects=all_six, question_count=1)

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.quiz_service._allocate_shuffle_subject_counts", return_value={"Programming": 1}),
    ):
        session, questions, cache_hit, difficulty, lesson = await generate_quiz(
            db=db_session, clerk_id=TEST_CLERK_ID, payload=payload,
        )

    assert len(questions) == 1
    assert questions[0].subject == "Programming"
    assert session.subject == "Programming"
    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_shuffle_submission_updates_mastery_independently_per_subject(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    session = QuizSession(user_id=user.id, subject="Mixed", lesson="Mixed", difficulty="easy", question_count=4)
    db_session.add(session)
    await db_session.flush()

    math_q1 = Question(
        question="M1?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="easy",
    )
    math_q2 = Question(
        question="M2?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="easy",
    )
    sci_q1 = Question(
        question="S1?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Science", lesson="Biology", difficulty="easy",
    )
    sci_q2 = Question(
        question="S2?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Science", lesson="Biology", difficulty="easy",
    )
    db_session.add_all([math_q1, math_q2, sci_q1, sci_q2])
    await db_session.commit()
    for q in (math_q1, math_q2, sci_q1, sci_q2):
        await db_session.refresh(q)

    resp = await client.post(
        "/api/v1/quiz/submit",
        json={
            "session_id": session.id,
            "ended_by": "submitted",
            "answers": [
                {"question_id": math_q1.id, "selected_answer": "A", "response_time": 4.0},
                {"question_id": math_q2.id, "selected_answer": "A", "response_time": 4.0},
                {"question_id": sci_q1.id, "selected_answer": "B", "response_time": 4.0},
                {"question_id": sci_q2.id, "selected_answer": "B", "response_time": 4.0},
            ],
        },
    )
    assert resp.status_code == 200

    math_mastery = (await db_session.execute(
        select(SubjectMastery).where(SubjectMastery.user_id == user.id, SubjectMastery.subject == "Mathematics")
    )).scalar_one()
    science_mastery = (await db_session.execute(
        select(SubjectMastery).where(SubjectMastery.user_id == user.id, SubjectMastery.subject == "Science")
    )).scalar_one()

    # The bug this test guards against: a naive implementation attributes
    # every subject's evidence to the session's own (literal "Mixed")
    # subject instead of each question's real subject.
    mixed_mastery = (await db_session.execute(
        select(SubjectMastery).where(SubjectMastery.user_id == user.id, SubjectMastery.subject == "Mixed")
    )).scalar_one_or_none()
    assert mixed_mastery is None

    assert math_mastery.evidence_count == 2
    assert science_mastery.evidence_count == 2
    assert math_mastery.last_accuracy == 100.0
    assert science_mastery.last_accuracy == 0.0
    assert math_mastery.mastery_score > science_mastery.mastery_score

    math_lesson = (await db_session.execute(
        select(LessonMastery).where(
            LessonMastery.user_id == user.id, LessonMastery.subject == "Mathematics", LessonMastery.lesson == "Algebra",
        )
    )).scalar_one()
    science_lesson = (await db_session.execute(
        select(LessonMastery).where(
            LessonMastery.user_id == user.id, LessonMastery.subject == "Science", LessonMastery.lesson == "Biology",
        )
    )).scalar_one()

    assert math_lesson.evidence_count == 2
    assert science_lesson.evidence_count == 2
    assert math_lesson.last_accuracy == 100.0
    assert science_lesson.last_accuracy == 0.0


@pytest.mark.asyncio
async def test_shuffle_global_db_fallback_fills_deficit_from_a_different_subject(db_session):
    """A subject the AI response doesn't cover AND has no cached questions
    of its own must not fail the whole shuffle request -- once its own
    subject-level DB fallback comes up empty, the deficit should be filled
    from ANOTHER shuffle subject's cache pool (the global fallback tier)
    instead, so the quiz still reaches the requested count.
    """
    # No cached Mathematics question at all -- its own subject-fallback tier
    # will come up empty. A cached Science question the AI phase won't
    # touch (AI always creates fresh rows) is there for the GLOBAL fallback
    # to find instead.
    db_session.add(Question(
        question="Cached Science fallback?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Science", lesson="Biology", difficulty="easy",
    ))
    await db_session.commit()

    # The single combined call covers Science and History but not
    # Mathematics -- as if the model just didn't produce one for it.
    mock_create = AsyncMock(
        return_value=_fake_chat_completion(
            _canned_shuffle_questions([("Science", 1, "Sci"), ("History", 1, "Hist")])
        )
    )
    payload = GenerateQuizRequest(shuffle=True, subjects=["Mathematics", "Science", "History"], question_count=3)

    with (
        patch.object(groq_service._groq_client.chat.completions, "create", mock_create),
        patch("app.services.quiz_service._is_near_duplicate", return_value=False),
    ):
        session, questions, cache_hit, difficulty, lesson = await generate_quiz(
            db=db_session, clerk_id=TEST_CLERK_ID, payload=payload,
        )

    assert mock_create.call_count == 1
    assert len(questions) == 3
    assert cache_hit is True
    # Mathematics never got a question -- no AI coverage, no cache of its
    # own -- so its slot was rebalanced onto Science's cache pool instead.
    subjects_present = {q.subject for q in questions}
    assert "Mathematics" not in subjects_present
    science_questions = [q for q in questions if q.subject == "Science"]
    history_questions = [q for q in questions if q.subject == "History"]
    assert len(science_questions) == 2  # 1 AI + 1 global-fallback cache row
    assert len(history_questions) == 1
    assert any(q.question == "Cached Science fallback?" for q in science_questions)


@pytest.mark.asyncio
async def test_shuffle_falls_back_to_db_entirely_when_the_ai_call_fails_outright(db_session):
    """If the single combined Groq call fails outright, every subject's
    full count becomes a DB-fallback deficit instead of failing the
    request -- as long as cached questions exist for those subjects.
    """
    db_session.add(Question(
        question="Cached Math fallback?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Arithmetic", difficulty="easy",
    ))
    db_session.add(Question(
        question="Cached Science fallback?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Science", lesson="Biology", difficulty="easy",
    ))
    await db_session.commit()

    mock_create = AsyncMock(side_effect=_bad_request())
    payload = GenerateQuizRequest(shuffle=True, subjects=["Mathematics", "Science"], question_count=2)

    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        session, questions, cache_hit, difficulty, lesson = await generate_quiz(
            db=db_session, clerk_id=TEST_CLERK_ID, payload=payload,
        )

    assert mock_create.call_count == 1
    assert len(questions) == 2
    assert cache_hit is True
    assert {q.subject for q in questions} == {"Mathematics", "Science"}


@pytest.mark.asyncio
async def test_shuffle_raises_clean_502_when_every_fallback_is_exhausted(db_session):
    """When the combined AI call fails and there's no cached question
    anywhere to fall back on, the request must still fail -- but with a
    clean, generic application-level message, never a leaked provider
    error string.
    """
    mock_create = AsyncMock(side_effect=_bad_request("some internal groq detail that must not leak"))
    payload = GenerateQuizRequest(shuffle=True, subjects=["Mathematics", "Science"], question_count=2)

    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        with pytest.raises(HTTPException) as exc_info:
            await generate_quiz(db=db_session, clerk_id=TEST_CLERK_ID, payload=payload)

    assert exc_info.value.status_code == 502
    detail = str(exc_info.value.detail)
    assert "Unable to generate enough unique questions for Shuffle Mode" in detail
    assert "some internal groq detail" not in detail


@pytest.mark.asyncio
async def test_shuffle_force_cache_skips_ai_entirely(db_session):
    """payload.force_cache (the "Use Cache" retry after a prior AI
    failure) must skip the AI phase entirely, going straight to the DB
    fallback for the full requested count."""
    db_session.add(Question(
        question="Cached Math fallback?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Arithmetic", difficulty="easy",
    ))
    await db_session.commit()

    mock_create = AsyncMock()
    payload = GenerateQuizRequest(
        shuffle=True, subjects=["Mathematics"], question_count=1, force_cache=True,
    )

    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        session, questions, cache_hit, difficulty, lesson = await generate_quiz(
            db=db_session, clerk_id=TEST_CLERK_ID, payload=payload,
        )

    mock_create.assert_not_called()
    assert cache_hit is True
    assert len(questions) == 1
    assert questions[0].question == "Cached Math fallback?"


def test_allocate_shuffle_subject_counts_even_division_needs_no_randomness():
    from app.services import quiz_service
    counts = quiz_service._allocate_shuffle_subject_counts(["Mathematics", "Science"], 4)
    assert counts == {"Mathematics": 2, "Science": 2}


def test_allocate_shuffle_subject_counts_selects_random_subset_when_subjects_exceed_total():
    from app.services import quiz_service
    subjects = ["Mathematics", "Science", "History", "English", "Geography", "Programming"]
    counts = quiz_service._allocate_shuffle_subject_counts(subjects, 2)
    assert len(counts) == 2
    assert all(count == 1 for count in counts.values())
    assert set(counts).issubset(set(subjects))


def test_allocate_shuffle_subject_counts_randomizes_which_subjects_get_the_remainder():
    from app.services import quiz_service
    subjects = ["Mathematics", "Science", "History"]
    # 4 questions across 3 subjects: base=1 each, 1 leftover -- across many
    # runs, the leftover should not always land on the same subject (the
    # whole point of randomizing it, vs. the deterministic largest-
    # remainder method challenge-zone tiers use).
    winners = set()
    for _ in range(50):
        counts = quiz_service._allocate_shuffle_subject_counts(subjects, 4)
        assert sum(counts.values()) == 4
        winners.add(next(s for s, c in counts.items() if c == 2))
    assert len(winners) > 1
