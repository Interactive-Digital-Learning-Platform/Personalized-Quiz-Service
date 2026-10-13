"""
tests/test_quiz_integrity.py
─────────────────────────────
Regression tests for the quiz-integrity fixes:

- the answer key never reaches the client before a quiz is submitted
  (/generate's question schema, GET /quiz/sessions/{id} while in progress),
  and comes back per question in the submit response instead;
- submit_quiz grades against the session's own questions: duplicates and
  foreign question ids are rejected, omitted questions count as unanswered;
- races and late writes: concurrent first-login user creation, autosave
  after completion;
- safe defaults: ENVIRONMENT, client-sent usernames.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, settings
from app.models.question import Question
from app.models.quiz_session import QuizSession
from app.models.user import User
from app.schemas.quiz import GenerateQuizResponse, QuestionPublic
from app.services import quiz_service
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_questions(db: AsyncSession, n: int) -> list[Question]:
    qs = [
        Question(
            question=f"Q{i}?", options=["A", "B", "C", "D"], correct_answer="A",
            subject="Mathematics", lesson="Algebra", difficulty="easy",
        )
        for i in range(n)
    ]
    db.add_all(qs)
    await db.flush()
    return qs


async def _make_session(db: AsyncSession, user_id: int, questions: list[Question]) -> QuizSession:
    session = QuizSession(
        user_id=user_id, subject="Mathematics", lesson="Algebra", difficulty="easy",
        question_count=len(questions),
        questions_snapshot=[
            {
                "id": q.id, "question": q.question, "options": q.options, "subject": q.subject,
                "lesson": q.lesson, "difficulty": q.difficulty, "correct_answer": q.correct_answer,
            }
            for q in questions
        ],
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def _setup(db: AsyncSession, n: int = 3):
    user = await get_or_create_user(db, TEST_CLERK_ID)
    questions = await _make_questions(db, n)
    session = await _make_session(db, user.id, questions)
    return session.id, [q.id for q in questions]


def _answer(qid: int, selected: str = "A") -> dict:
    return {"question_id": qid, "selected_answer": selected, "response_time": 2.0}


async def _submit(client, session_id: int, answers: list[dict]):
    return await client.post(
        "/api/v1/quiz/submit",
        json={"session_id": session_id, "ended_by": "submitted", "answers": answers},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Answer key
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_response_schema_has_no_answer_key():
    assert "correct_answer" not in QuestionPublic.model_fields
    q = Question(
        id=1, question="Q?", options=["A", "B"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="easy",
    )
    response = GenerateQuizResponse(
        session_id=1, questions=[QuestionPublic.model_validate(q)], difficulty="easy", lesson="Algebra",
    )
    assert "correct_answer" not in response.model_dump()["questions"][0]


async def test_in_progress_session_hides_answers_until_completed(client, db_session):
    session_id, qids = await _setup(db_session)

    before = await client.get(f"/api/v1/quiz/sessions/{session_id}")
    assert before.status_code == 200
    assert all(q["correct_answer"] is None for q in before.json()["questions"])

    await _submit(client, session_id, [_answer(q) for q in qids])

    after = await client.get(f"/api/v1/quiz/sessions/{session_id}")
    assert all(q["correct_answer"] == "A" for q in after.json()["questions"])


async def test_submit_returns_per_question_results_in_session_order(client, db_session):
    session_id, qids = await _setup(db_session)

    resp = await _submit(client, session_id, [_answer(qids[2], "B"), _answer(qids[0], "A"), _answer(qids[1], "A")])

    assert resp.status_code == 200
    results = resp.json()["question_results"]
    assert [r["question_id"] for r in results] == qids
    assert [r["is_correct"] for r in results] == [True, True, False]
    assert all(r["correct_answer"] == "A" for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Submission validation
# ─────────────────────────────────────────────────────────────────────────────

async def test_duplicate_answers_for_one_question_are_rejected(client, db_session):
    session_id, qids = await _setup(db_session)

    resp = await _submit(client, session_id, [_answer(qids[0])] * 3)

    assert resp.status_code == 422
    # Nothing was recorded, so a corrected submission still goes through.
    assert (await _submit(client, session_id, [_answer(q) for q in qids])).status_code == 200


async def test_answers_for_questions_outside_the_session_are_rejected(client, db_session):
    session_id, qids = await _setup(db_session)
    other_session_id, other_qids = await _setup(db_session)

    resp = await _submit(client, session_id, [_answer(qids[0]), _answer(other_qids[0])])

    assert resp.status_code == 422
    assert str(other_qids[0]) in resp.json()["detail"]


async def test_omitted_questions_count_as_unanswered(client, db_session):
    session_id, qids = await _setup(db_session, n=4)

    resp = await _submit(client, session_id, [_answer(qids[0])])

    data = resp.json()
    assert resp.status_code == 200
    assert data["total_questions"] == 4
    assert data["correct_count"] == 1
    assert data["accuracy"] == 25.0
    assert [r["selected_answer"] for r in data["question_results"]] == ["A", "", "", ""]


async def test_second_submit_is_a_409(client, db_session):
    session_id, qids = await _setup(db_session)

    assert (await _submit(client, session_id, [_answer(q) for q in qids])).status_code == 200
    assert (await _submit(client, session_id, [_answer(q) for q in qids])).status_code == 409


async def test_progress_save_after_completion_is_rejected(client, db_session):
    session_id, qids = await _setup(db_session)
    await _submit(client, session_id, [_answer(q) for q in qids])

    resp = await client.post(
        "/api/v1/quiz/progress/save",
        json={"session_id": session_id, "remaining_time": 10, "answered_count": 1},
    )

    assert resp.status_code == 409
    sessions = (await client.get("/api/v1/quiz/sessions")).json()
    assert sessions[0]["answered_count"] == len(qids)  # the final snapshot still wins


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency / defaults
# ─────────────────────────────────────────────────────────────────────────────

class _MissFirstSelect:
    """Wraps a session so its first execute() behaves as if the user row didn't
    exist yet -- i.e. another request inserted it between our SELECT and INSERT."""

    def __init__(self, inner: AsyncSession):
        self._inner = inner
        self._missed = False

    async def execute(self, *args, **kwargs):
        result = await self._inner.execute(*args, **kwargs)
        if not self._missed:
            self._missed = True

            class _Empty:
                def scalar_one_or_none(self):
                    return None

            return _Empty()
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def test_concurrent_first_login_reuses_the_row_instead_of_500(db_session):
    winner = await get_or_create_user(db_session, "racing-user")

    async with AsyncSession(bind=db_session.bind, expire_on_commit=False) as other:
        loser = await quiz_service.get_or_create_user(_MissFirstSelect(other), "racing-user")

    assert loser.id == winner.id


def test_environment_defaults_to_production():
    assert Settings.model_fields["ENVIRONMENT"].default == "production"


async def test_sync_rejects_usernames_outside_clerks_alphabet(client, db_session):
    bad = await client.post("/api/v1/user/sync", json={"username": "<b>Admin</b>"})
    good = await client.post("/api/v1/user/sync", json={"username": "quiz_fan-42"})

    assert bad.status_code == 422
    assert good.status_code == 200
    assert good.json()["username"] == "quiz_fan-42"


async def test_sync_uses_clerk_username_when_secret_configured(client, db_session, monkeypatch):
    monkeypatch.setattr(settings, "CLERK_SECRET_KEY", "sk_test_x")

    async def fake_lookup(clerk_id: str) -> str:
        assert clerk_id == TEST_CLERK_ID
        return "real_name"

    monkeypatch.setattr("app.api.routes.user.fetch_clerk_username", fake_lookup)

    resp = await client.post("/api/v1/user/sync", json={"username": "spoofed"})

    assert resp.json()["username"] == "real_name"
    user = await db_session.get(User, resp.json()["id"])
    await db_session.refresh(user)
    assert user.username == "real_name"


# ─────────────────────────────────────────────────────────────────────────────
# Progress snapshots
# ─────────────────────────────────────────────────────────────────────────────

async def _snapshot_count(db: AsyncSession, session_id: int) -> int:
    from sqlalchemy import func, select

    from app.models.quiz_tracking import QuizProgressSnapshot

    return (await db.execute(
        select(func.count()).select_from(QuizProgressSnapshot).where(QuizProgressSnapshot.session_id == session_id)
    )).scalar_one()


async def test_autosaves_update_one_snapshot_instead_of_piling_up(client, db_session):
    from app.models.quiz_tracking import QuizProgressSnapshot

    session_id, qids = await _setup(db_session)
    # A backlog from before the upsert: the newest of these should be reused, the rest dropped.
    db_session.add_all([QuizProgressSnapshot(session_id=session_id, answered_count=0) for _ in range(3)])
    await db_session.commit()

    for answered in (1, 2):
        resp = await client.post(
            "/api/v1/quiz/progress/save",
            json={"session_id": session_id, "remaining_time": 30, "answered_count": answered},
        )
        assert resp.status_code in (200, 201)

    assert await _snapshot_count(db_session, session_id) == 1
    resumed = (await client.get(f"/api/v1/quiz/sessions/{session_id}")).json()
    assert resumed["latest_progress"]["answered_count"] == 2

    await _submit(client, session_id, [_answer(q) for q in qids])
    assert await _snapshot_count(db_session, session_id) == 1
    assert (await client.get("/api/v1/quiz/sessions")).json()[0]["answered_count"] == len(qids)


async def test_delete_still_succeeds_when_the_analytics_refresh_fails(client, db_session, monkeypatch):
    session_id, qids = await _setup(db_session)
    await _submit(client, session_id, [_answer(q) for q in qids])

    async def boom(**_):
        raise RuntimeError("analytics down")

    monkeypatch.setattr("app.api.routes.quiz.update_analytics_after_submission", boom)

    resp = await client.delete(f"/api/v1/quiz/sessions/{session_id}")

    assert resp.status_code == 204
    assert (await client.get("/api/v1/quiz/sessions")).json() == []


# ─────────────────────────────────────────────────────────────────────────────
# Generated-question validation / retakes
# ─────────────────────────────────────────────────────────────────────────────

def test_question_validator_rejects_duplicate_and_blank_options():
    from app.services.groq_service import _checked_correct_answer as check

    assert check("Q?", ["A", "B", "C", "D"], "B", index=0, kind="q") == "B"
    assert check("Q?", ["Paris", "Rome", "Oslo", "Bern"], " rome ", index=0, kind="q") == "Rome"
    assert check("Q?", ["A", "a ", "C", "D"], "A", index=0, kind="q") is None  # duplicate
    assert check("Q?", ["A", "", "C", "D"], "A", index=0, kind="q") is None  # blank
    assert check("Q?", ["A", 2, "C", "D"], "A", index=0, kind="q") is None  # non-string
    assert check("Q?", ["A", "B", "C"], "A", index=0, kind="q") is None  # wrong count
    assert check("Q?", ["A", "B", "C", "D"], "E", index=0, kind="q") is None  # answer not an option


async def test_retake_keeps_the_grade(client, db_session):
    session_id, qids = await _setup(db_session)
    original = await db_session.get(QuizSession, session_id)
    original.grade = 11
    await db_session.commit()
    await _submit(client, session_id, [_answer(q) for q in qids])

    resp = await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")

    retake = await db_session.get(QuizSession, resp.json()["session_id"])
    assert retake.grade == 11
