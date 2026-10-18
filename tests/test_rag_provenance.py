"""
tests/test_rag_provenance.py
────────────────────────────
Which textbook page each AI question came from, and how many excerpts a generation got:

- excerpts carry a "[Source: <book>, page <n>]" label; the AI's "source" is kept only if it is one of the
  labels it was actually given;
- the new columns aren't mapped on the models (a mapped-but-missing column breaks single-row INSERTs on
  Postgres), and nothing writes them until RAG_PROVENANCE_COLUMNS_ENABLED is on, so the app keeps working
  against a database that hasn't had migration f7c1e2d3a4b5 applied.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, text

from app.core.config import settings
from app.models.ai_generation_event import AIGenerationEvent
from app.models.question import Question
from app.services import groq_service, provenance_columns, quiz_service, source_labels, telemetry_service

LABEL = "[Source: science grade 10 part 1.pdf, page 46]"


def test_labels_format_and_parse():
    assert source_labels.format_label({"filename": "science grade 10 part 1.pdf", "page_start": 46}) == LABEL
    assert source_labels.format_label({"filename": "x.pdf", "page_start": 0}) is None  # unknown page
    assert source_labels.labels_in([f"{LABEL} Acceleration is ...", "no label here"]) == {LABEL}
    assert source_labels.to_source_pages(LABEL, {LABEL}) == [{"file": "science grade 10 part 1.pdf", "page": 46}]


def test_only_labels_that_were_given_are_accepted():
    invented = "[Source: some other book.pdf, page 3]"
    assert source_labels.to_source_pages(invented, {LABEL}) is None
    assert source_labels.to_source_pages(None, {LABEL}) is None
    assert source_labels.to_source_pages("from the textbook", {LABEL}) is None


async def test_generated_questions_keep_real_sources_and_drop_invented_ones():
    def q(text, source):
        return {"question": text, "options": ["a", "b", "c", "d"], "correct_answer": "a",
                "explanation": "", "lesson": "Motion in a straight line", "source": source}

    content = json.dumps({"questions": [q("Real?", LABEL), q("Made up?", "[Source: fake.pdf, page 1]"), q("None?", None)]})
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    create = AsyncMock(return_value=reply)

    with patch.object(groq_service._groq_client.chat.completions, "create", create):
        result = await groq_service.generate_questions(
            grade=10, subject="Science", difficulty="easy", question_count=3, lesson="Motion in a straight line",
            reference_snippets=[f"{LABEL} Acceleration is the rate of change of velocity."],
        )

    assert [r["source_pages"] for r in result] == [
        [{"file": "science grade 10 part 1.pdf", "page": 46}], None, None,
    ]
    prompt = create.await_args.kwargs["messages"][0]["content"]
    assert 'Add a "source"' in prompt and "label of the excerpt the question is based on" in prompt


def test_models_never_put_the_new_columns_into_inserts():
    # Regression: as mapped (even deferred) columns, Postgres single-row INSERTs listed them as NULL and failed
    # with UndefinedColumnError on a database without the migration -- which rolled back and broke quiz generation.
    assert "rag_snippet_count" not in str(AIGenerationEvent.__table__.insert())
    assert "source_pages" not in str(Question.__table__.insert())


def test_snippet_count_covers_flat_and_per_lesson_excerpts():
    assert quiz_service._snippet_count(["a", "b"], {"L1": ["c"], "L2": ["d", "e"]}) == 5
    assert quiz_service._snippet_count(None, None) == 0


async def _record(db, **extra):
    await telemetry_service.record_generation_event(
        db, user_id=None, session_id=None, subject="Science", requested_question_count=5,
        generated_question_count=5, provider="groq", model_name="m", success=True, used_cache_fallback=False,
        retry_count=0, duplicate_count=0, invalid_question_count=0, latency_ms=10.0, error_category=None, **extra,
    )


async def test_telemetry_works_without_the_migration_when_disabled(db_session, monkeypatch):
    # The test DB is built from the models, so -- like an un-migrated database -- it has no rag_snippet_count.
    monkeypatch.setattr(settings, "RAG_PROVENANCE_COLUMNS_ENABLED", False)
    await _record(db_session, rag_snippet_count=7)

    assert (await db_session.execute(select(AIGenerationEvent.id))).scalars().all() != []


async def test_snippet_count_and_source_pages_are_written_once_enabled(db_session, monkeypatch):
    # Simulate migration f7c1e2d3a4b5 having been applied.
    await db_session.execute(text("ALTER TABLE ai_generation_events ADD COLUMN rag_snippet_count INTEGER"))
    await db_session.execute(text("ALTER TABLE questions ADD COLUMN source_pages JSON"))
    await db_session.commit()
    monkeypatch.setattr(settings, "RAG_PROVENANCE_COLUMNS_ENABLED", True)

    await _record(db_session, rag_snippet_count=4)
    question = Question(question="Q?", options=["a", "b", "c", "d"], correct_answer="a",
                        subject="Science", lesson="Motion in a straight line", difficulty="easy")
    db_session.add(question)
    await db_session.commit()
    await provenance_columns.save_source_pages(db_session, {question.id: [{"file": "b.pdf", "page": 46}]})
    await db_session.commit()

    assert (await db_session.execute(text("SELECT rag_snippet_count FROM ai_generation_events"))).scalar_one() == 4
    stored = (await db_session.execute(text("SELECT source_pages FROM questions WHERE id = :i"), {"i": question.id})).scalar_one()
    assert json.loads(stored) == [{"file": "b.pdf", "page": 46}]
