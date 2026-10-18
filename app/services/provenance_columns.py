"""Writes the RAG provenance columns added by alembic migration f7c1e2d3a4b5:

- questions.source_pages             -- which textbook page(s) an AI question was based on (see source_labels)
- ai_generation_events.rag_snippet_count -- how many textbook excerpts a generation request received

These columns are deliberately NOT mapped on the Question / AIGenerationEvent models: SQLAlchemy writes every
mapped column on a single-row INSERT (as NULL when unset), so a mapped-but-missing column breaks every insert
on a database that hasn't had the migration applied yet. Instead they're written here with explicit UPDATEs,
and only when RAG_PROVENANCE_COLUMNS_ENABLED is on -- with it off, no SQL ever mentions them.
"""
from sqlalchemy import JSON, Integer, column, table, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

_questions = table("questions", column("id", Integer), column("source_pages", JSON))
_events = table("ai_generation_events", column("id", Integer), column("rag_snippet_count", Integer))


def enabled() -> bool:
    return settings.RAG_PROVENANCE_COLUMNS_ENABLED


async def save_source_pages(db: AsyncSession, source_pages_by_question_id: dict[int, list[dict]]) -> None:
    """Caller commits. No-op when disabled or nothing to save."""
    if not enabled():
        return
    for question_id, source_pages in source_pages_by_question_id.items():
        if source_pages:
            await db.execute(update(_questions).where(_questions.c.id == question_id).values(source_pages=source_pages))


async def save_snippet_count(db: AsyncSession, event_id: int, rag_snippet_count: int | None) -> None:
    """Caller commits. No-op when disabled."""
    if not enabled() or rag_snippet_count is None:
        return
    await db.execute(update(_events).where(_events.c.id == event_id).values(rag_snippet_count=rag_snippet_count))
