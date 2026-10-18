"""add RAG provenance columns: ai_generation_events.rag_snippet_count, questions.source_pages

rag_snippet_count records how many textbook excerpts (PDF-Ingestion-Backend-Service's pdf_knowledge_base)
each AI generation request received; source_pages records which textbook page(s) each AI-generated
question was based on. Both nullable -- existing rows stay NULL. The app only writes them once
RAG_PROVENANCE_COLUMNS_ENABLED is set (see app/core/config.py).

Revision ID: f7c1e2d3a4b5
Revises: e5b8c1d2f3a4
Create Date: 2026-10-18 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f7c1e2d3a4b5'
down_revision: str | Sequence[str] | None = 'e5b8c1d2f3a4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('ai_generation_events', sa.Column('rag_snippet_count', sa.Integer(), nullable=True))
    op.add_column('questions', sa.Column('source_pages', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('questions', 'source_pages')
    op.drop_column('ai_generation_events', 'rag_snippet_count')
