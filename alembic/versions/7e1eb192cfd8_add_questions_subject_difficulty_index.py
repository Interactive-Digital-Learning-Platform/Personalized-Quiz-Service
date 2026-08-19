"""add questions subject+difficulty composite index

Quiz generation is moving from "call Groq on every request, DB is a
failure-only fallback" to "read a pre-generated question pool per
(subject, difficulty) first, call Groq only for the shortfall" (see
quiz_service._fetch_pool_questions). That read filters on both columns
together on every single quiz generation request; the existing
single-column indexes on `subject` and `difficulty` don't cover that
combined filter efficiently, so this adds a composite index for it.

Revision ID: 7e1eb192cfd8
Revises: d4a7f3c9e1b2
Create Date: 2026-08-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '7e1eb192cfd8'
down_revision: Union[str, Sequence[str], None] = 'd4a7f3c9e1b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "ix_questions_subject_difficulty",
        "questions",
        ["subject", "difficulty"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_questions_subject_difficulty", table_name="questions")
