"""add quiz_sessions retake fields

Restarting a quiz used to resubmit into the SAME session_id, which the
backend rejected with 409 once that session already had a QuizCompletion
row. Restart now clones the original session into a new row instead, and
this migration adds the columns that track it: `is_retake` (so every
analytics query that already filters `deleted_at IS NULL` can also filter
retakes out — the user has already seen the correct answers on a retake,
so its results must never influence analytics or adaptive difficulty) and
`retake_of_session_id` (points at the original session, for traceability).

Revision ID: d4a7f3c9e1b2
Revises: c2bf4d16bb87
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4a7f3c9e1b2'
down_revision: Union[str, Sequence[str], None] = 'c2bf4d16bb87'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "quiz_sessions",
        sa.Column("is_retake", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "quiz_sessions",
        sa.Column(
            "retake_of_session_id",
            sa.Integer(),
            sa.ForeignKey("quiz_sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_quiz_sessions_retake_of_session_id",
        "quiz_sessions",
        ["retake_of_session_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_quiz_sessions_retake_of_session_id", table_name="quiz_sessions")
    op.drop_column("quiz_sessions", "retake_of_session_id")
    op.drop_column("quiz_sessions", "is_retake")
