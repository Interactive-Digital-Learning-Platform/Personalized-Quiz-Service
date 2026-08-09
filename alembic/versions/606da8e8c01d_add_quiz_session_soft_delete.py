"""add quiz_sessions.deleted_at soft-delete column

Deleting a quiz session used to hard-delete the row (cascading to its
attempts/snapshots/completion). That destroyed data that analytics
(trend/growth/difficulty/repeated-question stats) and quiz generation
(lesson-variety avoidance) read directly from quiz_sessions and its
children, not just the separate aggregate `analytics` table. Switching
DELETE /quiz/sessions/{id} to soft-delete (set `deleted_at`) keeps that
history intact while still removing the session from the user-facing list.

Revision ID: 606da8e8c01d
Revises: 14a4056298a0
Create Date: 2026-08-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '606da8e8c01d'
down_revision: Union[str, Sequence[str], None] = '14a4056298a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "quiz_sessions",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("quiz_sessions", "deleted_at")
