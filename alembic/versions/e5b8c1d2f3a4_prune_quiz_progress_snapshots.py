"""prune quiz_progress_snapshots to one row per session

save_quiz_progress used to INSERT a new snapshot on every autosave, so
long-running quizzes accumulated hundreds of rows each. It now upserts a
single row per session; this deletes the backlog, keeping only the newest
snapshot (by saved_at, then id) for every session.

Revision ID: e5b8c1d2f3a4
Revises: ace03358b63f
Create Date: 2026-10-13 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e5b8c1d2f3a4'
down_revision: str | Sequence[str] | None = 'ace03358b63f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM quiz_progress_snapshots
        WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY session_id ORDER BY saved_at DESC, id DESC
                ) AS rn
                FROM quiz_progress_snapshots
            ) ranked
            WHERE rn > 1
        )
        """
    )


def downgrade() -> None:
    # The deleted rows were superseded autosaves; there's nothing to restore.
    pass
