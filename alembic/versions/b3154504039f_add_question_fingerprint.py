"""add question_fingerprint

Adds `questions.question_fingerprint` — a deterministic sha256 hash of each
question's normalized (subject, lesson, question text), used to identify
"the same or an equivalent question" shown more than once across separate
generation calls (see app/services/question_fingerprint.py and the
repeated-question analytics in analytics_service.get_user_analytics()).

Existing rows are backfilled with the same normalization/hash logic the app
uses for new rows, so old and new rows group together correctly. The column
is added nullable first, backfilled, then locked to NOT NULL — safe for a
table that already has rows (fresh installs get the column NOT NULL from the
start via Base.metadata.create_all(), so this migration only matters for a
database that already has a `questions` table without this column).

Revision ID: b3154504039f
Revises:
Create Date: 2026-07-30 03:31:08.795373

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.services.question_fingerprint import compute_question_fingerprint


# revision identifiers, used by Alembic.
revision: str = 'b3154504039f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "questions",
        sa.Column("question_fingerprint", sa.String(length=64), nullable=True),
    )

    # ── Backfill existing rows ──────────────────────────────────────────────
    connection = op.get_bind()
    questions_table = sa.table(
        "questions",
        sa.column("id", sa.Integer),
        sa.column("subject", sa.String),
        sa.column("lesson", sa.String),
        sa.column("question", sa.String),
        sa.column("question_fingerprint", sa.String),
    )
    rows = connection.execute(
        sa.select(
            questions_table.c.id,
            questions_table.c.subject,
            questions_table.c.lesson,
            questions_table.c.question,
        )
    ).fetchall()
    for row in rows:
        fingerprint = compute_question_fingerprint(row.subject, row.lesson, row.question)
        connection.execute(
            questions_table.update()
            .where(questions_table.c.id == row.id)
            .values(question_fingerprint=fingerprint)
        )

    op.alter_column("questions", "question_fingerprint", nullable=False)

    # Name matches what SQLAlchemy's `Column(..., index=True)` would generate
    # on a fresh install via create_all(), so there's never a naming mismatch
    # or duplicate-index conflict between the two schema-creation paths.
    op.create_index(
        "ix_questions_question_fingerprint", "questions", ["question_fingerprint"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_questions_question_fingerprint", table_name="questions")
    op.drop_column("questions", "question_fingerprint")
