"""add grade to curriculum tables

Introduces a fixed subject/lesson curriculum taxonomy (Grade 10/11 Sri
Lankan syllabus, see app/services/curriculum_service.py). Lesson names
collide across grades (e.g. "Percentages" appears in both Grade 10 and
Grade 11 Mathematics), so `grade` must become part of a lesson's identity
for mastery tracking to stay correct — otherwise different-grade content
would silently share one LessonMastery row.

`grade` is added as nullable on all three tables and NOT backfilled:
existing rows predate the taxonomy and are left as legacy data (grade
IS NULL). Only rows created going forward populate it.

Revision ID: 08db25e82b12
Revises: 3c540a2ec056
Create Date: 2026-08-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '08db25e82b12'
down_revision: Union[str, Sequence[str], None] = '3c540a2ec056'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("questions", sa.Column("grade", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_questions_grade"), "questions", ["grade"], unique=False)

    op.add_column("quiz_sessions", sa.Column("grade", sa.Integer(), nullable=True))

    op.add_column("lesson_mastery", sa.Column("grade", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_lesson_mastery_grade"), "lesson_mastery", ["grade"], unique=False)

    # NULL is distinct-from-NULL in a unique constraint, so legacy rows
    # (grade IS NULL) don't collide with each other or with new rows.
    op.drop_constraint("uq_lesson_mastery_user_subject_lesson", "lesson_mastery", type_="unique")
    op.create_unique_constraint(
        "uq_lesson_mastery_user_subject_lesson_grade",
        "lesson_mastery",
        ["user_id", "subject", "lesson", "grade"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_lesson_mastery_user_subject_lesson_grade", "lesson_mastery", type_="unique")
    op.create_unique_constraint(
        "uq_lesson_mastery_user_subject_lesson",
        "lesson_mastery",
        ["user_id", "subject", "lesson"],
    )

    op.drop_index(op.f("ix_lesson_mastery_grade"), table_name="lesson_mastery")
    op.drop_column("lesson_mastery", "grade")

    op.drop_column("quiz_sessions", "grade")

    op.drop_index(op.f("ix_questions_grade"), table_name="questions")
    op.drop_column("questions", "grade")
