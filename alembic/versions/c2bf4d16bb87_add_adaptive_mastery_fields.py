"""add adaptive mastery fields

Extends subject_mastery and lesson_mastery with the columns the Continuous
Evidence-Weighted Mastery System needs (mastery_score, fluency_score,
confidence_score, evidence_count, recent_accuracy, previous_accuracy,
trend_score, trend_label, retention_score, last_mastery_update). Purely
additive — existing columns (difficulty, last_accuracy, consecutive_strong,
consecutive_weak) are untouched, and every new column has a server-side
default so existing rows stay valid without a data migration.

Revision ID: c2bf4d16bb87
Revises: 606da8e8c01d
Create Date: 2026-08-13 03:57:24.504466

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2bf4d16bb87'
down_revision: Union[str, Sequence[str], None] = '606da8e8c01d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES = ("subject_mastery", "lesson_mastery")


def upgrade() -> None:
    """Upgrade schema."""
    for table in _TABLES:
        op.add_column(table, sa.Column("mastery_score", sa.Float(), nullable=False, server_default="50.0"))
        op.add_column(table, sa.Column("fluency_score", sa.Float(), nullable=False, server_default="50.0"))
        op.add_column(table, sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0.0"))
        op.add_column(table, sa.Column("evidence_count", sa.Integer(), nullable=False, server_default="0"))
        op.add_column(table, sa.Column("recent_accuracy", sa.Float(), nullable=True))
        op.add_column(table, sa.Column("previous_accuracy", sa.Float(), nullable=True))
        op.add_column(table, sa.Column("trend_score", sa.Float(), nullable=True))
        op.add_column(
            table,
            sa.Column("trend_label", sa.String(length=32), nullable=False, server_default="insufficient_data"),
        )
        op.add_column(table, sa.Column("retention_score", sa.Float(), nullable=True))
        op.add_column(table, sa.Column("last_mastery_update", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    for table in _TABLES:
        op.drop_column(table, "last_mastery_update")
        op.drop_column(table, "retention_score")
        op.drop_column(table, "trend_label")
        op.drop_column(table, "trend_score")
        op.drop_column(table, "previous_accuracy")
        op.drop_column(table, "recent_accuracy")
        op.drop_column(table, "evidence_count")
        op.drop_column(table, "confidence_score")
        op.drop_column(table, "fluency_score")
        op.drop_column(table, "mastery_score")
