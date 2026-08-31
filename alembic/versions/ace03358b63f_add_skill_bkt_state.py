"""add skill bkt state

Adds skill_bkt_state, a new table holding Bayesian Knowledge Tracing
state per (user, subject, lesson, grade) skill — a read-only P(know)
signal, additive alongside (not replacing) LessonMastery's Continuous
Evidence-Weighted System. See app/services/bkt_service.py.

Purely additive: no existing tables are touched.

Revision ID: ace03358b63f
Revises: 08db25e82b12
Create Date: 2026-08-30 21:47:19.562855

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ace03358b63f'
down_revision: Union[str, Sequence[str], None] = '08db25e82b12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "skill_bkt_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=100), nullable=False),
        sa.Column("lesson", sa.String(length=255), nullable=False),
        sa.Column("grade", sa.Integer(), nullable=True),
        sa.Column("p_know", sa.Float(), nullable=False),
        sa.Column("opportunities", sa.Integer(), nullable=False),
        sa.Column("last_correct", sa.Boolean(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "subject", "lesson", "grade", name="uq_skill_bkt_state_user_subject_lesson_grade"
        ),
    )
    op.create_index(op.f("ix_skill_bkt_state_id"), "skill_bkt_state", ["id"], unique=False)
    op.create_index(op.f("ix_skill_bkt_state_user_id"), "skill_bkt_state", ["user_id"], unique=False)
    op.create_index(op.f("ix_skill_bkt_state_subject"), "skill_bkt_state", ["subject"], unique=False)
    op.create_index(op.f("ix_skill_bkt_state_lesson"), "skill_bkt_state", ["lesson"], unique=False)
    op.create_index(op.f("ix_skill_bkt_state_grade"), "skill_bkt_state", ["grade"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_skill_bkt_state_grade"), table_name="skill_bkt_state")
    op.drop_index(op.f("ix_skill_bkt_state_lesson"), table_name="skill_bkt_state")
    op.drop_index(op.f("ix_skill_bkt_state_subject"), table_name="skill_bkt_state")
    op.drop_index(op.f("ix_skill_bkt_state_user_id"), table_name="skill_bkt_state")
    op.drop_index(op.f("ix_skill_bkt_state_id"), table_name="skill_bkt_state")
    op.drop_table("skill_bkt_state")
