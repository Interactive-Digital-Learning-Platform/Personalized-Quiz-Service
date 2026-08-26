"""add battle gameplay tables

Adds `battle_match_questions` (the frozen, ordered shared question set for a
match) and `battle_answers` (one player's graded answer to one question,
unique(match_id, question_id, user_id) as the DB-level anti-replay guard),
plus `battle_matches.cancel_reason` (populated when a match is cancelled --
e.g. AI question generation failure). See
app/services/battle_gameplay_service.py and
app/services/battle_question_service.py -- Phase 3 of the 1v1 battle system.

Revision ID: 3c540a2ec056
Revises: 361d31e650e9
Create Date: 2026-08-21 09:43:42.306330

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3c540a2ec056'
down_revision: Union[str, Sequence[str], None] = '361d31e650e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("battle_matches", sa.Column("cancel_reason", sa.String(length=255), nullable=True))

    op.create_table(
        "battle_match_questions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "match_id", sa.Integer(),
            sa.ForeignKey("battle_matches.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "question_id", sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("question_order", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint("match_id", "question_order", name="uq_battle_match_question_order"),
    )
    op.create_index("ix_battle_match_questions_id", "battle_match_questions", ["id"])
    op.create_index("ix_battle_match_questions_match_id", "battle_match_questions", ["match_id"])
    op.create_index("ix_battle_match_questions_question_id", "battle_match_questions", ["question_id"])

    op.create_table(
        "battle_answers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "match_id", sa.Integer(),
            sa.ForeignKey("battle_matches.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "question_id", sa.Integer(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "user_id", sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("selected_option", sa.String(length=512), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("response_time_ms", sa.Integer(), nullable=False),
        sa.Column("base_score", sa.Integer(), nullable=False),
        sa.Column("speed_bonus", sa.Integer(), nullable=False),
        sa.Column("streak_bonus", sa.Integer(), nullable=False),
        sa.Column("total_question_score", sa.Integer(), nullable=False),
        sa.Column(
            "answered_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint(
            "match_id", "question_id", "user_id", name="uq_battle_answer_match_question_user"
        ),
    )
    op.create_index("ix_battle_answers_id", "battle_answers", ["id"])
    op.create_index("ix_battle_answers_match_id", "battle_answers", ["match_id"])
    op.create_index("ix_battle_answers_user_id", "battle_answers", ["user_id"])
    op.create_index("ix_battle_answers_match_user", "battle_answers", ["match_id", "user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_battle_answers_match_user", table_name="battle_answers")
    op.drop_index("ix_battle_answers_user_id", table_name="battle_answers")
    op.drop_index("ix_battle_answers_match_id", table_name="battle_answers")
    op.drop_index("ix_battle_answers_id", table_name="battle_answers")
    op.drop_table("battle_answers")

    op.drop_index("ix_battle_match_questions_question_id", table_name="battle_match_questions")
    op.drop_index("ix_battle_match_questions_match_id", table_name="battle_match_questions")
    op.drop_index("ix_battle_match_questions_id", table_name="battle_match_questions")
    op.drop_table("battle_match_questions")

    op.drop_column("battle_matches", "cancel_reason")
