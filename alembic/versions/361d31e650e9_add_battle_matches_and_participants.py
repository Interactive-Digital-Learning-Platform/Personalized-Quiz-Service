"""add battle matches and participants

Adds `battle_matches` and `battle_participants` -- Phase 2 of the 1v1 battle
system: the durable Postgres rows created the instant matchmaking pairs two
players (see app/services/matchmaking_service.py, which owns Redis's
transient queue state). BattleMatch.status only ever gets written as
"waiting" by this phase; later phases transition it through
ready/countdown/active/completed/cancelled/forfeited. BattleParticipant's
final_score/correct_count/rating_after/result are nullable -- populated by a
later phase's battle-completion flow, not forced not-null now.

Revision ID: 361d31e650e9
Revises: a185a9d544df
Create Date: 2026-08-21 01:45:16.449913

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '361d31e650e9'
down_revision: Union[str, Sequence[str], None] = 'a185a9d544df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "battle_matches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("subject", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("difficulty", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "winner_user_id", sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
        ),
    )
    op.create_index("ix_battle_matches_id", "battle_matches", ["id"])
    op.create_index("ix_battle_matches_status", "battle_matches", ["status"])

    op.create_table(
        "battle_participants",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "match_id", sa.Integer(),
            sa.ForeignKey("battle_matches.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column(
            "user_id", sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("rating_before", sa.Integer(), nullable=False),
        sa.Column("final_score", sa.Integer(), nullable=True),
        sa.Column("correct_count", sa.Integer(), nullable=True),
        sa.Column("rating_after", sa.Integer(), nullable=True),
        sa.Column("result", sa.String(length=10), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint("match_id", "user_id", name="uq_battle_participant_match_user"),
    )
    op.create_index("ix_battle_participants_id", "battle_participants", ["id"])
    # Match lookup -- e.g. fetch both participants of a match for opponent info.
    op.create_index("ix_battle_participants_match_id", "battle_participants", ["match_id"])
    # User's active-match lookup (join to battle_matches.status).
    op.create_index("ix_battle_participants_user_id", "battle_participants", ["user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_battle_participants_user_id", table_name="battle_participants")
    op.drop_index("ix_battle_participants_match_id", table_name="battle_participants")
    op.drop_index("ix_battle_participants_id", table_name="battle_participants")
    op.drop_table("battle_participants")

    op.drop_index("ix_battle_matches_status", table_name="battle_matches")
    op.drop_index("ix_battle_matches_id", table_name="battle_matches")
    op.drop_table("battle_matches")
