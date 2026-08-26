"""add competitive subject ratings

Adds the `competitive_subject_ratings` table -- Phase 1 of the 1v1 battle
system: per-(user, subject) Elo-style competitive rating (see
app/models/competitive_rating.py, app/services/elo_engine.py,
app/services/league_engine.py). Independent of subject_mastery/lesson_mastery
(the adaptive learning system) -- no FK between them. No live battle engine
exists yet, so nothing writes to this table until a later phase's
battle-completion flow does.

Revision ID: a185a9d544df
Revises: 7e1eb192cfd8
Create Date: 2026-08-20 20:36:34.378639

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a185a9d544df'
down_revision: Union[str, Sequence[str], None] = '7e1eb192cfd8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "competitive_subject_ratings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("subject", sa.String(length=100), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("highest_rating", sa.Integer(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=False),
        sa.Column("losses", sa.Integer(), nullable=False),
        sa.Column("draws", sa.Integer(), nullable=False),
        sa.Column("matches_played", sa.Integer(), nullable=False),
        sa.Column("current_win_streak", sa.Integer(), nullable=False),
        sa.Column("highest_win_streak", sa.Integer(), nullable=False),
        sa.Column("correct_answers", sa.Integer(), nullable=False),
        sa.Column("total_answers", sa.Integer(), nullable=False),
        sa.Column("average_response_time", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint("user_id", "subject", name="uq_competitive_rating_user_subject"),
    )

    # Matches the ORM model's `index=True` columns.
    op.create_index("ix_competitive_subject_ratings_id", "competitive_subject_ratings", ["id"])
    op.create_index(
        "ix_competitive_subject_ratings_user_id", "competitive_subject_ratings", ["user_id"]
    )
    # Leaderboard queries filter by subject and order by rating -- this
    # composite index covers that filter+sort together. The unique
    # constraint above already gives (user_id, subject) lookups an index
    # for the profile-row path.
    op.create_index(
        "ix_competitive_subject_ratings_subject_rating",
        "competitive_subject_ratings", ["subject", "rating"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_competitive_subject_ratings_subject_rating", table_name="competitive_subject_ratings"
    )
    op.drop_index("ix_competitive_subject_ratings_user_id", table_name="competitive_subject_ratings")
    op.drop_index("ix_competitive_subject_ratings_id", table_name="competitive_subject_ratings")
    op.drop_table("competitive_subject_ratings")
