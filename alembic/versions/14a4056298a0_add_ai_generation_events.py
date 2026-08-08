"""add ai_generation_events

Adds the `ai_generation_events` table — internal, technical telemetry for
the Groq quiz-generation pipeline (see app/models/ai_generation_event.py and
app/services/telemetry_service.py). One row per POST /quiz/generate request,
recorded regardless of success/failure. Stores no sensitive data: no API
keys, JWTs, prompts, or raw exception messages — only categorical/numeric
fields.

Revision ID: 14a4056298a0
Revises: b3154504039f
Create Date: 2026-07-30 04:40:35.908928

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '14a4056298a0'
down_revision: Union[str, Sequence[str], None] = 'b3154504039f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ai_generation_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id", sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column(
            "session_id", sa.Integer(),
            sa.ForeignKey("quiz_sessions.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("subject", sa.String(length=100), nullable=False),
        sa.Column("requested_question_count", sa.Integer(), nullable=False),
        sa.Column("generated_question_count", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("used_cache_fallback", sa.Boolean(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("duplicate_count", sa.Integer(), nullable=False),
        sa.Column("invalid_question_count", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("error_category", sa.String(length=50), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
    )

    # Matches the ORM model's `index=True` columns, and satisfies the
    # requirement to index created_at/success/used_cache_fallback
    # specifically (the fields the admin analytics endpoint filters/groups
    # by most).
    op.create_index("ix_ai_generation_events_id", "ai_generation_events", ["id"])
    op.create_index("ix_ai_generation_events_user_id", "ai_generation_events", ["user_id"])
    op.create_index("ix_ai_generation_events_success", "ai_generation_events", ["success"])
    op.create_index(
        "ix_ai_generation_events_used_cache_fallback", "ai_generation_events", ["used_cache_fallback"]
    )
    op.create_index("ix_ai_generation_events_created_at", "ai_generation_events", ["created_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_ai_generation_events_created_at", table_name="ai_generation_events")
    op.drop_index("ix_ai_generation_events_used_cache_fallback", table_name="ai_generation_events")
    op.drop_index("ix_ai_generation_events_success", table_name="ai_generation_events")
    op.drop_index("ix_ai_generation_events_user_id", table_name="ai_generation_events")
    op.drop_index("ix_ai_generation_events_id", table_name="ai_generation_events")
    op.drop_table("ai_generation_events")
