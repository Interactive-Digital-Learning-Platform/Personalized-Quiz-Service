"""
core/config.py
──────────────
Central application settings using Pydantic v2 BaseSettings.
All values are loaded from environment variables (or the .env file).
This single source of truth prevents config from being scattered across the app.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── General ───────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Personalized Quiz Service"
    ENVIRONMENT: str = "development"

    # ── Database (Neon PostgreSQL) ────────────────────────────────────────────
    # Must use the asyncpg dialect: postgresql+asyncpg://...
    DATABASE_URL: str

    # ── Groq AI ───────────────────────────────────────────────────────────────
    GROQ_API_KEY: str
    # llama3-70b-8192 gives the best quality/speed trade-off on Groq's free tier
    GROQ_MODEL: str = "llama3-70b-8192"
    # Retries apply ONLY to transient errors (rate limit, timeout, connection,
    # 5xx) — never to auth/bad-request errors, where retrying can't help.
    GROQ_MAX_RETRIES: int = 2
    GROQ_RETRY_BASE_DELAY_SECONDS: float = 1.0

    # ── Clerk Authentication ───────────────────────────────────────────────────
    # JWKS URL is used to fetch Clerk's public keys and verify RS256 JWTs
    CLERK_JWKS_URL: str
    CLERK_ISSUER: str
    # Audience is optional — only set if your Clerk app enforces it
    CLERK_AUDIENCE: Optional[str] = None
    # Allow local development and testing without a real Clerk session token.
    AUTH_BYPASS: bool = False
    # Comma-separated Clerk user IDs (the JWT "sub" claim) allowed to access
    # internal/admin-only endpoints (e.g. GET /analytics/system/ai-generation)
    # outside of development — see core/security.py:require_admin_or_dev().
    ADMIN_CLERK_IDS: str = ""

    # ── Analytics ─────────────────────────────────────────────────────────────
    # An incomplete session (no QuizCompletion) with no activity for this many
    # hours is classified as "abandoned" in GET /analytics/me. See
    # analytics_service.get_user_analytics() for the full classification rules.
    ANALYTICS_ABANDONED_AFTER_HOURS: int = 24
    # A topic with fewer than this many graded attempts is classified as
    # "insufficient_data" rather than strong/developing/weak, regardless of
    # its accuracy — too few data points to draw a reliable conclusion.
    ANALYTICS_TOPIC_MIN_ATTEMPTS: int = 3
    # Max number of per-topic breakdown rows returned per subject in
    # GET /analytics/me (weakest topics first — see get_user_analytics()).
    ANALYTICS_MAX_TOPICS_PER_SUBJECT: int = 10

    # ── Response-time analytics ───────────────────────────────────────────────
    # A QuestionAttempt.response_time is excluded from every response-time
    # statistic (median, avg, stddev, fastest/slowest, etc.) if it's negative
    # or larger than this many seconds. Raw DB rows are NEVER modified because
    # of this — see get_user_analytics()'s response-time section.
    ANALYTICS_MAX_VALID_RESPONSE_TIME_SECONDS: float = 600.0
    # Minimum number of valid-response-time graded attempts required before a
    # scope (overall/subject/topic) gets a real `answering_behavior` value
    # instead of "insufficient_data".
    ANALYTICS_BEHAVIOR_MIN_ATTEMPTS: int = 5
    # Accuracy percentage (0-100) at/above which a scope counts as "accurate"
    # for answering_behavior classification.
    ANALYTICS_ACCURATE_THRESHOLD: float = 70.0
    # "Fast" cutoff used only when the user has zero valid-response-time
    # attempts yet, so their real overall median can't be calculated.
    ANALYTICS_MEDIAN_FALLBACK_SECONDS: float = 10.0
    # If a scope's avg response time is within this fraction of the user's
    # overall median (e.g. 0.15 = +/-15%), classify it as "balanced" rather
    # than definitively "fast" or "slow".
    ANALYTICS_BALANCED_TIME_RATIO: float = 0.15

    # ── Performance trend ─────────────────────────────────────────────────────
    # Number of completed sessions per period for the "recent sessions" trend
    # method (latest N vs the N immediately before them). Used whenever a
    # scope (overall/subject/topic) has at least 2x this many completed
    # sessions; otherwise falls back to weekly comparison. See
    # scoring_service.compute_performance_trend().
    ANALYTICS_TREND_SESSION_WINDOW: int = 5
    # A comparison period (current or previous) with fewer than this many
    # graded attempts is "insufficient_data" rather than a real trend value.
    ANALYTICS_TREND_MIN_ATTEMPTS_PER_PERIOD: int = 1
    # accuracy_change (percentage points) at/above which a scope is
    # "improving", at/below whose negation it's "declining"; otherwise "stable".
    ANALYTICS_TREND_CHANGE_THRESHOLD: float = 5.0

    # ── Mastery score ──────────────────────────────────────────────────────────
    # See app/services/mastery_service.py for the full formula. Weights below
    # must sum to 1.0 (not enforced at runtime — a deliberately simple,
    # directly-editable config surface rather than a derived/validated one).
    ANALYTICS_MASTERY_ACCURACY_WEIGHT: float = 0.35
    ANALYTICS_MASTERY_RECENT_PERFORMANCE_WEIGHT: float = 0.25
    ANALYTICS_MASTERY_DIFFICULTY_WEIGHT: float = 0.15
    ANALYTICS_MASTERY_RETENTION_WEIGHT: float = 0.15
    ANALYTICS_MASTERY_CONSISTENCY_WEIGHT: float = 0.10
    # A subject/topic with fewer than this many graded attempts gets
    # mastery_score=null, mastery_level="insufficient_data" instead of a
    # computed score.
    ANALYTICS_MASTERY_MIN_ATTEMPTS: int = 5
    # Score substituted for a component when ITS OWN data is unavailable
    # (e.g. retention_score with zero repeated-question data, or
    # consistency_score with too few completed sessions) — deliberately the
    # midpoint of the 0-100 range so a missing component neither helps nor
    # hurts the weighted average.
    ANALYTICS_MASTERY_NEUTRAL_SCORE: float = 50.0
    # Minimum completed sessions (with graded attempts) required before
    # consistency_score is computed from actual score variation; below this,
    # ANALYTICS_MASTERY_NEUTRAL_SCORE is used instead.
    ANALYTICS_MASTERY_CONSISTENCY_MIN_SESSIONS: int = 3
    # mastery_level boundaries (mastery_score in 0-100): below DEVELOPING is
    # "beginner", below PROFICIENT is "developing", below ADVANCED is
    # "proficient", at/above ADVANCED is "advanced".
    ANALYTICS_MASTERY_DEVELOPING_THRESHOLD: float = 40.0
    ANALYTICS_MASTERY_PROFICIENT_THRESHOLD: float = 70.0
    ANALYTICS_MASTERY_ADVANCED_THRESHOLD: float = 85.0

    # ── Growth analytics ───────────────────────────────────────────────────────
    # See app/services/growth_service.py for the full formula. Growth is
    # deliberately about RECENT trajectory (effort/consistency/improvement),
    # not raw performance — mastery_score is included but capped to a modest
    # weight so a high raw score can't dominate the result (requirement: growth
    # should reward the process, not just the outcome).
    ANALYTICS_GROWTH_IMPROVEMENT_WEIGHT: float = 0.40
    ANALYTICS_GROWTH_CONSISTENCY_WEIGHT: float = 0.25
    ANALYTICS_GROWTH_EFFORT_WEIGHT: float = 0.20
    ANALYTICS_GROWTH_MASTERY_WEIGHT: float = 0.15
    # growth_level boundaries (growth_score in 0-100).
    ANALYTICS_GROWTH_GROWING_THRESHOLD: float = 40.0
    ANALYTICS_GROWTH_STRONG_THRESHOLD: float = 70.0
    ANALYTICS_GROWTH_EXCEPTIONAL_THRESHOLD: float = 85.0
    # Rolling window (days) that effort_score and consistency_score are
    # computed over — see get_user_analytics()'s growth section.
    ANALYTICS_GROWTH_WINDOW_DAYS: int = 30
    # Fewer than this many graded attempts within the window means there's
    # not enough recent activity to describe a growth trajectory at all —
    # the whole `growth` object returns null scores / "insufficient_data".
    ANALYTICS_GROWTH_MIN_ATTEMPTS_IN_WINDOW: int = 3
    # "Full marks" targets for effort_score's count-based components — raw
    # counts are normalized against these (count/target*100, capped at 100),
    # so effort has a ceiling instead of growing unbounded with volume.
    ANALYTICS_GROWTH_EFFORT_TARGET_QUIZZES: int = 20
    ANALYTICS_GROWTH_EFFORT_TARGET_QUESTIONS: int = 150
    ANALYTICS_GROWTH_EFFORT_TARGET_WEAK_TOPIC_ATTEMPTS: int = 10

    # ── Recommendations ────────────────────────────────────────────────────────
    # See app/services/recommendation_service.py. Deterministic and
    # database-driven only — never calls Groq.
    ANALYTICS_RECOMMENDATION_MAX_COUNT: int = 5
    # A topic's repeated_same_mistakes count at/above this triggers a
    # "repeated_mistake" recommendation.
    ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_THRESHOLD: int = 2
    # Abandonment rate (% of all-time sessions) at/above which an
    # "incomplete_quiz" recommendation is generated.
    ANALYTICS_RECOMMENDATION_ABANDONMENT_THRESHOLD: float = 20.0
    # Weights for the within-type severity score used ONLY to break ties
    # between recommendations that share the same type — see
    # recommendation_service.TYPE_BASE_PRIORITY for the coarser, type-level
    # ordering that actually decides which types outrank which.
    ANALYTICS_RECOMMENDATION_ACCURACY_WEIGHT: float = 1.0
    ANALYTICS_RECOMMENDATION_ATTEMPTS_WEIGHT: float = 0.5
    ANALYTICS_RECOMMENDATION_ATTEMPTS_CAP: int = 50
    ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_WEIGHT: float = 5.0
    ANALYTICS_RECOMMENDATION_MASTERY_WEIGHT: float = 0.5
    ANALYTICS_RECOMMENDATION_RECENCY_WEIGHT: float = 0.3
    # Days until a topic's "recency" contribution to severity halves — e.g. a
    # topic last attempted this many days ago counts half as urgent (from a
    # recency standpoint) as one attempted today.
    ANALYTICS_RECOMMENDATION_RECENCY_HALF_LIFE_DAYS: float = 14.0

    # ── Pydantic v2 config ────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env in the project root
        env_file_encoding="utf-8",
        case_sensitive=True,      # Env vars are case-sensitive on Linux
        extra="ignore",           # Silently ignore unknown env vars
    )


# Singleton — import `settings` from this module everywhere
settings = Settings()
