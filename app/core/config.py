from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Personalized Quiz Service"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str

    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-70b-8192"
    GROQ_MAX_RETRIES: int = 2
    GROQ_RETRY_BASE_DELAY_SECONDS: float = 1.0
    # Ceiling on Groq calls per generate request, shared across challenge-zone tiers.
    GROQ_MAX_GENERATION_CALLS_PER_REQUEST: int = 4

    CLERK_JWKS_URL: str
    CLERK_ISSUER: str
    CLERK_AUDIENCE: Optional[str] = None
    AUTH_BYPASS: bool = False
    ADMIN_CLERK_IDS: str = ""

    ANALYTICS_ABANDONED_AFTER_HOURS: int = 24
    ANALYTICS_TOPIC_MIN_ATTEMPTS: int = 3
    ANALYTICS_MAX_TOPICS_PER_SUBJECT: int = 10

    ANALYTICS_MAX_VALID_RESPONSE_TIME_SECONDS: float = 600.0
    ANALYTICS_BEHAVIOR_MIN_ATTEMPTS: int = 5
    ANALYTICS_ACCURATE_THRESHOLD: float = 70.0
    ANALYTICS_MEDIAN_FALLBACK_SECONDS: float = 10.0
    ANALYTICS_BALANCED_TIME_RATIO: float = 0.15

    ANALYTICS_TREND_SESSION_WINDOW: int = 5
    ANALYTICS_TREND_MIN_ATTEMPTS_PER_PERIOD: int = 1
    ANALYTICS_TREND_CHANGE_THRESHOLD: float = 5.0

    # These five mastery weights should add up to 1.0.
    ANALYTICS_MASTERY_ACCURACY_WEIGHT: float = 0.35
    ANALYTICS_MASTERY_RECENT_PERFORMANCE_WEIGHT: float = 0.25
    ANALYTICS_MASTERY_DIFFICULTY_WEIGHT: float = 0.15
    ANALYTICS_MASTERY_RETENTION_WEIGHT: float = 0.15
    ANALYTICS_MASTERY_CONSISTENCY_WEIGHT: float = 0.10
    ANALYTICS_MASTERY_MIN_ATTEMPTS: int = 5
    ANALYTICS_MASTERY_NEUTRAL_SCORE: float = 50.0
    ANALYTICS_MASTERY_CONSISTENCY_MIN_SESSIONS: int = 3
    ANALYTICS_MASTERY_DEVELOPING_THRESHOLD: float = 40.0
    ANALYTICS_MASTERY_PROFICIENT_THRESHOLD: float = 70.0
    ANALYTICS_MASTERY_ADVANCED_THRESHOLD: float = 85.0

    # These four growth weights should add up to 1.0.
    ANALYTICS_GROWTH_IMPROVEMENT_WEIGHT: float = 0.40
    ANALYTICS_GROWTH_CONSISTENCY_WEIGHT: float = 0.25
    ANALYTICS_GROWTH_EFFORT_WEIGHT: float = 0.20
    ANALYTICS_GROWTH_MASTERY_WEIGHT: float = 0.15
    ANALYTICS_GROWTH_GROWING_THRESHOLD: float = 40.0
    ANALYTICS_GROWTH_STRONG_THRESHOLD: float = 70.0
    ANALYTICS_GROWTH_EXCEPTIONAL_THRESHOLD: float = 85.0
    ANALYTICS_GROWTH_WINDOW_DAYS: int = 30
    ANALYTICS_GROWTH_MIN_ATTEMPTS_IN_WINDOW: int = 3
    ANALYTICS_GROWTH_EFFORT_TARGET_QUIZZES: int = 20
    ANALYTICS_GROWTH_EFFORT_TARGET_QUESTIONS: int = 150
    ANALYTICS_GROWTH_EFFORT_TARGET_WEAK_TOPIC_ATTEMPTS: int = 10

    ANALYTICS_RECOMMENDATION_MAX_COUNT: int = 5
    ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_THRESHOLD: int = 2
    ANALYTICS_RECOMMENDATION_ABANDONMENT_THRESHOLD: float = 20.0
    ANALYTICS_RECOMMENDATION_ACCURACY_WEIGHT: float = 1.0
    ANALYTICS_RECOMMENDATION_ATTEMPTS_WEIGHT: float = 0.5
    ANALYTICS_RECOMMENDATION_ATTEMPTS_CAP: int = 50
    ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_WEIGHT: float = 5.0
    ANALYTICS_RECOMMENDATION_MASTERY_WEIGHT: float = 0.5
    ANALYTICS_RECOMMENDATION_RECENCY_WEIGHT: float = 0.3
    ANALYTICS_RECOMMENDATION_RECENCY_HALF_LIFE_DAYS: float = 14.0

    # ── Continuous Evidence-Weighted Mastery System ──────────────────────
    # Drives actual difficulty selection -- distinct from ANALYTICS_MASTERY_*
    # above, which only affects the read-only display in GET /analytics/me.

    # Quiz evidence formula -- must add up to 1.0.
    ADAPTIVE_MASTERY_ACCURACY_WEIGHT: float = 0.75
    ADAPTIVE_MASTERY_DIFFICULTY_WEIGHT: float = 0.10
    ADAPTIVE_MASTERY_RETENTION_WEIGHT: float = 0.10
    ADAPTIVE_MASTERY_COMPLETION_WEIGHT: float = 0.05

    # Difficulty multipliers, kept close to 1.0 so accuracy stays dominant.
    ADAPTIVE_MASTERY_EASY_DIFFICULTY_MULTIPLIER: float = 0.95
    ADAPTIVE_MASTERY_MEDIUM_DIFFICULTY_MULTIPLIER: float = 1.00
    ADAPTIVE_MASTERY_HARD_DIFFICULTY_MULTIPLIER: float = 1.05

    # Retention weight by gap since a repeated question was last seen.
    ADAPTIVE_MASTERY_RETENTION_UNDER_1_DAY_WEIGHT: float = 0.3
    ADAPTIVE_MASTERY_RETENTION_1_TO_3_DAYS_WEIGHT: float = 0.6
    ADAPTIVE_MASTERY_RETENTION_4_TO_14_DAYS_WEIGHT: float = 0.85
    ADAPTIVE_MASTERY_RETENTION_OVER_14_DAYS_WEIGHT: float = 1.0
    ADAPTIVE_MASTERY_RETENTION_FALLBACK_SCORE: float = 70.0  # no repeat history yet

    # Gradual update -- how much one quiz's evidence can move mastery.
    ADAPTIVE_MASTERY_OLD_MASTERY_WEIGHT: float = 0.75
    ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT: float = 0.25
    # Below this evidence_count, new evidence blends in more heavily.
    ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD: int = 10
    ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT: float = 0.5

    # Confidence: piecewise-linear in evidence_count between these breakpoints.
    ADAPTIVE_MASTERY_CONFIDENCE_LOW_EVIDENCE: int = 5
    ADAPTIVE_MASTERY_CONFIDENCE_LOW_SCORE: float = 35.0
    ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_EVIDENCE: int = 15
    ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_SCORE: float = 70.0
    ADAPTIVE_MASTERY_CONFIDENCE_HIGH_EVIDENCE: int = 30
    ADAPTIVE_MASTERY_CONFIDENCE_HIGH_SCORE: float = 100.0

    # Fluency: response time vs. a fixed per-difficulty baseline.
    ADAPTIVE_MASTERY_FLUENCY_EASY_BASELINE_SECONDS: float = 15.0
    ADAPTIVE_MASTERY_FLUENCY_MEDIUM_BASELINE_SECONDS: float = 25.0
    ADAPTIVE_MASTERY_FLUENCY_HARD_BASELINE_SECONDS: float = 40.0
    ADAPTIVE_MASTERY_FLUENCY_MIN_SCORE: float = 20.0
    ADAPTIVE_MASTERY_FLUENCY_MAX_SCORE: float = 100.0
    ADAPTIVE_MASTERY_FLUENCY_RATIO_CAP: float = 2.0  # caps near-zero response times
    ADAPTIVE_MASTERY_FLUENCY_INCORRECT_CAP: float = 40.0  # fast-wrong != fluent

    # Recency weighting for historical/trend analysis.
    ADAPTIVE_MASTERY_RECENCY_0_TO_7_DAYS: float = 1.00
    ADAPTIVE_MASTERY_RECENCY_8_TO_14_DAYS: float = 0.90
    ADAPTIVE_MASTERY_RECENCY_15_TO_30_DAYS: float = 0.75
    ADAPTIVE_MASTERY_RECENCY_31_TO_60_DAYS: float = 0.60
    ADAPTIVE_MASTERY_RECENCY_OVER_60_DAYS: float = 0.40

    # Trend detection -- recent N quizzes vs. previous N quizzes.
    ADAPTIVE_MASTERY_TREND_WINDOW_SIZE: int = 3
    ADAPTIVE_MASTERY_TREND_IMPROVING_THRESHOLD: float = 5.0
    ADAPTIVE_MASTERY_TREND_DECLINING_THRESHOLD: float = -5.0

    # Hysteresis: separate promote/demote thresholds prevent bouncing.
    ADAPTIVE_MASTERY_EASY_TO_MEDIUM_THRESHOLD: float = 65.0
    ADAPTIVE_MASTERY_MEDIUM_TO_EASY_THRESHOLD: float = 40.0
    ADAPTIVE_MASTERY_MEDIUM_TO_HARD_THRESHOLD: float = 82.0
    ADAPTIVE_MASTERY_HARD_TO_MEDIUM_THRESHOLD: float = 60.0

    # Minimum evidence required before promotion, even if mastery qualifies.
    ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_EVIDENCE_COUNT: int = 15
    ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_QUALIFYING_COMPLETIONS: int = 2
    ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_EVIDENCE_COUNT: int = 25
    ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_QUALIFYING_COMPLETIONS: int = 3
    ADAPTIVE_MASTERY_QUALIFYING_COMPLETION_MIN_QUESTIONS: int = 3  # min answered to "qualify"
    ADAPTIVE_MASTERY_TRANSITION_LOOKBACK_QUIZZES: int = 10

    # Demotion needs mastery below threshold AND this many weak results.
    ADAPTIVE_MASTERY_DEMOTION_MIN_WEAK_RESULTS: int = 2
    # Emergency demotion skips the weak-results requirement below this mastery.
    ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_THRESHOLD: float = 35.0
    ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_EVIDENCE_COUNT: int = 10
    ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_CONFIDENCE: float = 40.0

    # Subject mastery = blend of direct quiz evidence and lesson roll-up,
    # weighted toward the roll-up so lesson mastery carries more say.
    ADAPTIVE_MASTERY_SUBJECT_LESSON_ROLLUP_WEIGHT: float = 0.60
    ADAPTIVE_MASTERY_SUBJECT_DIRECT_EVIDENCE_WEIGHT: float = 0.40

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
