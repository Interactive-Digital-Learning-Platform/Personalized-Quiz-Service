from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Personalized Quiz Service"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str
    # Per-process Postgres connection pool (ignored for the SQLite dev/test
    # fallback -- see database.py).
    # IMPORTANT: multiply (DB_POOL_SIZE + DB_MAX_OVERFLOW) by the number of
    # uvicorn workers/processes to get the real total connection count this
    # service can open against the database -- keep that under whatever
    # ceiling your DB provider actually enforces (e.g. Neon's plan-level
    # connection limit).
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-70b-8192"
    GROQ_MAX_RETRIES: int = 2
    GROQ_RETRY_BASE_DELAY_SECONDS: float = 1.0
    # Hard ceiling on Groq API calls for one POST /quiz/generate request,
    # shared across every difficulty tier a challenge-zone quiz splits into
    # (each tier would otherwise run its own dedup-retry loop independently,
    # multiplying calls per tier and risking 429s on a single user action).
    GROQ_MAX_GENERATION_CALLS_PER_REQUEST: int = 4

    # Question pool: quiz generation reads pre-generated questions for a
    # (subject, difficulty) bucket first and only calls Groq synchronously
    # for whatever the pool doesn't cover (see
    # quiz_service._fetch_pool_questions / _maybe_replenish_pool). A bucket
    # below POOL_MIN_SIZE triggers a fire-and-forget background top-up back
    # up to POOL_TARGET_SIZE, capped at POOL_TOPUP_MAX_BATCH questions per
    # Groq call (mirrors GROQ_MAX_GENERATION_CALLS_PER_REQUEST's per-call
    # sizing elsewhere).
    POOL_MIN_SIZE: int = 20
    POOL_TARGET_SIZE: int = 60
    POOL_TOPUP_MAX_BATCH: int = 20

    CLERK_JWKS_URL: str
    CLERK_ISSUER: str
    CLERK_AUDIENCE: str | None = None
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

    # These five mastery weights should add up to 1.0 — nothing enforces
    # that automatically, so keep it in mind if you tweak them.
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

    # ── Bayesian Knowledge Tracing ────────────────────────────────────────
    # Per-skill P(know) signal (SkillBKTState), independent of both the
    # ANALYTICS_MASTERY_* display score and the ADAPTIVE_MASTERY_* system
    # that drives DIFFICULTY selection (BKT never touches difficulty) — see
    # app/services/bkt_service.py. Standard textbook defaults for the BKT
    # parameters themselves; not per-skill-calibrated.
    BKT_P_INIT: float = 0.30
    BKT_P_TRANSIT: float = 0.20
    BKT_P_SLIP: float = 0.10
    BKT_P_GUESS: float = 0.20
    BKT_MASTERED_THRESHOLD: float = 0.80
    BKT_LEARNING_THRESHOLD: float = 0.30
    # How much BKT's p_know nudges automatic quiz generation's weak-lesson
    # targeting (see difficulty_mastery_engine.blend_lesson_weakness_scores),
    # blended with CEWM's LessonMastery.mastery_score. 0.0 = pure CEWM
    # (today's behavior, instant rollback); 1.0 = pure BKT. Kept low by
    # default since CEWM is tuned against real usage and BKT's own
    # parameters above aren't yet.
    BKT_LESSON_TARGETING_WEIGHT: float = 0.30

    # Same deal — these four growth weights should add up to 1.0.
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
    # Drives actual difficulty selection (SubjectMastery/LessonMastery),
    # distinct from the ANALYTICS_MASTERY_* weights above which only affect
    # the read-only mastery_score/mastery_level shown in GET /analytics/me.

    # Quiz evidence formula — must add up to 1.0.
    ADAPTIVE_MASTERY_ACCURACY_WEIGHT: float = 0.75
    ADAPTIVE_MASTERY_DIFFICULTY_WEIGHT: float = 0.10
    ADAPTIVE_MASTERY_RETENTION_WEIGHT: float = 0.10
    ADAPTIVE_MASTERY_COMPLETION_WEIGHT: float = 0.05

    # Difficulty evidence multipliers — kept close to 1.0 so correctness
    # stays the dominant signal; harder-question correctness counts
    # slightly more, easier-question correctness slightly less.
    ADAPTIVE_MASTERY_EASY_DIFFICULTY_MULTIPLIER: float = 0.95
    ADAPTIVE_MASTERY_MEDIUM_DIFFICULTY_MULTIPLIER: float = 1.00
    ADAPTIVE_MASTERY_HARD_DIFFICULTY_MULTIPLIER: float = 1.05

    # Retention evidence — weight applied to a repeated question's
    # correctness based on the gap since it was last seen. Longer gaps
    # answered correctly are stronger evidence of real retention.
    ADAPTIVE_MASTERY_RETENTION_UNDER_1_DAY_WEIGHT: float = 0.3
    ADAPTIVE_MASTERY_RETENTION_1_TO_3_DAYS_WEIGHT: float = 0.6
    ADAPTIVE_MASTERY_RETENTION_4_TO_14_DAYS_WEIGHT: float = 0.85
    ADAPTIVE_MASTERY_RETENTION_OVER_14_DAYS_WEIGHT: float = 1.0
    # Used when a student has no repeated-question history yet to score
    # retention from — a neutral-positive placeholder, not a penalty.
    ADAPTIVE_MASTERY_RETENTION_FALLBACK_SCORE: float = 70.0

    # Gradual update — how much a single quiz's evidence can move mastery.
    ADAPTIVE_MASTERY_OLD_MASTERY_WEIGHT: float = 0.75
    ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT: float = 0.25
    # Below this evidence_count, new evidence is blended in more heavily
    # (up to MAX_NEW_EVIDENCE_WEIGHT at evidence_count=0) since a new
    # student has no track record yet to protect against one bad/lucky quiz.
    ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD: int = 10
    ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT: float = 0.5

    # Confidence — the system's confidence in its own mastery estimate,
    # piecewise-linear in evidence_count between these breakpoints.
    ADAPTIVE_MASTERY_CONFIDENCE_LOW_EVIDENCE: int = 5
    ADAPTIVE_MASTERY_CONFIDENCE_LOW_SCORE: float = 35.0
    ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_EVIDENCE: int = 15
    ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_SCORE: float = 70.0
    ADAPTIVE_MASTERY_CONFIDENCE_HIGH_EVIDENCE: int = 30
    ADAPTIVE_MASTERY_CONFIDENCE_HIGH_SCORE: float = 100.0

    # Fluency — response time relative to a fixed per-difficulty baseline.
    # Fixed rather than per-student-adaptive baselines, to stay simple and
    # explainable and to work correctly for brand-new students.
    ADAPTIVE_MASTERY_FLUENCY_EASY_BASELINE_SECONDS: float = 15.0
    ADAPTIVE_MASTERY_FLUENCY_MEDIUM_BASELINE_SECONDS: float = 25.0
    ADAPTIVE_MASTERY_FLUENCY_HARD_BASELINE_SECONDS: float = 40.0
    ADAPTIVE_MASTERY_FLUENCY_MIN_SCORE: float = 20.0
    ADAPTIVE_MASTERY_FLUENCY_MAX_SCORE: float = 100.0
    # Caps baseline/actual so a near-zero response time can't blow the
    # ratio up arbitrarily.
    ADAPTIVE_MASTERY_FLUENCY_RATIO_CAP: float = 2.0
    # Fast-but-wrong answers are capped well below a merely-average correct
    # answer, so guessing quickly never reads as fluency.
    ADAPTIVE_MASTERY_FLUENCY_INCORRECT_CAP: float = 40.0

    # Recency weighting for historical/trend analysis.
    ADAPTIVE_MASTERY_RECENCY_0_TO_7_DAYS: float = 1.00
    ADAPTIVE_MASTERY_RECENCY_8_TO_14_DAYS: float = 0.90
    ADAPTIVE_MASTERY_RECENCY_15_TO_30_DAYS: float = 0.75
    ADAPTIVE_MASTERY_RECENCY_31_TO_60_DAYS: float = 0.60
    ADAPTIVE_MASTERY_RECENCY_OVER_60_DAYS: float = 0.40

    # Trend detection — recent N quizzes vs. previous N quizzes.
    ADAPTIVE_MASTERY_TREND_WINDOW_SIZE: int = 3
    ADAPTIVE_MASTERY_TREND_IMPROVING_THRESHOLD: float = 5.0
    ADAPTIVE_MASTERY_TREND_DECLINING_THRESHOLD: float = -5.0

    # Difficulty thresholds with hysteresis — promote/demote thresholds are
    # deliberately different so mastery hovering near one boundary doesn't
    # bounce the difficulty back and forth every quiz.
    ADAPTIVE_MASTERY_EASY_TO_MEDIUM_THRESHOLD: float = 65.0
    ADAPTIVE_MASTERY_MEDIUM_TO_EASY_THRESHOLD: float = 40.0
    ADAPTIVE_MASTERY_MEDIUM_TO_HARD_THRESHOLD: float = 82.0
    ADAPTIVE_MASTERY_HARD_TO_MEDIUM_THRESHOLD: float = 60.0

    # Minimum evidence required before a promotion is allowed, even if the
    # mastery score alone would qualify — protects against a lucky streak
    # on a handful of questions promoting a student prematurely.
    ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_EVIDENCE_COUNT: int = 15
    ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_QUALIFYING_COMPLETIONS: int = 2
    ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_EVIDENCE_COUNT: int = 25
    ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_QUALIFYING_COMPLETIONS: int = 3
    # A completed quiz must have at least this many answered questions to
    # count as a "qualifying completion" towards promotion — stops a
    # 100%-on-2-questions result from carrying as much weight as a full quiz.
    ADAPTIVE_MASTERY_QUALIFYING_COMPLETION_MIN_QUESTIONS: int = 3
    # How many recent qualifying quizzes to inspect when counting qualifying
    # completions / weak results for a difficulty-transition decision.
    ADAPTIVE_MASTERY_TRANSITION_LOOKBACK_QUIZZES: int = 10

    # Demotion normally needs the mastery score to cross the threshold AND
    # at least this many weak/declining results, to avoid over-reacting to
    # one bad quiz.
    ADAPTIVE_MASTERY_DEMOTION_MIN_WEAK_RESULTS: int = 2
    # Emergency demotion bypasses the "N weak results" requirement when
    # mastery has fallen far enough that waiting would leave a struggling
    # student stuck on questions well above their level.
    ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_THRESHOLD: float = 35.0
    ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_EVIDENCE_COUNT: int = 10
    ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_CONFIDENCE: float = 40.0

    # Subject mastery_score = blend of the subject's own direct quiz
    # evidence and the log-capped roll-up of its lessons' mastery scores
    # (once lesson data exists) — weighted towards the roll-up so lesson
    # mastery carries more say than a bare subject-wide accuracy average.
    ADAPTIVE_MASTERY_SUBJECT_LESSON_ROLLUP_WEIGHT: float = 0.60
    ADAPTIVE_MASTERY_SUBJECT_DIRECT_EVIDENCE_WEIGHT: float = 0.40

    # Curriculum-aware coverage in the roll-up above (grades with curriculum
    # data only — see curriculum_service.py): a curriculum lesson the
    # student hasn't attempted yet folds into the roll-up at this neutral
    # default score, so grinding one lesson can't drive the whole subject's
    # mastery up without ever touching the others. Weight kept modest
    # (~log1p(1)) so real practice across many lessons still dominates.
    ADAPTIVE_MASTERY_UNTESTED_LESSON_DEFAULT: float = 50.0
    ADAPTIVE_MASTERY_UNTESTED_LESSON_WEIGHT: float = 1.0

    # ── RAG (curriculum retrieval for quiz grounding) ────────────────────
    # Grounds Groq's generated questions in real curriculum excerpts pulled
    # from Qdrant, mirroring AI-Learning-Assistant-Service's embed/retrieve/
    # rerank pattern — see app/services/rag_service.py. Pure enhancement:
    # every rag_service call degrades to empty results on any failure/
    # timeout, never blocking or failing quiz generation itself.
    RAG_ENABLED: bool = True
    QDRANT_URL: str = ""
    # Reuses PDF-Ingestion-Backend-Service's existing pdf_knowledge_base
    # collection (already holds the ingested Grade 10/11 textbook PDFs)
    # rather than a separate quiz-only collection. Points there were
    # backfilled with `grade` (int) / `subject` (str) payload fields by
    # qdrant-migrations/003_create_quiz_knowledge_base_collection.py so
    # retrieval_service.py's grade+subject filter has something to match —
    # see that migration for the filename → (grade, subject) mapping.
    QUIZ_KNOWLEDGE_COLLECTION: str = "pdf_knowledge_base"
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 768
    RAG_MAX_QUERY_TOKENS: int = 8192
    RAG_TOP_K_CHUNKS: int = 5
    RAG_SCORE_THRESHOLD: float = 0.6
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_OVERFETCH: int = 4
    # Every rag_service call is bounded by this timeout and falls back to
    # empty results rather than ever raising into quiz_service/groq_service.
    RAG_TIMEOUT_SECONDS: float = 8.0
    # Bounds how many per-lesson retrieval queries one request can fan out
    # into (each is an embed + Qdrant search + rerank) — a single-subject
    # quiz caps at RAG_MAX_LESSON_QUERIES; a shuffle quiz spanning several
    # subjects shares one larger combined cap across the whole batch so a
    # many-subject shuffle can't fan out unboundedly.
    RAG_MAX_LESSON_QUERIES: int = 8
    RAG_MAX_LESSON_QUERIES_SHUFFLE: int = 12
    RAG_MAX_CONCURRENT_QUERIES: int = 4
    RAG_SNIPPETS_LESSON_PINNED: int = 4
    RAG_SNIPPETS_PER_LESSON: int = 2
    RAG_MAX_SNIPPET_CHARS: int = 700

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Required fields (DATABASE_URL, GROQ_API_KEY, CLERK_JWKS_URL, CLERK_ISSUER)
# come from the environment / .env at runtime via pydantic-settings, not
# constructor arguments -- static analysis can't see that, hence the ignore.
settings = Settings()  # type: ignore[call-arg]
