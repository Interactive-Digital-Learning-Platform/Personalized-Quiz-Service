# Analytics Data Report — Personalized Quiz Service

Generated: 2026-08-28. Covers the quiz-analytics data model and its current live contents (Neon Postgres, `neondb`).

## 1. Scope

Two parts:
- **Architecture** — every field the service collects or computes about user quiz behavior, and where it lives in code.
- **Empirical snapshot** — real row counts and aggregate statistics pulled directly from the live database at report time.

Battle/competitive mode is included where it shares data-relevance, but is called out separately since its application source is not present in the current working tree (see §7.7).

## 2. Tech stack

FastAPI + SQLAlchemy 2.x async ORM over PostgreSQL (`asyncpg`), Alembic migrations, Pydantic v2 schemas. Auth via Clerk (JWT `sub` = `clerk_id`). AI question generation via Groq (LLaMA 3).

## 3. Raw data model

### `QuizSession` (`app/models/quiz_session.py`)
One row per generated quiz. `id`, `user_id`, `subject`, `lesson`, `difficulty`, `question_count`, `questions_snapshot` (JSON), `score`, `accuracy`, `total_time`, `created_at`, `deleted_at` (soft delete — kept for analytics), `is_retake`, `retake_of_session_id`.

### `QuestionAttempt` (`app/models/quiz_session.py`)
One row per answered question. `id`, `session_id`, `question_id`, `selected_answer`, `correct` (nullable = unanswered), `response_time` (client-reported seconds). **No timestamp column** — all attempts in a session are inserted together at submission time (`app/services/quiz_service.py:1191`), so there is no way to reconstruct real per-question timing or an authoritative within-session order.

### `Question` (`app/models/question.py`)
Shared pool, not per-user. `id`, `question`, `options` (JSON), `correct_answer`, `subject`, `lesson`, `difficulty`, `question_fingerprint` (hash of subject+lesson+text — the only stable identity across repeats, since every AI generation writes a fresh row rather than reusing one).

### `QuizCompletion` (`app/models/quiz_tracking.py`)
One row per graded session. `ended_by` ("submitted"/"timeout"), `total_time`, `score`, `accuracy`, `correct_count`, `total_questions`, `lesson_time_breakdown` (JSON), `lesson_accuracy_breakdown` (JSON), `repeated_lessons_right/wrong`, `repeated_correct_count`, `repeated_wrong_count`, `completed_at`.

### `QuizProgressSnapshot` (`app/models/quiz_tracking.py`)
In-progress save state: `remaining_time`, `answered_count`, `repeated_question_ids`, `weak_lessons_hint`, `draft_answers` (JSON), `saved_at`.

### `Analytics` (`app/models/analytics.py`)
Rolling per-(user, subject) summary: `accuracy`, `avg_response_time`, `weak_topic`, `updated_at`.

### `SubjectMastery` / `LessonMastery` (`app/models/subject_mastery.py`, `lesson_mastery.py`)
Adaptive-difficulty state, identical field set at subject- and lesson-scope: `difficulty`, `last_accuracy`, `consecutive_strong/weak`, `mastery_score`, `fluency_score`, `confidence_score`, `evidence_count`, `recent_accuracy`, `previous_accuracy`, `trend_score`, `trend_label`, `retention_score`, `last_mastery_update`.

### `AIGenerationEvent` (`app/models/ai_generation_event.py`)
Telemetry for the generation pipeline: `requested/generated_question_count`, `provider`, `model_name`, `success`, `used_cache_fallback`, `retry_count`, `duplicate_count`, `invalid_question_count`, `latency_ms`, `error_category`.

### `User` (`app/models/user.py`)
`id`, `clerk_id`, `username`, `created_at`.

## 4. Derived / computed analytics layer

Computed on read, not stored (`app/services/analytics/*`), exposed via `GET /analytics/me`:

- **Timing**: median/fastest/slowest/correct-vs-incorrect response time, std-dev, `answering_behavior` label.
- **Completion**: completed/incomplete/timed-out/abandoned session counts, completion rate, timeout rate, avg session duration and questions/session.
- **Performance trend**: current vs. previous period accuracy, `accuracy_change`, trend label.
- **Repeated-mistake analytics**: repeated question/correct/incorrect counts, mistake-correction rate — keyed on `question_fingerprint`.
- **Per-topic analytics**: attempted/correct/incorrect, accuracy, status (weak/moderate/strong/insufficient_data), plus timing/trend/mastery sub-fields per lesson.
- **Difficulty-level performance**: accuracy/attempts/avg response time per tier, promotion/demotion hysteresis state (`app/services/difficulty_service.py`).
- **Mastery score** (analytics-only, distinct from adaptive `SubjectMastery`): accuracy/recent-performance/difficulty/retention/consistency components.
- **Growth score**: effort/consistency/improvement components → `growth_score`, `growth_level`.
- **Recommendations**: rule-based, priority-ordered (`app/services/recommendation_service.py`) — see prior evaluation, no impression/outcome logging exists yet.
- **Fluency** and **retention evidence**: computed per-question/per-repeat in `difficulty_mastery_engine.py`, feed `SubjectMastery`/`LessonMastery`.

## 5. API surface

`GET /analytics/me`, `GET/POST /analytics/feedback` (Groq coaching text), `GET /analytics/system/ai-generation` (admin), and the raw-data writers `POST /quiz/generate`, `POST /quiz/submit`, `GET /quiz/sessions[/​{id}]`.

## 6. Shuffle-mode data caveat

Shuffle sessions collapse `QuizSession.subject`/`.lesson`/`.difficulty` to the sentinel string `"Mixed"` (`app/services/quiz_service.py:1029-1047`) even though the session spans multiple real subjects/difficulties. The adaptive-mastery pipeline already handles this correctly — `GradedAnswer.difficulty` is sourced from `question.difficulty`, not `session.difficulty` (`quiz_service.py:1204`), and subject-level analytics reads `Question.subject` per attempt rather than `QuizSession.subject` (`analytics_service.py`). **Anyone building new analysis directly against `quiz_sessions` must join through `question_attempts → questions` for true subject/difficulty** — the session-level columns are not reliable once shuffle mode is used, which per the live data below is the single most common session type.

## 7. Empirical snapshot (live database, 2026-08-28)

### 7.1 Row counts

| Table | Rows |
|---|---|
| users | 10 |
| quiz_sessions | 59 |
| question_attempts | 368 |
| questions | 519 (517 unique fingerprints) |
| quiz_completions | 37 |
| quiz_progress_snapshots | 325 |
| analytics | 10 |
| subject_mastery | 10 |
| lesson_mastery | 259 |
| ai_generation_events | 68 |

Data spans 2026-08-13 → 2026-08-27 (14 days).

### 7.2 Usage is heavily concentrated in one account

Only 4 of 10 users have any quiz activity. User `1` alone accounts for 46 of 57 non-retake sessions (81%) and 318 of 368 attempts (86%). Users `2`/`9`/`10` contribute the remainder (3, 10, 40 attempts respectively). **This is effectively single-user data, not a population** — any per-user statistic other than user 1's is currently near-meaningless, and any modeling approach that relies on pooling across many users (e.g. population-level BKT) has almost no population to pool from yet.

### 7.3 Completion and abandonment

57 non-retake sessions, only 37 completions (65%), 2 retakes, 20 soft-deleted. Of completions: 33 "submitted" (avg accuracy 36.7%), 4 "timeout" (avg accuracy 0.0% — all four timed out having answered nothing scored). No unanswered (`correct IS NULL`) attempt rows exist despite this — consistent with timeouts here being early abandonment before any question was graded, not partial completion.

### 7.4 Accuracy and response time

Session accuracy: mean 32.7%, min 0%, max 90%, std-dev 27.6 — wide spread, skewed low. Response time on answered questions: mean 14.1s, max 786s (a clear outlier — either a long real pause or an app-backgrounding artifact; `response_time` is client-reported with no server-side timestamp to cross-check, per §3).

### 7.5 Subject/difficulty mix

Sessions by (subject, difficulty): `Mixed/easy` ×23, `Mathematics/Mixed` ×15, `Programming/Mixed` ×6, `Science/Mixed` ×6, `Geography/Mixed` ×4, `English/Mixed` ×3, `Mixed/Mixed` ×1, `History/Mixed` ×1. **40 of 59 sessions (68%) carry a "Mixed" sentinel in subject and/or difficulty** — i.e. shuffle mode is the dominant usage pattern in the real data so far, reinforcing that §6's join-through-`questions` requirement isn't an edge case, it's the majority case.

### 7.6 Mastery/difficulty state

All 10 `subject_mastery` rows and all 259 `lesson_mastery` rows show `difficulty = "easy"` — **no user has been promoted past the easy tier in any subject or lesson yet.** `lesson_mastery.evidence_count` averages 1.38 (min 1, max 7) — far below the engine's own "high confidence" bar of 30 questions (`difficulty_service.py`, confidence curve). Subject-level mastery scores range 32.9–52.6 (all near the 50.0 cold-start default), and 9 of 10 subject rows report `trend_label = "insufficient_data"`. **This directly confirms the volume caveats raised earlier**: individual per-lesson evidence is far too thin for anything beyond the simplest heuristics right now, and per-user difficulty transitions haven't been exercised in real data at all — there's no promoted/demoted case to even validate the hysteresis logic against yet.

### 7.7 AI generation reliability

68 events: 44 successful-with-cache-fallback (avg latency 28.8s), 20 successful-without-fallback (avg latency 35.3s), 4 failures-with-fallback (avg latency 15.0s). No failures without a fallback available — the cache appears to be masking all outright generation failures so far.

### 7.8 Battle mode — adjacent system, not part of this component

`battle_matches` (294), `battle_participants` (588), `battle_answers` (1,288), `battle_match_questions` (2,110) contain **far more volume than the quiz-analytics tables above**, generated in just 6 days (2026-08-22 → 08-28) across only 8 distinct users (~37 matches/user in 6 days — a rate consistent with automated/dev testing traffic rather than organic use, worth confirming before treating it as real usage). Notably, the corresponding application source (`battle_match.py`, `elo_engine.py`, `matchmaking_service.py`) is **not present in the current working tree** and has no git history — only Alembic migrations and stale `.pyc` files remain. The tables and data exist; the code that produced them does not, in this checkout. Flagging this since it's the largest pool of behavioral data in the database, but it sits outside the quiz-analytics component this report covers and its provenance should be confirmed before use.

## 8. Implications for any modeling work

- **Sample size is the binding constraint**, not modeling technique — 368 graded attempts total, 86% from one user. This is far below what any of the previously discussed approaches (BKT, dropout classifiers, recommendation ranking) need to produce a trustworthy result; treat all of them as designs to validate once more organic usage accumulates, not as ready to train today.
- **No difficulty transitions have occurred yet** in real data — there's currently no ground truth to check whether `determine_difficulty_transition`'s thresholds are even reasonable.
- **Shuffle mode dominates usage (68% of sessions)** — every analysis must read subject/difficulty from `question_attempts → questions`, never from `quiz_sessions` directly.
- **`response_time` has no server-side corroboration** and includes at least one extreme outlier (786s) — worth clipping/flagging outliers before using it as a feature.
- Battle-mode tables are a much larger dataset but sit outside this component and their source code is currently missing from the repo — confirm what they represent before treating them as usable training data.
