# Personalized Quiz Service

Backend for **Nous**, an AI learning platform for Sri Lankan Grade 10–11 students. This service generates personalised
practice quizzes grounded in the national textbooks, adapts difficulty to each student, tracks what they have
mastered, and turns their results into analytics and study advice.

Built with **FastAPI** (async), **PostgreSQL** (Neon) via SQLAlchemy 2 + Alembic, **Groq** for question generation,
and **Qdrant** for textbook retrieval. Users sign in with **Clerk**.

---

## Features

**Quiz generation**
- Quizzes for a subject, a single lesson, or a mixed "AI picks" set, for Grade 10 and 11 (Mathematics, Science,
  Geography, History, English, ICT, and Health & Physical Education for Grade 11), following the syllabus in
  `app/data/curriculum/`.
- Questions come from three sources, in order:
  1. **AI + textbook (RAG):** Groq writes the questions from excerpts of the actual textbook, retrieved from Qdrant.
  2. **AI only:** if no excerpts are found, Groq generates from the syllabus alone.
  3. **Question pool:** previously generated questions the student hasn't seen yet, used to fill any shortfall
     (the pool is topped up in the background).
- Every question is validated (four distinct options, a valid answer, no duplicates or near-duplicates) and
  fingerprinted so a student isn't given the same question twice.

**Textbook retrieval (RAG)**
- Hybrid search in Qdrant: dense vectors (`nomic-embed-text-v1.5`) + BM25 keyword vectors, fused with RRF.
- Results are filtered to the student's grade and subject, matched to the lesson's section headings, and reranked
  with a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
- Optional provenance: each AI question can record the textbook page it came from.

**Adaptive learning**
- **Adaptive difficulty** — a Continuous Evidence-Weighted Mastery score (0–100) per student, subject and lesson
  decides between easy, medium and hard; see [`docs/adaptive-mastery-system.md`](docs/adaptive-mastery-system.md).
- **Bayesian Knowledge Tracing** — a per-lesson probability that the student knows the skill
  (mastered / learning / not started), used to steer quizzes towards weak lessons.
- **Challenge zone** and **retakes** of previous quizzes.

**Quiz sessions**
- Answers are checked on the server; correct answers are never sent before the quiz is submitted.
- Progress autosave, timed submission (`submit-timeout`), session history and deletion.

**Analytics and coaching**
- Per-student analytics: accuracy and score trends, subject and lesson mastery, topic performance, response times,
  daily activity, growth over time and repeated-question tracking ([`docs/analytics-data-report.md`](docs/analytics-data-report.md)).
- Personalised recommendations ("practise next"), AI study feedback and motivational quotes.

**Operations**
- Clerk JWT authentication (JWKS, issuer, optional audience / authorised-party checks).
- Per-user rate limits on the AI endpoints.
- AI generation telemetry (latency, success, retries, fallbacks, errors) with an admin endpoint.

---

## API

All routes are under `/api/v1` and need a Clerk session token (`Authorization: Bearer <token>`), except `/health`.
Interactive docs are at `/docs` when `ENVIRONMENT=development`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/user/sync` | Create / update the user from the Clerk session |
| `GET` | `/user/me` | Current user |
| `GET` | `/quiz/curriculum` | Subjects and lessons for a grade |
| `POST` | `/quiz/generate` | Generate a quiz |
| `GET` | `/quiz/sessions` | Quiz history |
| `GET` / `DELETE` | `/quiz/sessions/{session_id}` | One quiz: details / delete |
| `POST` | `/quiz/sessions/{session_id}/retake` | Retake a quiz |
| `POST` | `/quiz/progress/save` | Autosave answers in progress |
| `POST` | `/quiz/submit` | Submit and grade a quiz |
| `POST` | `/quiz/submit-timeout` | Submit when the timer runs out |
| `POST` | `/quiz/motivational-quote` | Motivational quote for the results screen |
| `GET` | `/analytics/me` | Full analytics for the student |
| `GET` | `/analytics/daily` | Daily activity |
| `GET` / `POST` | `/analytics/feedback` | AI study feedback |
| `GET` | `/analytics/system/ai-generation` | AI generation telemetry (admins in `ADMIN_CLERK_IDS`, or anyone in development) |

The mobile app reaches this service through the API gateway at `/api/quiz/...`.

---

## Project structure

```
app/
  api/routes/     quiz.py, analytics.py, user.py — HTTP endpoints
  core/           settings (config.py), database, Clerk auth, rate limiting
  models/         SQLAlchemy models (users, questions, sessions, attempts, mastery, BKT, telemetry, ...)
  schemas/        Pydantic request/response models
  services/       quiz generation, Groq, RAG / retrieval, difficulty, BKT, analytics, recommendations
  data/curriculum/  Grade 10/11 syllabus: subjects, lessons, subtopics
  scripts/        one-off backfills for mastery / BKT
alembic/          database migrations
scripts/          model download for the Docker image, retrieval evaluation
tests/            pytest suite
docs/             design notes for adaptive mastery and analytics
```

---

## Running locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # then fill it in (see below)
uv sync                       # creates .venv from uv.lock
uv run alembic upgrade head   # create / update the database tables
uv run uvicorn app.main:app --reload
```

Set `ENVIRONMENT=development` in `.env` for `/docs`, open CORS and debug logging. The first start downloads the
embedding and rerank models (~650 MB) from Hugging Face.

### Configuration

All settings live in `app/core/config.py`; `.env.example` lists the ones a deployment needs:

| Variable | What it is |
|---|---|
| `DATABASE_URL` | PostgreSQL, asyncpg format (`postgresql+asyncpg://...`) |
| `CLERK_JWKS_URL`, `CLERK_ISSUER` | Clerk instance used to verify tokens (`CLERK_AUDIENCE`, `CLERK_AUTHORIZED_PARTIES` optional) |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq credentials and model |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QUIZ_KNOWLEDGE_COLLECTION` | Textbook knowledge base (`RAG_ENABLED=false` turns retrieval off) |
| `ADMIN_CLERK_IDS` | Clerk user ids allowed to use admin endpoints |

The textbook collection is built by the separate **PDF-Ingestion-Backend-Service**.

### Docker

```bash
docker build -t personalized-quiz-service .
docker run --env-file .env -p 8000:8000 personalized-quiz-service
```

The image bakes the RAG models in at build time (no download on start) and runs `alembic upgrade head` on start
unless `RUN_MIGRATIONS=false`.

---

## Tests and evaluation

```bash
uv run pytest
```

The suite uses an in-memory SQLite database with Groq and Qdrant mocked, so it needs no network. It covers quiz
generation and integrity, RAG, BKT, adaptive difficulty, analytics and security.

`scripts/eval_retrieval.py` checks textbook retrieval against a labelled set of lessons
(`tests/data/retrieval_eval_set.json`).

---

## Deployment

Pushing to `main` builds the image, pushes it to Azure Container Registry and deploys it to Azure Container Apps
(`.github/workflows/deploy.yml`). Changes to Markdown files, `docs/` and `tests/` don't trigger a deploy. Production
runs with `RUN_MIGRATIONS=false`: apply Alembic migrations by hand before merging changes that need them.

---

## Related services

- **Quiz-Battle-Service** — online 1v1 battles (matchmaking, ratings, leaderboards, live WebSocket matches). Separate
  service that shares this service's database and is reached through the gateway at `/api/battle/...`.
- **PDF-Ingestion-Backend-Service** — turns the textbooks into the Qdrant knowledge base used here.
- **Mobile-Application** — the Expo / React Native app (Nous).
- **Deployment-Infrastructure** — the API gateway.
