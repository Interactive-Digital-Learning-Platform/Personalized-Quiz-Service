# Personalized Quiz Service (FastAPI)

Backend for AI-powered practice quizzes for Sri Lankan Grade 10–11 students.

Run locally (requires [uv](https://docs.astral.sh/uv/)):

```
uv sync
uv run uvicorn app.main:app --reload
```

`uv sync` creates `.venv` and installs everything pinned in `uv.lock` (add
`--no-dev` to skip test-only dependencies). To run a command inside the venv
without the `uv run` prefix, activate it the usual way:
`source .venv/bin/activate`.

Configure via environment variables (or .env):
- `DATABASE_URL` (asyncpg format)
- `GROQ_API_KEY` and `GROQ_ENDPOINT`
- `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUDIENCE` (optional)
