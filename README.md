# Personalized Quiz Service (FastAPI)

Backend for AI-powered practice quizzes for Sri Lankan Grade 10–11 students.

Run locally:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Configure via environment variables (or .env):
- `DATABASE_URL` (asyncpg format)
- `GROQ_API_KEY` and `GROQ_ENDPOINT`
- `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUDIENCE` (optional)
