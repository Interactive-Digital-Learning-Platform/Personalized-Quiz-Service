import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401 -- registers model metadata on Base
from app.api.routes import analytics, quiz, user
from app.core.config import settings
from app.core.database import Base, engine
from app.services import rag_service

logging.basicConfig(
    level=logging.DEBUG if settings.ENVIRONMENT == "development" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "AI-powered Personalised Quiz Service backend. "
        "Generates adaptive quizzes using Groq LLaMA 3, "
        "provides per-subject analytics, and delivers AI study coaching."
    ),
    version="0.5.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting %s [%s]", settings.PROJECT_NAME, settings.ENVIRONMENT)
    logger.info("Groq model: %s", settings.GROQ_MODEL)

    async with engine.begin() as conn:
        # create_all only creates tables that don't exist yet — it never adds
        # a column to a table that's already there. If you add a field to a
        # model, you still need a real migration (see alembic/) for it to
        # show up on a database that's already been through this once.
        logger.info("Running create_all — creating missing tables...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables are ready.")

    # Best-effort: pre-loads the embedding/rerank models so the first quiz
    # generation request doesn't pay that cost. Never blocks startup -- if
    # Qdrant/models are unreachable now, rag_service's own timeouts and
    # empty-result fallbacks mean quiz generation still works without RAG
    # grounding, and a later request will retry the lazy init itself.
    try:
        rag_service.warm_up()
    except Exception:
        logger.exception("RAG warm-up failed — continuing without it")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutting down — disposing DB engine...")
    await engine.dispose()
    logger.info("Shutdown complete.")


API_PREFIX = "/api/v1"

app.include_router(quiz.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(user.router, prefix=API_PREFIX)


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    return {"status": "ok", "service": settings.PROJECT_NAME}
