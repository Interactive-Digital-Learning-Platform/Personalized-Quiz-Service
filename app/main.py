"""
main.py
────────
FastAPI application entrypoint.

Responsibilities:
- Create and configure the FastAPI app instance.
- Register CORS middleware (permissive for development, lock down in production).
- Register all API routers with their prefixes.
- On startup: create all DB tables (via SQLAlchemy metadata) and log config.
- On shutdown: dispose of the DB engine connection pool cleanly.

This file stays intentionally thin — all business logic lives in services/,
all DB logic in core/database.py, and all routing in api/routes/.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analytics, quiz, user
from app.core.config import settings
from app.core.database import Base, engine

# ── Logging Setup ─────────────────────────────────────────────────────────────
# Basic logging configuration — uvicorn will also log HTTP access separately.
logging.basicConfig(
    level=logging.DEBUG if settings.ENVIRONMENT == "development" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── FastAPI App Instance ───────────────────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "AI-powered Personalised Quiz Service backend. "
        "Generates adaptive quizzes using Groq LLaMA 3, "
        "provides per-subject analytics, and delivers AI study coaching."
    ),
    version="0.5.0",   # 50% MVP
    # Disable docs in production to avoid leaking internal API structure
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)


# ── CORS Middleware ────────────────────────────────────────────────────────────
# In development, allow all origins so the React Native dev client can connect.
# In production, restrict origins to your actual frontend domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup / Shutdown Event Handlers ─────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when the server starts.

    Creates all tables defined in the SQLAlchemy models if they don't exist.
    In a production environment you would use Alembic migrations instead —
    but for an MVP, auto-creation is fast and reliable.
    """
    logger.info("Starting %s [%s]", settings.PROJECT_NAME, settings.ENVIRONMENT)
    logger.info("Groq model: %s", settings.GROQ_MODEL)

    # Import all models so their metadata is registered on Base before create_all
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        logger.info("Running create_all — creating missing tables...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables are ready.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """
    Runs once when the server shuts down.
    Disposes the async engine, closing all pooled connections gracefully.
    """
    logger.info("Shutting down — disposing DB engine...")
    await engine.dispose()
    logger.info("Shutdown complete.")


# ── API Routers ────────────────────────────────────────────────────────────────
# Each router is mounted at a versioned prefix (/api/v1) for future-proofing.
API_PREFIX = "/api/v1"

app.include_router(quiz.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(user.router, prefix=API_PREFIX)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    """
    Simple liveness probe endpoint for Docker / load-balancer health checks.
    Returns 200 OK if the server is running.
    """
    return {"status": "ok", "service": settings.PROJECT_NAME}
