"""
core/database.py
────────────────
Async SQLAlchemy engine and session factory for Neon PostgreSQL.
Using asyncpg as the driver gives us true non-blocking DB I/O inside FastAPI's
async event loop — critical for performance under concurrent requests.

All DB interaction in the app flows through the `get_db` dependency injected
by FastAPI into route handlers.
"""
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


# ── Engine ────────────────────────────────────────────────────────────────────
database_url = settings.DATABASE_URL
database_url_obj = make_url(database_url)

engine_kwargs = {
    "future": True,              # Use SQLAlchemy 2.x style
    "echo": (settings.ENVIRONMENT == "development"),  # Log SQL only in dev
    "pool_pre_ping": True,       # Health-check connections before use
}

# Neon requires SSL; asyncpg respects the `sslmode=require` query param in the URL.
# Local development can fall back to SQLite, which uses simpler pool settings.
if database_url_obj.drivername.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["connect_args"] = {"ssl": True}
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 10

engine: AsyncEngine = create_async_engine(database_url, **engine_kwargs)

# ── Session Factory ────────────────────────────────────────────────────────────
# `async_sessionmaker` is the SQLAlchemy 2.x replacement for the old
# `sessionmaker(class_=AsyncSession, ...)` pattern.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Don't expire objects after commit (needed for async)
    autoflush=False,
    autocommit=False,
)


# ── Declarative Base ──────────────────────────────────────────────────────────
# All ORM models inherit from this. Placing it here avoids circular imports.
class Base(DeclarativeBase):
    pass


# ── FastAPI Dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncSession:  # type: ignore[override]
    """
    Yields a database session per request.
    The session is automatically closed (and rolled back on error) when the
    request context exits, thanks to the `async with` context manager.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
