from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

database_url = settings.DATABASE_URL
database_url_obj = make_url(database_url)

engine_kwargs = {
    "future": True,
    "echo": (settings.ENVIRONMENT == "development"),
    "pool_pre_ping": True,
}

if database_url_obj.drivername.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["connect_args"] = {"ssl": True}
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 10

engine: AsyncEngine = create_async_engine(database_url, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:  # type: ignore[override]
    # FastAPI calls this per-request via Depends(get_db) — the `async with`
    # takes care of closing (and rolling back on error) once the request ends.
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
