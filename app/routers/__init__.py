from .quiz import router as quiz_router
from .analytics import router as analytics_router
from .health import router as health_router

__all__ = ["quiz_router", "analytics_router", "health_router"]
