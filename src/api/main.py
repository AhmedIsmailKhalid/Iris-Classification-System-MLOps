"""
Main FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health, monitoring, predict
from src.core.config import settings
from src.core.logging import setup_logging
from src.models.model_loader import model_loader

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Starting Iris ML Pipeline API")
    logger.info("=" * 80)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API Version: {settings.api_version}")
    logger.info(f"Model Path: {settings.model_path}")

    # Load model
    try:
        if settings.model_path.exists():
            logger.info("Loading ML model...")
            model_loader.load_model(settings.model_path)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning(
                f"⚠️  Model file not found at {settings.model_path}. "
                "Train a model first using: python scripts/train_model.py"
            )
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}", exc_info=True)
        logger.warning("API will start but predictions will fail until model is loaded")

    logger.info("=" * 80)
    logger.info(f" API is ready at http://{settings.api_host}:{settings.api_port}")
    logger.info(
        f" Docs available at http://{settings.api_host}:{settings.api_port}/docs"
    )
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Shutting down Iris ML Pipeline API")
    model_loader.unload_model()
    logger.info("Model unloaded from memory")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Args:
        request: Incoming request
        exc: Exception that was raised

    Returns:
        JSON error response
    """
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else None,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error_type": "internal_error"},
    )


# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(monitoring.router, prefix="/api/v1/monitoring")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests.

    Args:
        request: Incoming request
        call_next: Next middleware/route handler

    Returns:
        Response from next handler
    """
    logger.debug(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    response = await call_next(request)

    logger.debug(
        f"Response status: {response.status_code}",
        extra={"status_code": response.status_code},
    )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
