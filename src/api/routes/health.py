"""
Health check endpoints for API monitoring.
"""

import logging

from fastapi import APIRouter

from src.api.schemas.iris import HealthResponse
from src.core.config import settings
from src.models.model_loader import model_loader

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running and model is loaded",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with API status
    """
    model_loaded = model_loader.get_predictor() is not None

    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        model_loaded=model_loaded,
        environment=settings.environment,
    )


@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def root() -> HealthResponse:
    """
    Root endpoint - same as health check.
    """
    return await health_check()
