"""
API routes for monitoring, drift detection, and data generation.
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.config import settings
from src.data.load_data import load_iris_from_sklearn
from src.monitoring.data_generator import SyntheticDataGenerator
from src.monitoring.data_logger import DataLogger
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.github_integration import GitHubWorkflowTrigger

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])

# Initialize services
data_generator = SyntheticDataGenerator()
data_logger = DataLogger()
drift_detector = DriftDetector()
github_trigger = GitHubWorkflowTrigger(
    token=settings.github_token,
    repo=settings.github_repo
)


class GenerateDataRequest(BaseModel):
    """Request model for data generation."""

    n_samples: int = Field(50, ge=10, le=500, description="Number of samples to generate")
    data_type: Literal["normal", "drifted"] = Field(
        "normal", description="Type of data to generate"
    )
    drift_type: Optional[Literal["shift", "scale", "extreme"]] = Field(
        "shift", description="Type of drift (if data_type=drifted)"
    )
    drift_magnitude: float = Field(
        2.0, ge=0.5, le=5.0, description="Drift magnitude multiplier"
    )


class DataStatsResponse(BaseModel):
    """Response model for data statistics."""

    total_predictions: int
    new_data_samples: int
    drift_status: str
    last_check: Optional[str]


@router.post(
    "/generate-data",
    status_code=status.HTTP_200_OK,
    summary="Generate synthetic data",
    description="Generate synthetic iris data for testing drift detection"
)
async def generate_synthetic_data(request: GenerateDataRequest) -> Dict:
    """
    Generate synthetic data (normal or drifted).

    Args:
        request: Data generation parameters

    Returns:
        Dictionary with generation results
    """
    try:
        logger.info(
            f"Generating {request.n_samples} {request.data_type} samples"
        )

        if request.data_type == "normal":
            data = data_generator.generate_normal_data(request.n_samples)
        else:
            data = data_generator.generate_drifted_data(
                n_samples=request.n_samples,
                drift_type=request.drift_type,
                drift_magnitude=request.drift_magnitude,
            )

        # Save generated data
        data_logger.save_new_data(data)

        # Calculate statistics
        stats = data_generator.get_feature_statistics(data)

        return {
            "success": True,
            "samples_generated": len(data),
            "data_type": request.data_type,
            "statistics": stats,
            "message": f"Generated {len(data)} {request.data_type} samples",
        }

    except Exception as e:
        logger.error(f"Failed to generate data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data generation failed: {str(e)}"
        )


@router.get(
    "/data-stats",
    response_model=DataStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get data statistics",
    description="Get current data collection and drift statistics"
)
async def get_data_stats() -> DataStatsResponse:
    """
    Get current data statistics.

    Returns:
        Data statistics including prediction count and drift status
    """
    try:
        prediction_count = data_logger.get_prediction_count()

        new_data = data_logger.load_new_data()
        new_data_count = len(new_data) if new_data is not None else 0

        # Simple drift status
        if new_data_count >= 50:
            drift_status = "ready_for_check"
        elif new_data_count > 0:
            drift_status = f"collecting ({new_data_count}/50)"
        else:
            drift_status = "no_new_data"

        return DataStatsResponse(
            total_predictions=prediction_count,
            new_data_samples=new_data_count,
            drift_status=drift_status,
            last_check=None,
        )

    except Exception as e:
        logger.error(f"Failed to get data stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data statistics: {str(e)}"
        )


@router.post(
    "/check-drift",
    status_code=status.HTTP_200_OK,
    summary="Check for data drift",
    description="Perform drift detection on accumulated data"
)
async def check_drift() -> Dict:
    """
    Check for data drift.

    Returns:
        Drift detection results
    """
    try:
        logger.info("Starting drift detection check")

        # Load reference data (original Iris dataset)
        reference_X, _ = load_iris_from_sklearn()

        # Load new data
        new_data = data_logger.load_new_data()

        if new_data is None or len(new_data) < 30:
            return {
                "drift_detected": False,
                "message": "Insufficient data for drift detection (need at least 30 samples)",
                "current_samples": len(new_data) if new_data is not None else 0,
            }

        # Perform drift detection
        drift_result = drift_detector.detect_dataset_drift(reference_X, new_data)

        return drift_result

    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )


@router.post(
    "/trigger-retraining",
    status_code=status.HTTP_200_OK,
    summary="Trigger model retraining",
    description="Manually trigger automated retraining workflow via GitHub Actions API"
)
async def trigger_retraining(force: bool = False) -> Dict:
    """
    Trigger GitHub Actions retraining workflow.
    
    Args:
        force: Force retraining even with insufficient data
    
    Returns:
        Status of trigger request
    """
    try:
        logger.info("Manual retraining trigger requested")
        
        # Check if we have data
        new_data = data_logger.load_new_data()
        
        if new_data is None or len(new_data) < 30:
            if not force:
                return {
                    "success": False,
                    "message": "Insufficient data for retraining",
                    "current_samples": len(new_data) if new_data is not None else 0,
                    "required_samples": 30,
                    "suggestion": "Use force=true to retrain anyway, or generate more data",
                }
        
        # Trigger GitHub Actions workflow
        result = github_trigger.trigger_workflow(
            workflow_id="automated-retraining.yml",
            ref="main",
            inputs={"force": str(force).lower()}
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": "Retraining workflow triggered successfully! 🚀",
                "current_samples": len(new_data) if new_data is not None else 0,
                "workflow_url": result["url"],
                "estimated_time": "3-5 minutes",
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Failed to trigger workflow"),
                "error": result.get("error"),
                "manual_trigger_url": f"https://github.com/{settings.github_repo}/actions/workflows/automated-retraining.yml",
            }
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger retraining: {str(e)}"
        )


@router.get(
    "/workflow-status",
    status_code=status.HTTP_200_OK,
    summary="Get retraining workflow status",
    description="Get recent runs of the automated retraining workflow"
)
async def get_workflow_status() -> Dict:
    """
    Get status of recent retraining workflows.
    
    Returns:
        Recent workflow run information
    """
    try:
        result = github_trigger.get_workflow_status(
            workflow_id="automated-retraining.yml",
            limit=5
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve workflow status: {str(e)}"
        )