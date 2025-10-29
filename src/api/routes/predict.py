"""
Prediction endpoints for Iris classification.
"""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.iris import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.core.config import settings
from src.models.model_loader import model_loader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


def _get_predictor():
    """
    Get predictor instance or raise HTTP exception.

    Returns:
        IrisPredictor instance

    Raises:
        HTTPException: If model not loaded
    """
    predictor = model_loader.get_predictor()
    if predictor is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )
    return predictor


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make a prediction",
    description="Classify an Iris flower based on its measurements",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make a prediction for a single Iris sample.

    Args:
        request: Prediction request with features

    Returns:
        Prediction response with class and probabilities

    Raises:
        HTTPException: If prediction fails
    """
    try:
        predictor = _get_predictor()
        
        # Convert request to feature dict
        features = {
            "sepal length (cm)": request.features.sepal_length,
            "sepal width (cm)": request.features.sepal_width,
            "petal length (cm)": request.features.petal_length,
            "petal width (cm)": request.features.petal_width,
        }
        
        # Make prediction with SHAP values
        result = predictor.predict(features, include_shap=True)  # ← ADD include_shap=True
        
        logger.info(
            f"Prediction made: {result['prediction']} "
            f"(confidence: {result['confidence']:.4f})"
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version=settings.model_version,
            feature_contributions=result.get("feature_contributions"),  # ← ADD this line
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input features: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make batch predictions",
    description="Classify multiple Iris flowers at once",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make predictions for multiple Iris samples.

    Args:
        request: Batch prediction request with list of features

    Returns:
        Batch prediction response with all predictions

    Raises:
        HTTPException: If prediction fails
    """
    try:
        predictor = _get_predictor()

        # Convert list of Pydantic models to list of dicts
        features_list = [
            {
                "sepal length (cm)": sample.sepal_length,
                "sepal width (cm)": sample.sepal_width,
                "petal length (cm)": sample.petal_length,
                "petal width (cm)": sample.petal_width,
            }
            for sample in request.samples
        ]

        # Make batch prediction
        import pandas as pd

        df = pd.DataFrame(features_list)
        results = predictor.predict_batch(df)

        # Convert to response format
        predictions = [
            PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                model_version=settings.model_version,
            )
            for result in results
        ]

        logger.info(f"Batch prediction completed: {len(predictions)} samples")

        return BatchPredictionResponse(predictions=predictions, count=len(predictions))

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input features: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.get(
    "/model/info",
    status_code=status.HTTP_200_OK,
    summary="Get model information and performance metrics",
    description="Retrieve model metadata, training metrics, and performance statistics",
)
async def get_model_info() -> Dict:
    """
    Get model information and performance metrics.

    Returns:
        Dictionary with model metadata and metrics

    Raises:
        HTTPException: If model not loaded or metrics unavailable
    """
    try:
        predictor = _get_predictor()

        # Get model registry to read metrics
        import json
        from pathlib import Path

        registry_path = Path("models/model_registry.json")

        if not registry_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Model registry not found"
            )

        with open(registry_path, "r") as f:
            registry = json.load(f)

        # Get active model metrics
        active_model_id = registry.get("active_model")
        active_model = next(
            (m for m in registry["models"] if m["model_id"] == active_model_id), None
        )

        if not active_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Active model not found in registry",
            )

        # Return comprehensive model info
        return {
            "model_id": active_model["model_id"],
            "model_type": active_model.get("model_type", "logistic_regression"),
            "model_version": settings.model_version,
            "timestamp": active_model.get("timestamp"),
            "metrics": {
                "train_accuracy": active_model["metrics"]["train_accuracy"],
                "test_accuracy": active_model["metrics"]["test_accuracy"],
                "test_f1_macro": active_model["metrics"]["test_f1_macro"],
            },
            "dataset": {
                "name": registry["metadata"]["dataset"],
                "features": registry["metadata"]["features"],
                "target_classes": registry["metadata"]["target_classes"],
                "num_classes": registry["metadata"]["num_classes"],
            },
            "class_names": predictor.class_names,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}",
        )


@router.post(
    "/compare",
    status_code=status.HTTP_200_OK,
    summary="Compare two flower predictions",
    description="Compare predictions for two different iris flowers side-by-side",
)
async def compare_predictions(
    flower1: PredictionRequest, flower2: PredictionRequest
) -> Dict:
    """
    Compare predictions for two flowers.

    Args:
        flower1: First flower's features
        flower2: Second flower's features

    Returns:
        Comparison results with both predictions and differences

    Raises:
        HTTPException: If prediction fails
    """
    try:
        predictor = _get_predictor()

        # Convert to dicts
        features1 = {
            "sepal length (cm)": flower1.features.sepal_length,
            "sepal width (cm)": flower1.features.sepal_width,
            "petal length (cm)": flower1.features.petal_length,
            "petal width (cm)": flower1.features.petal_width,
        }

        features2 = {
            "sepal length (cm)": flower2.features.sepal_length,
            "sepal width (cm)": flower2.features.sepal_width,
            "petal length (cm)": flower2.features.petal_length,
            "petal width (cm)": flower2.features.petal_width,
        }

        # Make predictions with SHAP
        result1 = predictor.predict(features1, include_shap=True)
        result2 = predictor.predict(features2, include_shap=True)

        # Calculate feature differences
        feature_differences = {
            feature: features2[feature] - features1[feature]
            for feature in features1.keys()
        }

        # Calculate confidence difference
        confidence_diff = result2["confidence"] - result1["confidence"]

        logger.info(f"Comparison: {result1['prediction']} vs {result2['prediction']}")

        return {
            "flower1": {
                "features": features1,
                "prediction": result1["prediction"],
                "confidence": result1["confidence"],
                "probabilities": result1["probabilities"],
                "feature_contributions": result1.get("feature_contributions"),
            },
            "flower2": {
                "features": features2,
                "prediction": result2["prediction"],
                "confidence": result2["confidence"],
                "probabilities": result2["probabilities"],
                "feature_contributions": result2.get("feature_contributions"),
            },
            "comparison": {
                "same_prediction": result1["prediction"] == result2["prediction"],
                "confidence_difference": confidence_diff,
                "feature_differences": feature_differences,
                "most_different_feature": max(
                    feature_differences.items(), key=lambda x: abs(x[1])
                )[0],
            },
            "model_version": settings.model_version,
        }

    except ValueError as e:
        logger.error(f"Validation error in comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input features: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}",
        )
