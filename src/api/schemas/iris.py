"""
Pydantic schemas for API request/response validation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class IrisFeatures(BaseModel):
    """
    Input features for Iris classification.
    """

    sepal_length: float = Field(
        ...,
        alias="sepal length (cm)",
        description="Sepal length in centimeters",
        ge=0.0,
        le=10.0,
        examples=[5.1],
    )
    sepal_width: float = Field(
        ...,
        alias="sepal width (cm)",
        description="Sepal width in centimeters",
        ge=0.0,
        le=10.0,
        examples=[3.5],
    )
    petal_length: float = Field(
        ...,
        alias="petal length (cm)",
        description="Petal length in centimeters",
        ge=0.0,
        le=10.0,
        examples=[1.4],
    )
    petal_width: float = Field(
        ...,
        alias="petal width (cm)",
        description="Petal width in centimeters",
        ge=0.0,
        le=10.0,
        examples=[0.2],
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        },
    }

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    @classmethod
    def validate_positive(cls, v: float, info) -> float:
        """Validate that all measurements are positive."""
        if v < 0:
            raise ValueError(f"{info.field_name} must be positive")
        if v > 100:
            raise ValueError(f"{info.field_name} seems unreasonably large (>{100})")
        return v


class PredictionRequest(BaseModel):
    """
    Request body for prediction endpoint.
    """

    features: IrisFeatures = Field(..., description="Iris flower measurements")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "sepal length (cm)": 5.1,
                        "sepal width (cm)": 3.5,
                        "petal length (cm)": 1.4,
                        "petal width (cm)": 0.2,
                    }
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Response from prediction endpoint.
    """

    prediction: str = Field(
        ..., description="Predicted Iris species", examples=["setosa"]
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the prediction",
        ge=0.0,
        le=1.0,
        examples=[0.98],
    )
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across all classes"
    )
    model_version: str = Field(
        ..., description="Version of the model used", examples=["v1.0.0"]
    )
    feature_contributions: Optional[Dict[str, float]] = Field(  # ← Should be here
        None, 
        description="SHAP feature contributions"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "setosa",
                    "confidence": 0.98,
                    "probabilities": {
                        "setosa": 0.98,
                        "versicolor": 0.01,
                        "virginica": 0.01,
                    },
                    "model_version": "v1.0.0",
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Request body for batch prediction endpoint.
    """

    samples: List[IrisFeatures] = Field(
        ...,
        description="List of Iris flower measurements",
        min_length=1,
        max_length=100,
    )


class BatchPredictionResponse(BaseModel):
    """
    Response from batch prediction endpoint.
    """

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    count: int = Field(..., description="Number of predictions made")


class HealthResponse(BaseModel):
    """
    Response from health check endpoint.
    """

    status: str = Field(..., description="API status", examples=["healthy"])
    version: str = Field(..., description="API version", examples=["1.0.0"])
    model_loaded: bool = Field(
        ..., description="Whether ML model is loaded", examples=[True]
    )
    environment: str = Field(
        ..., description="Current environment", examples=["development"]
    )


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """

    detail: str = Field(
        ..., description="Error message", examples=["Invalid input features"]
    )
    error_type: str = Field(
        default="validation_error",
        description="Type of error",
        examples=["validation_error", "model_error", "internal_error"],
    )
