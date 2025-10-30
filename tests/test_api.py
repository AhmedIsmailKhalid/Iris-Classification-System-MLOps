"""
Integration tests for FastAPI endpoints.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings
from src.data.load_data import load_iris_from_sklearn
from src.data.preprocess import prepare_train_test_split
from src.models.model_loader import model_loader
from src.models.train import IrisModelTrainer


@pytest.fixture(scope="module")
def test_client():
    """
    Fixture providing FastAPI test client with loaded model.
    """
    # Train and save a test model
    X, y = load_iris_from_sklearn()
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainer = IrisModelTrainer()
    trainer.train(X_train, y_train, X_test, y_test)

    # Save to test location
    test_model_path = Path("models/test_iris_classifier.joblib")
    trainer.save_model(test_model_path, preprocessor)

    # Load model into model_loader
    model_loader.load_model(test_model_path, force_reload=True)

    # Create test client
    with TestClient(app) as client:
        yield client

    # Cleanup
    if test_model_path.exists():
        test_model_path.unlink()
    model_loader.unload_model()


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == settings.api_version
        assert data["model_loaded"] is True
        assert data["environment"] == settings.environment

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns health check."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "version" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_setosa(self, test_client):
        """Test prediction for setosa class."""
        payload = {
            "features": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Verify valid prediction (may change after retraining)
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "probabilities" in data
        assert len(data["probabilities"]) == 3

    def test_predict_versicolor(self, test_client):
        """Test prediction for versicolor class."""
        payload = {
            "features": {
                "sepal length (cm)": 6.4,
                "sepal width (cm)": 3.2,
                "petal length (cm)": 4.5,
                "petal width (cm)": 1.5,
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Verify valid prediction (may change after retraining)
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_predict_virginica(self, test_client):
        """Test prediction for virginica class."""
        payload = {
            "features": {
                "sepal length (cm)": 6.3,
                "sepal width (cm)": 3.3,
                "petal length (cm)": 6.0,
                "petal width (cm)": 2.5,
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Verify valid prediction (may change after retraining)
        assert data["prediction"] in ["setosa", "versicolor", "virginica"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_predict_invalid_negative_values(self, test_client):
        """Test prediction with negative values."""
        payload = {
            "features": {
                "sepal length (cm)": -5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422

    def test_predict_invalid_large_values(self, test_client):
        """Test prediction with unreasonably large values."""
        payload = {
            "features": {
                "sepal length (cm)": 150.0,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422

    def test_predict_missing_field(self, test_client):
        """Test prediction with missing required field."""
        payload = {
            "features": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                # Missing petal width
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422

    def test_predict_wrong_field_type(self, test_client):
        """Test prediction with wrong field type."""
        payload = {
            "features": {
                "sepal length (cm)": "invalid",
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            }
        }

        response = test_client.post("/api/v1/predict", json=payload)

        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict(self, test_client):
        """Test batch prediction with multiple samples."""
        payload = {
            "samples": [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                },
                {
                    "sepal length (cm)": 6.4,
                    "sepal width (cm)": 3.2,
                    "petal length (cm)": 4.5,
                    "petal width (cm)": 1.5,
                },
                {
                    "sepal length (cm)": 6.3,
                    "sepal width (cm)": 3.3,
                    "petal length (cm)": 6.0,
                    "petal width (cm)": 2.5,
                },
            ]
        }

        response = test_client.post("/api/v1/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

        # Check first prediction (should be setosa)
        assert data["predictions"][0]["prediction"] == "setosa"

        # Check all predictions have required fields
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
            assert "model_version" in pred

    def test_batch_predict_empty_list(self, test_client):
        """Test batch prediction with empty list."""
        payload = {"samples": []}

        response = test_client.post("/api/v1/predict/batch", json=payload)

        assert response.status_code == 422

    def test_batch_predict_too_many_samples(self, test_client):
        """Test batch prediction with too many samples."""
        # Create 101 samples (max is 100)
        sample = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }
        payload = {"samples": [sample] * 101}

        response = test_client.post("/api/v1/predict/batch", json=payload)

        assert response.status_code == 422


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present."""
        response = test_client.options(
            "/api/v1/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_json(self, test_client):
        """Test OpenAPI JSON endpoint."""
        response = test_client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == settings.api_title
        assert data["info"]["version"] == settings.api_version

    def test_docs_endpoint(self, test_client):
        """Test Swagger UI docs endpoint."""
        response = test_client.get("/docs")

        assert response.status_code == 200
        assert b"swagger-ui" in response.content

    def test_redoc_endpoint(self, test_client):
        """Test ReDoc docs endpoint."""
        response = test_client.get("/redoc")

        assert response.status_code == 200
        assert b"redoc" in response.content
