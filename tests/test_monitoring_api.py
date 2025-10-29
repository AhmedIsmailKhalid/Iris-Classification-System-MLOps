"""
Integration tests for monitoring API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestMonitoringEndpoints:
    """Tests for monitoring API endpoints."""

    @pytest.fixture
    def client(self):
        """Fixture providing test client."""
        return TestClient(app)

    def test_generate_normal_data(self, client):
        """Test generating normal synthetic data."""
        response = client.post(
            "/api/v1/monitoring/generate-data",
            json={
                "n_samples": 50,
                "data_type": "normal",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["samples_generated"] == 50
        assert data["data_type"] == "normal"
        assert "statistics" in data

    def test_generate_drifted_data(self, client):
        """Test generating drifted synthetic data."""
        response = client.post(
            "/api/v1/monitoring/generate-data",
            json={
                "n_samples": 50,
                "data_type": "drifted",
                "drift_type": "shift",
                "drift_magnitude": 2.5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["samples_generated"] == 50
        assert data["data_type"] == "drifted"

    def test_generate_data_invalid_sample_count(self, client):
        """Test generating data with invalid sample count."""
        response = client.post(
            "/api/v1/monitoring/generate-data",
            json={
                "n_samples": 5,  # Below minimum of 10
                "data_type": "normal",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_get_data_stats(self, client):
        """Test getting data statistics."""
        response = client.get("/api/v1/monitoring/data-stats")

        assert response.status_code == 200
        data = response.json()

        assert "total_predictions" in data
        assert "new_data_samples" in data
        assert "drift_status" in data
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["new_data_samples"], int)

    def test_check_drift_insufficient_data(self, client):
        """Test drift detection with insufficient data."""
        # Clear any existing data first by checking drift
        response = client.post("/api/v1/monitoring/check-drift")

        # Should indicate insufficient data or no drift
        assert response.status_code == 200
        data = response.json()

        # Either insufficient data message or drift result
        assert "drift_detected" in data or "message" in data

    def test_check_drift_with_data(self, client):
        """Test drift detection after generating data."""
        # Generate some normal data
        client.post(
            "/api/v1/monitoring/generate-data",
            json={"n_samples": 50, "data_type": "normal"},
        )

        # Check for drift
        response = client.post("/api/v1/monitoring/check-drift")

        assert response.status_code == 200
        data = response.json()

        assert "drift_detected" in data

    def test_trigger_retraining_endpoint_exists(self, client):
        """Test that trigger retraining endpoint exists."""
        response = client.post("/api/v1/monitoring/trigger-retraining")

        # Should return 200 (even if GitHub token not configured)
        assert response.status_code == 200
        data = response.json()

        assert "success" in data
        assert "message" in data

    def test_workflow_status_endpoint_exists(self, client):
        """Test that workflow status endpoint exists."""
        response = client.get("/api/v1/monitoring/workflow-status")

        # Should return 200 (even if fails to get status)
        assert response.status_code == 200
        data = response.json()

        # Should have success field
        assert "success" in data

    def test_data_generation_workflow(self, client):
        """Test complete data generation and drift workflow."""
        # Step 1: Generate normal data
        response1 = client.post(
            "/api/v1/monitoring/generate-data",
            json={"n_samples": 50, "data_type": "normal"},
        )
        assert response1.status_code == 200

        # Step 2: Check stats
        response2 = client.get("/api/v1/monitoring/data-stats")
        assert response2.status_code == 200
        stats = response2.json()
        assert stats["new_data_samples"] >= 50

        # Step 3: Check drift
        response3 = client.post("/api/v1/monitoring/check-drift")
        assert response3.status_code == 200

    def test_generate_data_different_drift_types(self, client):
        """Test generating data with different drift types."""
        drift_types = ["shift", "scale", "extreme"]

        for drift_type in drift_types:
            response = client.post(
                "/api/v1/monitoring/generate-data",
                json={
                    "n_samples": 30,
                    "data_type": "drifted",
                    "drift_type": drift_type,
                    "drift_magnitude": 2.0,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_generate_data_various_magnitudes(self, client):
        """Test generating drifted data with various magnitudes."""
        magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0]

        for magnitude in magnitudes:
            response = client.post(
                "/api/v1/monitoring/generate-data",
                json={
                    "n_samples": 20,
                    "data_type": "drifted",
                    "drift_type": "shift",
                    "drift_magnitude": magnitude,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
