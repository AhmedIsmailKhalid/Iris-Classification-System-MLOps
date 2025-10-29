"""
Tests for drift detection.
"""

import numpy as np
import pytest

from src.monitoring.data_generator import SyntheticDataGenerator
from src.monitoring.drift_detector import DriftDetector


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def detector(self):
        """Fixture providing drift detector."""
        return DriftDetector(significance_level=0.05)

    @pytest.fixture
    def generator(self):
        """Fixture providing data generator."""
        return SyntheticDataGenerator()

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.significance_level == 0.05

    def test_kolmogorov_smirnov_test_no_drift(self, detector):
        """Test KS test with identical distributions."""
        ref_data = np.random.normal(0, 1, 1000)
        curr_data = np.random.normal(0, 1, 1000)

        statistic, p_value = detector.kolmogorov_smirnov_test(ref_data, curr_data)

        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        # With identical distributions, p-value should be high
        assert p_value > 0.05

    def test_kolmogorov_smirnov_test_with_drift(self, detector):
        """Test KS test with different distributions."""
        ref_data = np.random.normal(0, 1, 1000)
        curr_data = np.random.normal(2, 1, 1000)  # Shifted mean

        statistic, p_value = detector.kolmogorov_smirnov_test(ref_data, curr_data)

        # With different distributions, p-value should be low
        assert p_value < 0.05

    def test_population_stability_index_no_drift(self, detector):
        """Test PSI with similar distributions."""
        ref_data = np.random.normal(5, 1, 1000)
        curr_data = np.random.normal(5.1, 1.05, 1000)

        psi = detector.population_stability_index(ref_data, curr_data)

        assert isinstance(psi, float)
        assert psi >= 0
        # PSI should be low for similar distributions
        assert psi < 0.1

    def test_population_stability_index_with_drift(self, detector):
        """Test PSI with different distributions."""
        ref_data = np.random.normal(0, 1, 1000)
        curr_data = np.random.normal(3, 1, 1000)

        psi = detector.population_stability_index(ref_data, curr_data)

        # PSI should be high for different distributions
        assert psi > 0.2

    def test_detect_feature_drift_no_drift(self, detector, generator):
        """Test feature drift detection with no drift."""
        reference = generator.generate_normal_data(200)
        current = generator.generate_normal_data(200)

        feature = "sepal length (cm)"
        result = detector.detect_feature_drift(reference, current, feature)

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "ks_pvalue" in result
        assert "psi" in result
        assert result["feature"] == feature

    def test_detect_feature_drift_with_drift(self, detector, generator):
        """Test feature drift detection with actual drift."""
        reference = generator.generate_normal_data(200)
        current = generator.generate_drifted_data(
            n_samples=200, drift_type="shift", drift_magnitude=2.5
        )

        feature = "petal length (cm)"
        result = detector.detect_feature_drift(reference, current, feature)

        assert result["drift_detected"] is True
        assert result["psi"] > 0.2 or result["ks_pvalue"] < 0.05

    def test_detect_dataset_drift_no_drift(self, detector, generator):
        """Test dataset drift detection with no drift."""
        reference = generator.generate_normal_data(200)
        current = generator.generate_normal_data(200)

        result = detector.detect_dataset_drift(reference, current)

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "drifted_features" in result
        assert "drift_severity" in result
        assert "recommendation" in result
        assert "feature_drift_details" in result

        # Should not detect drift
        assert result["drift_detected"] is False or result["drift_severity"] < 0.5

    def test_detect_dataset_drift_with_drift(self, detector, generator):
        """Test dataset drift detection with actual drift."""
        reference = generator.generate_normal_data(200)
        current = generator.generate_drifted_data(
            n_samples=200, drift_type="shift", drift_magnitude=2.5
        )

        result = detector.detect_dataset_drift(reference, current)

        # Should detect drift
        assert result["drift_detected"] is True
        assert len(result["drifted_features"]) > 0
        assert result["drift_severity"] > 0

    def test_drift_severity_calculation(self, detector, generator):
        """Test drift severity is calculated correctly."""
        reference = generator.generate_normal_data(200)
        current = generator.generate_drifted_data(
            n_samples=200, drift_type="shift", drift_magnitude=3.0
        )

        result = detector.detect_dataset_drift(reference, current)

        # Drift severity should be between 0 and 1
        assert 0 <= result["drift_severity"] <= 1

    def test_recommendation_logic(self, detector, generator):
        """Test recommendation based on drift severity."""
        reference = generator.generate_normal_data(200)

        # High drift
        high_drift = generator.generate_drifted_data(
            n_samples=200, drift_type="shift", drift_magnitude=3.0
        )
        result_high = detector.detect_dataset_drift(reference, high_drift)

        if result_high["drift_detected"]:
            assert result_high["recommendation"] in [
                "immediate_retraining",
                "schedule_retraining",
                "monitor",
            ]

    def test_feature_drift_details(self, detector, generator):
        """Test that feature drift details are comprehensive."""
        reference = generator.generate_normal_data(100)
        current = generator.generate_drifted_data(
            n_samples=100, drift_type="shift", drift_magnitude=2.0
        )

        result = detector.detect_dataset_drift(reference, current)

        for detail in result["feature_drift_details"]:
            assert "feature" in detail
            assert "drift_detected" in detail
            assert "ks_statistic" in detail
            assert "ks_pvalue" in detail
            assert "psi" in detail
            assert "reference_mean" in detail
            assert "current_mean" in detail
            assert "mean_difference_percent" in detail
