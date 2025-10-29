"""
Tests for synthetic data generation.
"""

import pandas as pd
import pytest

from src.monitoring.data_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator class."""

    @pytest.fixture
    def generator(self):
        """Fixture providing data generator."""
        return SyntheticDataGenerator()

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.feature_stats is not None
        assert len(generator.feature_stats) == 4
        assert "sepal length (cm)" in generator.feature_stats
        assert generator.class_distribution is not None

    def test_generate_normal_data(self, generator):
        """Test normal data generation."""
        n_samples = 50
        data = generator.generate_normal_data(n_samples)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == n_samples
        assert "species" in data.columns

        # Check all features present
        for feature in generator.feature_stats.keys():
            assert feature in data.columns

    def test_generate_normal_data_statistics(self, generator):
        """Test that normal data follows expected distribution."""
        n_samples = 1000
        data = generator.generate_normal_data(n_samples)

        # Check means are approximately correct (within 20%)
        for feature, stats in generator.feature_stats.items():
            mean = data[feature].mean()
            expected_mean = stats["mean"]
            assert abs(mean - expected_mean) / expected_mean < 0.2

    def test_generate_normal_data_no_negative(self, generator):
        """Test that normal data has no negative values."""
        data = generator.generate_normal_data(100)

        for feature in generator.feature_stats.keys():
            assert (data[feature] >= 0.1).all()

    def test_generate_drifted_data_shift(self, generator):
        """Test drifted data generation with mean shift."""
        data = generator.generate_drifted_data(
            n_samples=50, drift_type="shift", drift_magnitude=2.0
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert "species" in data.columns

    def test_generate_drifted_data_scale(self, generator):
        """Test drifted data generation with variance change."""
        data = generator.generate_drifted_data(
            n_samples=50, drift_type="scale", drift_magnitude=2.0
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50

    def test_generate_drifted_data_extreme(self, generator):
        """Test drifted data generation with outliers."""
        data = generator.generate_drifted_data(
            n_samples=50, drift_type="extreme", drift_magnitude=2.0
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50

    def test_generate_drifted_data_has_shift(self, generator):
        """Test that drifted data actually differs from normal."""
        normal_data = generator.generate_normal_data(500)
        drifted_data = generator.generate_drifted_data(
            n_samples=500, drift_type="shift", drift_magnitude=2.5
        )

        # Check that means are significantly different
        for feature in generator.feature_stats.keys():
            normal_mean = normal_data[feature].mean()
            drifted_mean = drifted_data[feature].mean()

            # Drifted data should have higher mean
            assert drifted_mean > normal_mean

    def test_get_feature_statistics(self, generator):
        """Test feature statistics calculation."""
        data = generator.generate_normal_data(100)
        stats = generator.get_feature_statistics(data)

        assert isinstance(stats, dict)
        assert len(stats) == 4

        for feature, feature_stats in stats.items():
            assert "mean" in feature_stats
            assert "std" in feature_stats
            assert "min" in feature_stats
            assert "max" in feature_stats
            assert "median" in feature_stats

    def test_different_sample_sizes(self, generator):
        """Test data generation with different sample sizes."""
        for n in [10, 50, 100, 500]:
            data = generator.generate_normal_data(n)
            assert len(data) == n

    def test_drift_magnitude_effect(self, generator):
        """Test that higher drift magnitude creates more drift."""
        data_low = generator.generate_drifted_data(
            n_samples=200, drift_type="shift", drift_magnitude=1.0
        )
        data_high = generator.generate_drifted_data(
            n_samples=200, drift_type="shift", drift_magnitude=3.0
        )

        # Higher magnitude should have higher mean
        feature = "petal length (cm)"
        assert data_high[feature].mean() > data_low[feature].mean()
