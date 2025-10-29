"""
Tests for data logging functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.monitoring.data_logger import DataLogger


class TestDataLogger:
    """Tests for DataLogger class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Fixture providing temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def logger(self, temp_log_dir):
        """Fixture providing data logger with temp directory."""
        return DataLogger(log_dir=temp_log_dir)

    def test_initialization(self, logger, temp_log_dir):
        """Test logger initialization."""
        assert logger.log_dir == temp_log_dir
        assert logger.log_dir.exists()
        assert logger.prediction_log_path.parent.exists()
        assert logger.new_data_path.parent.exists()

    def test_log_prediction(self, logger):
        """Test logging a prediction."""
        features = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }

        logger.log_prediction(features=features, prediction="setosa", confidence=0.95)

        # Check log file created
        assert logger.prediction_log_path.exists()

        # Check log contents
        df = pd.read_csv(logger.prediction_log_path)
        assert len(df) == 1
        assert df.iloc[0]["prediction"] == "setosa"
        assert df.iloc[0]["confidence"] == 0.95

    def test_log_multiple_predictions(self, logger):
        """Test logging multiple predictions."""
        for i in range(5):
            features = {
                "sepal length (cm)": 5.0 + i * 0.1,
                "sepal width (cm)": 3.0,
                "petal length (cm)": 1.5,
                "petal width (cm)": 0.2,
            }
            logger.log_prediction(
                features=features, prediction="setosa", confidence=0.9
            )

        df = pd.read_csv(logger.prediction_log_path)
        assert len(df) == 5

    def test_log_prediction_with_feedback(self, logger):
        """Test logging prediction with user feedback."""
        features = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }

        logger.log_prediction(
            features=features,
            prediction="setosa",
            confidence=0.95,
            user_feedback="correct",
        )

        df = pd.read_csv(logger.prediction_log_path)
        assert df.iloc[0]["user_feedback"] == "correct"

    def test_save_new_data(self, logger):
        """Test saving new data."""
        data = pd.DataFrame(
            {
                "sepal length (cm)": [5.1, 5.2],
                "sepal width (cm)": [3.5, 3.6],
                "petal length (cm)": [1.4, 1.5],
                "petal width (cm)": [0.2, 0.3],
                "species": ["setosa", "setosa"],
            }
        )

        logger.save_new_data(data)

        assert logger.new_data_path.exists()

        loaded = pd.read_csv(logger.new_data_path)
        assert len(loaded) == 2

    def test_save_new_data_append(self, logger):
        """Test appending new data."""
        data1 = pd.DataFrame(
            {
                "sepal length (cm)": [5.1],
                "sepal width (cm)": [3.5],
                "petal length (cm)": [1.4],
                "petal width (cm)": [0.2],
                "species": ["setosa"],
            }
        )

        data2 = pd.DataFrame(
            {
                "sepal length (cm)": [6.0],
                "sepal width (cm)": [3.0],
                "petal length (cm)": [4.5],
                "petal width (cm)": [1.5],
                "species": ["versicolor"],
            }
        )

        logger.save_new_data(data1)
        logger.save_new_data(data2)

        loaded = pd.read_csv(logger.new_data_path)
        assert len(loaded) == 2

    def test_load_new_data(self, logger):
        """Test loading new data."""
        # Initially no data
        result = logger.load_new_data()
        assert result is None

        # Save some data
        data = pd.DataFrame(
            {
                "sepal length (cm)": [5.1],
                "sepal width (cm)": [3.5],
                "petal length (cm)": [1.4],
                "petal width (cm)": [0.2],
                "species": ["setosa"],
            }
        )
        logger.save_new_data(data)

        # Load data
        loaded = logger.load_new_data()
        assert loaded is not None
        assert len(loaded) == 1

    def test_clear_new_data(self, logger):
        """Test clearing new data."""
        # Save some data
        data = pd.DataFrame(
            {
                "sepal length (cm)": [5.1],
                "sepal width (cm)": [3.5],
                "petal length (cm)": [1.4],
                "petal width (cm)": [0.2],
                "species": ["setosa"],
            }
        )
        logger.save_new_data(data)

        assert logger.new_data_path.exists()

        # Clear data
        logger.clear_new_data()

        assert not logger.new_data_path.exists()

    def test_get_prediction_count_empty(self, logger):
        """Test getting prediction count when empty."""
        count = logger.get_prediction_count()
        assert count == 0

    def test_get_prediction_count(self, logger):
        """Test getting prediction count."""
        features = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }

        for _ in range(10):
            logger.log_prediction(
                features=features, prediction="setosa", confidence=0.9
            )

        count = logger.get_prediction_count()
        assert count == 10
