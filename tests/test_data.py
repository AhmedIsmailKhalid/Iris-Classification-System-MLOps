"""
Unit tests for data loading and preprocessing modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import (
    get_iris_data,
    load_iris_from_csv,
    load_iris_from_sklearn,
    save_iris_to_csv,
)
from src.data.preprocess import IrisPreprocessor, prepare_train_test_split


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_iris_from_sklearn(self):
        """Test loading Iris dataset from sklearn."""
        X, y = load_iris_from_sklearn()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 150
        assert len(y) == 150
        assert "species" in X.columns
        assert y.nunique() == 3

    def test_save_and_load_csv(self):
        """Test saving and loading dataset to/from CSV."""
        # Load original data
        X_orig, y_orig = load_iris_from_sklearn()

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_iris.csv"
            save_iris_to_csv(X_orig, y_orig, csv_path)

            # Verify file exists
            assert csv_path.exists()

            # Load back
            X_loaded, y_loaded = load_iris_from_csv(csv_path)

            # Verify data integrity
            assert len(X_loaded) == len(X_orig)
            assert len(y_loaded) == len(y_orig)

            # Check values are equal (dtype may differ between int32/int64)
            pd.testing.assert_series_equal(
                y_loaded,
                y_orig,
                check_names=False,
                check_dtype=False,  # ✅ Added: Ignore dtype differences
            )

            # Verify actual values match
            assert (y_loaded.values == y_orig.values).all()

    def test_load_nonexistent_csv(self):
        """Test loading from non-existent CSV raises error."""
        with pytest.raises(FileNotFoundError):
            load_iris_from_csv(Path("nonexistent.csv"))

    def test_get_iris_data(self):
        """Test main data retrieval function."""
        X, y = get_iris_data(use_csv=False)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 150


class TestPreprocessing:
    """Tests for preprocessing functions."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample Iris data."""
        X, y = load_iris_from_sklearn()
        return X, y

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = IrisPreprocessor(scale_features=True)
        assert preprocessor.scale_features is True
        assert preprocessor.scaler is not None
        assert preprocessor._is_fitted is False

    def test_preprocessor_validate_data(self, sample_data):
        """Test data validation."""
        X, y = sample_data
        preprocessor = IrisPreprocessor()

        # Should not raise for valid data
        preprocessor.validate_data(X, y)

    def test_preprocessor_validate_missing_values(self, sample_data):
        """Test validation fails with missing values."""
        X, y = sample_data
        X_missing = X.copy()
        X_missing.iloc[0, 0] = np.nan

        preprocessor = IrisPreprocessor()
        with pytest.raises(ValueError, match="missing values"):
            preprocessor.validate_data(X_missing, y)

    def test_preprocessor_validate_negative_values(self, sample_data):
        """Test validation fails with negative values."""
        X, y = sample_data
        X_negative = X.copy()
        X_negative.iloc[0, 0] = -1.0

        preprocessor = IrisPreprocessor()
        with pytest.raises(ValueError, match="negative values"):
            preprocessor.validate_data(X_negative, y)

    def test_preprocessor_fit_transform(self, sample_data):
        """Test fit and transform."""
        X, y = sample_data
        preprocessor = IrisPreprocessor(scale_features=True)

        X_transformed = preprocessor.fit_transform(X, y)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == (150, 4)
        assert preprocessor._is_fitted is True

        # Check scaling worked (mean ~0, std ~1)
        assert np.abs(X_transformed.mean()) < 0.1
        assert np.abs(X_transformed.std() - 1.0) < 0.1

    def test_preprocessor_transform_before_fit(self, sample_data):
        """Test transform fails before fit."""
        X, _ = sample_data
        preprocessor = IrisPreprocessor()

        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(X)

    def test_prepare_train_test_split(self, sample_data):
        """Test train/test split preparation."""
        X, y = sample_data

        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Check shapes
        assert X_train.shape[0] == 120
        assert X_test.shape[0] == 30
        assert y_train.shape[0] == 120
        assert y_test.shape[0] == 30

        # Check preprocessor is fitted
        assert preprocessor._is_fitted is True

        # Check feature dimension
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

    def test_train_test_split_stratification(self, sample_data):
        """Test that train/test split maintains class distribution."""
        X, y = sample_data

        _, _, y_train, y_test, _ = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Check all classes present in both splits
        assert len(np.unique(y_train)) == 3
        assert len(np.unique(y_test)) == 3
