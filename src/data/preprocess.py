"""
Data preprocessing module for Iris dataset.
Handles data validation, cleaning, and feature engineering.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IrisPreprocessor:
    """
    Preprocessor for Iris dataset with fit/transform pattern.
    """

    def __init__(self, scale_features: bool = True):
        """
        Initialize preprocessor.

        Args:
            scale_features: Whether to apply standard scaling
        """
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.feature_columns = None
        self._is_fitted = False

    def validate_data(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Validate input data format and content.

        Args:
            X: Features DataFrame
            y: Target Series (optional)

        Raises:
            ValueError: If data validation fails
        """
        # Check for missing values
        if X.isnull().any().any():
            raise ValueError("Input data contains missing values")

        # Check for expected columns (excluding 'species' which is for reference)
        expected_features = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        missing_cols = set(expected_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate numeric types
        numeric_cols = X[expected_features]
        if not all(numeric_cols.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All feature columns must be numeric")

        # Validate value ranges (basic sanity checks)
        # Only check for negative values if data hasn't been scaled yet
        # Scaled data can have negative values (mean-centered)
        if not self._is_fitted:  # Only validate raw data, not scaled data
            if (numeric_cols < 0).any().any():
                raise ValueError("Features contain negative values")

        if (numeric_cols > 100).any().any():
            logger.warning("Some feature values are unusually large (>100)")

        # Validate target if provided
        if y is not None:
            if len(X) != len(y):
                raise ValueError(
                    f"Feature and target length mismatch: {len(X)} vs {len(y)}"
                )

            unique_targets = y.nunique()
            if unique_targets != 3:
                raise ValueError(f"Expected 3 classes, found {unique_targets}")

        logger.info("Data validation passed")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "IrisPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            X: Training features
            y: Training target (optional, for validation)

        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting preprocessor")

        # Validate data
        self.validate_data(X, y)

        # Store feature columns
        self.feature_columns = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        # Fit scaler if enabled
        if self.scale_features and self.scaler is not None:
            X_numeric = X[self.feature_columns]
            self.scaler.fit(X_numeric)
            logger.info("Scaler fitted on training data")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input data using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            np.ndarray: Transformed features

        Raises:
            RuntimeError: If preprocessor hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform. Call fit() first."
            )

        logger.info("Transforming data")

        # Validate input
        self.validate_data(X)

        # Extract numeric features
        X_numeric = X[self.feature_columns]

        # Apply scaling if enabled
        if self.scale_features and self.scaler is not None:
            X_transformed = self.scaler.transform(X_numeric)
            logger.info("Features scaled")
        else:
            X_transformed = X_numeric.values
            logger.info("Features extracted (no scaling)")

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.

        Args:
            X: Training features
            y: Training target (optional)

        Returns:
            np.ndarray: Transformed features
        """
        return self.fit(X, y).transform(X)


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, IrisPreprocessor]:
    """
    Prepare train/test split with preprocessing.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        scale_features: Whether to scale features

    Returns:
        Tuple containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training targets
            - y_test: Test targets
            - preprocessor: Fitted preprocessor
    """
    logger.info(
        f"Preparing train/test split (test_size={test_size}, "
        f"random_state={random_state})"
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Split sizes - Train: {len(X_train)}, Test: {len(X_test)}")

    # Initialize and fit preprocessor on training data only
    preprocessor = IrisPreprocessor(scale_features=scale_features)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)

    # Transform test data using fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train.values,
        y_test.values,
        preprocessor,
    )
