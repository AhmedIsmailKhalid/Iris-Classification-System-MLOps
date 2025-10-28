"""
Data loading module for Iris dataset.
Handles loading data from sklearn and saving/loading from CSV.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris

logger = logging.getLogger(__name__)


def load_iris_from_sklearn() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset from scikit-learn.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series

    Raises:
        RuntimeError: If dataset cannot be loaded
    """
    try:
        logger.info("Loading Iris dataset from scikit-learn")
        iris = load_iris(as_frame=True)

        # Get features and target
        X = iris.data
        y = iris.target

        # Add target names as a column for better interpretability
        X["species"] = iris.target_names[y]

        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        return X, y

    except Exception as e:
        logger.error(f"Failed to load Iris dataset: {str(e)}")
        raise RuntimeError(f"Could not load Iris dataset: {str(e)}")


def save_iris_to_csv(
    X: pd.DataFrame, y: pd.Series, output_path: Path = Path("data/raw/iris.csv")
) -> None:
    """
    Save Iris dataset to CSV file.

    Args:
        X: Features DataFrame
        y: Target Series
        output_path: Path where CSV will be saved

    Raises:
        IOError: If file cannot be written
    """
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combine features and target
        df = X.copy()
        df["target"] = y

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save dataset to CSV: {str(e)}")
        raise IOError(f"Could not save dataset: {str(e)}")


def load_iris_from_csv(
    input_path: Path = Path("data/raw/iris.csv"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Iris dataset from CSV file.

    Args:
        input_path: Path to CSV file

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV has invalid format
    """
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"CSV file not found at {input_path}")

        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)

        # Validate expected columns
        expected_cols = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "species",
            "target",
        ]

        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"CSV missing required columns. Expected: {expected_cols}")

        # Separate features and target
        X = df.drop("target", axis=1)
        y = df["target"]

        logger.info(f"Loaded {len(X)} samples from CSV")
        return X, y

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to load dataset from CSV: {str(e)}")
        raise ValueError(f"Could not load dataset from CSV: {str(e)}")


def get_iris_data(use_csv: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Main function to get Iris dataset.

    Args:
        use_csv: If True, load from CSV (if exists), else from sklearn

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series
    """
    csv_path = Path("data/raw/iris.csv")

    if use_csv and csv_path.exists():
        logger.info("Loading from existing CSV file")
        return load_iris_from_csv(csv_path)
    else:
        logger.info("Loading from scikit-learn and saving to CSV")
        X, y = load_iris_from_sklearn()
        save_iris_to_csv(X, y, csv_path)
        return X, y
