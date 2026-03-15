"""
Model training module for Iris classification.
Handles model training, evaluation, and serialization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.preprocess import IrisPreprocessor

logger = logging.getLogger(__name__)


class IrisModelTrainer:
    """
    Trainer class for Iris classification model.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        random_state: int = 42,
        max_iter: int = 200,
    ):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to train (currently only 'logistic_regression')
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for solver convergence
        """
        self.model_type = model_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.preprocessor = None
        self.training_metrics = {}
        self.class_names = ["setosa", "versicolor", "virginica"]

        logger.info(
            f"Initialized {model_type} trainer with random_state={random_state}"
        )

    def _create_model(self) -> LogisticRegression:
        """
        Create the ML model based on model_type.

        Returns:
            Initialized model
        """
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=self.max_iter,
                solver="lbfgs",
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train the model and evaluate on test set.

        Args:
            X_train: Training features (preprocessed)
            y_train: Training targets
            X_test: Test features (preprocessed)
            y_test: Test targets

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        logger.info("Training completed")

        # Evaluate on both train and test sets
        train_metrics = self._evaluate(X_train, y_train, "train")
        test_metrics = self._evaluate(X_test, y_test, "test")

        # Store metrics
        self.training_metrics = {
            "train": train_metrics,
            "test": test_metrics,
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "random_state": self.random_state,
        }

        return self.training_metrics

    def _evaluate(
        self, X: np.ndarray, y: np.ndarray, split_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on given data.

        Args:
            X: Features
            y: True labels
            split_name: Name of the split ('train' or 'test')

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on {split_name} set...")

        # Predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)

        # Get classification report as dict
        report = classification_report(
            y, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
        )

        # Get confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report,
        }

        logger.info(
            f"{split_name.capitalize()} - Accuracy: {accuracy:.4f}, "
            f"F1 (macro): {metrics['f1_macro']:.4f}"
        )

        return metrics

    def save_model(
        self, model_path: Path, preprocessor: IrisPreprocessor = None
    ) -> None:
        """
        Save trained model and preprocessor to disk.

        Args:
            model_path: Path where model will be saved
            preprocessor: Optional preprocessor to save alongside model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Create model directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and preprocessor together
        model_data = {
            "model": self.model,
            "preprocessor": preprocessor,
            "model_type": self.model_type,
            "class_names": self.class_names,
            "training_metrics": self.training_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path: Path) -> Tuple[object, IrisPreprocessor, Dict]:
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)

        return (
            model_data["model"],
            model_data.get("preprocessor"),
            {
                "model_type": model_data.get("model_type"),
                "class_names": model_data.get("class_names"),
                "training_metrics": model_data.get("training_metrics"),
                "timestamp": model_data.get("timestamp"),
            },
        )


def update_model_registry(
    model_path: Path,
    metrics: Dict,
    registry_path: Path = Path("models/model_registry.json"),
) -> None:
    """
    Update model registry with new model information.

    Args:
        model_path: Path to the saved model
        metrics: Training metrics dictionary
        registry_path: Path to model registry JSON
    """
    # Load existing registry
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {
            "models": [],
            "active_model": None,
            "metadata": {
                "dataset": "iris",
                "features": [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ],
                "target_classes": ["setosa", "versicolor", "virginica"],
                "num_classes": 3,
            },
        }

    # Create unique model ID with microseconds to avoid collisions
    timestamp = datetime.now()
    model_id = f"model_{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond}"

    # Create model entry
    model_entry = {
        "model_id": model_id,
        "model_path": str(model_path),
        "timestamp": timestamp.isoformat(),
        "metrics": {
            "test_accuracy": metrics["test"]["accuracy"],
            "test_f1_macro": metrics["test"]["f1_macro"],
            "train_accuracy": metrics["train"]["accuracy"],
        },
        "model_type": metrics.get("model_type", "logistic_regression"),
    }

    # Add to registry
    registry["models"].append(model_entry)

    # Set as active if it's the best model (highest test accuracy)
    if not registry["active_model"]:
        # First model - set as active
        registry["active_model"] = model_entry["model_id"]
    else:
        # Find the model with highest test accuracy
        best_model = max(
            registry["models"], key=lambda m: m["metrics"]["test_accuracy"]
        )
        registry["active_model"] = best_model["model_id"]

    # Save registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Model registry updated: {model_entry['model_id']}")
