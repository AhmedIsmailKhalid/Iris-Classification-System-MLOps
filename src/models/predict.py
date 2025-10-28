"""
Prediction module for Iris classification.
Handles inference and prediction validation.
"""

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IrisPredictor:
    """
    Predictor class for making predictions with trained Iris model.
    """

    def __init__(self, model, preprocessor, class_names: List[str]):
        """
        Initialize predictor.

        Args:
            model: Trained sklearn model
            preprocessor: Fitted IrisPreprocessor
            class_names: List of class names in order
        """
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names

        logger.info("Predictor initialized")

    def predict(
        self, features: Union[Dict[str, float], pd.DataFrame, np.ndarray]
    ) -> Dict[str, Union[str, float, List[float]]]:
        """
        Make prediction for given features.

        Args:
            features: Input features as dict, DataFrame, or array

        Returns:
            Dictionary containing prediction, confidence, and probabilities
        """
        # Convert input to DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            df = pd.DataFrame(
                features,
                columns=[
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ],
            )
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")

        # Add species column if not present (required by preprocessor)
        if "species" not in df.columns:
            df["species"] = "unknown"

        # Validate and preprocess
        try:
            X_processed = self.preprocessor.transform(df)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ValueError(f"Invalid input features: {str(e)}")

        # Make prediction
        prediction_idx = self.model.predict(X_processed)[0]
        prediction_proba = self.model.predict_proba(X_processed)[0]

        # Get class name and confidence
        predicted_class = self.class_names[prediction_idx]
        confidence = float(prediction_proba[prediction_idx])

        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, prediction_proba)
            },
        }

        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")

        return result

    def predict_batch(
        self, features: Union[pd.DataFrame, np.ndarray]
    ) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        Make predictions for multiple samples.

        Args:
            features: Multiple input features as DataFrame or array

        Returns:
            List of prediction dictionaries
        """
        # Convert to DataFrame if needed
        if isinstance(features, np.ndarray):
            df = pd.DataFrame(
                features,
                columns=[
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ],
            )
        else:
            df = features.copy()

        # Add species column if not present
        if "species" not in df.columns:
            df["species"] = "unknown"

        # Preprocess
        X_processed = self.preprocessor.transform(df)

        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)

        # Format results
        results = []
        for pred_idx, proba in zip(predictions, probabilities):
            predicted_class = self.class_names[pred_idx]
            confidence = float(proba[pred_idx])

            result = {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, proba)
                },
            }
            results.append(result)

        logger.info(f"Batch prediction completed for {len(results)} samples")

        return results
