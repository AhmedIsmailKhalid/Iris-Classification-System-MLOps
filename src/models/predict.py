"""
Prediction module for Iris classification.
Handles inference and prediction validation with SHAP explanations.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class IrisPredictor:
    """
    Predictor class for making predictions with trained Iris model.
    Includes SHAP explanations for feature contributions.
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
        self.feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        # SHAP explainer - lazy initialization
        # SHAP explainer creation has overhead
        # Not every prediction needs explanations (batch predictions don't), create once when first needed, reuse after that

        self._explainer = None
        self._background_data = None

        logger.info("Predictor initialized")

    def _get_explainer(self):
        """
        Get or create SHAP explainer (lazy initialization).

        Returns:
            SHAP TreeExplainer or KernelExplainer
        """
        if self._explainer is None:
            try:
                # For LogisticRegression, use LinearExplainer (faster)
                from sklearn.linear_model import LogisticRegression

                if isinstance(self.model, LogisticRegression):
                    # Create background dataset (mean values from training)
                    # Using typical Iris dataset mean values
                    background = np.array([[5.8, 3.0, 4.3, 1.3]])
                    self._explainer = shap.LinearExplainer(
                        self.model, background, feature_names=self.feature_names
                    )
                    logger.info("SHAP LinearExplainer created")
                else:
                    # Fallback to KernelExplainer for other models
                    background = np.array([[5.8, 3.0, 4.3, 1.3]])
                    self._explainer = shap.KernelExplainer(
                        self.model.predict_proba, background
                    )
                    logger.info("SHAP KernelExplainer created")

            except Exception as e:
                logger.error(f"Failed to create SHAP explainer: {str(e)}")
                self._explainer = None

        return self._explainer

    def _calculate_shap_values(self, X_processed: np.ndarray) -> Optional[Dict]:
        """
        Calculate SHAP values for processed features.

        Args:
            X_processed: Preprocessed feature array (single sample)

        Returns:
            Dictionary with feature contributions or None if SHAP fails
        """
        try:
            explainer = self._get_explainer()
            if explainer is None:
                return None

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_processed)

            # Handle multi-class output
            # shap_values shape: (n_samples, n_features, n_classes) or (n_samples, n_features)
            if isinstance(shap_values, list):
                # Legacy format: list of arrays per class
                # Get values for predicted class (will determine after prediction)
                return shap_values
            else:
                # New format: single array
                return shap_values

        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            return None

    def predict(
        self,
        features: Union[Dict[str, float], pd.DataFrame, np.ndarray],
        include_shap: bool = True,
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Make prediction for given features.

        Args:
            features: Input features as dict, DataFrame, or array
            include_shap: Whether to include SHAP feature contributions

        Returns:
            Dictionary containing prediction, confidence, probabilities, and optionally SHAP values
        """
        # Convert input to DataFrame
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            df = pd.DataFrame(features, columns=self.feature_names)
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

        # Add SHAP values if requested
        if include_shap:
            shap_values = self._calculate_shap_values(X_processed)

            if shap_values is not None:
                try:
                    # Extract SHAP values for the predicted class
                    if isinstance(shap_values, list):
                        # List format: [class0_shap, class1_shap, class2_shap]
                        class_shap = shap_values[prediction_idx][0]
                    else:
                        # Array format: (n_samples, n_features, n_classes)
                        if shap_values.ndim == 3:
                            class_shap = shap_values[0, :, prediction_idx]
                        else:
                            class_shap = shap_values[0]

                    # Create feature contribution dictionary
                    feature_contributions = {
                        feature: float(shap_val)
                        for feature, shap_val in zip(self.feature_names, class_shap)
                    }

                    # Sort by absolute contribution (most important first)
                    sorted_contributions = dict(
                        sorted(
                            feature_contributions.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )
                    )

                    result["feature_contributions"] = sorted_contributions

                except Exception as e:
                    logger.warning(f"Failed to format SHAP values: {str(e)}")
                    result["feature_contributions"] = None
            else:
                result["feature_contributions"] = None

        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")

        return result

    def predict_batch(
        self, features: Union[pd.DataFrame, np.ndarray], include_shap: bool = False
    ) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        Make predictions for multiple samples.

        Args:
            features: Multiple input features as DataFrame or array
            include_shap: Whether to include SHAP values (disabled by default for performance)

        Returns:
            List of prediction dictionaries
        """
        # Convert to DataFrame if needed
        if isinstance(features, np.ndarray):
            df = pd.DataFrame(features, columns=self.feature_names)
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

            # SHAP for batch is expensive - only if explicitly requested
            if include_shap:
                # Calculate SHAP for this single sample
                shap_result = self._calculate_shap_values(
                    X_processed[len(results) : len(results) + 1]
                )
                if shap_result is not None:
                    try:
                        if isinstance(shap_result, list):
                            class_shap = shap_result[pred_idx][0]
                        else:
                            if shap_result.ndim == 3:
                                class_shap = shap_result[0, :, pred_idx]
                            else:
                                class_shap = shap_result[0]

                        result["feature_contributions"] = dict(
                            sorted(
                                {
                                    feature: float(shap_val)
                                    for feature, shap_val in zip(
                                        self.feature_names, class_shap
                                    )
                                }.items(),
                                key=lambda x: abs(x[1]),
                                reverse=True,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"SHAP failed for batch sample: {str(e)}")
                        result["feature_contributions"] = None

            results.append(result)

        logger.info(f"Batch prediction completed for {len(results)} samples")

        return results
