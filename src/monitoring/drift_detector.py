"""
Data drift detection using statistical tests.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect data drift using statistical tests.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize drift detector.

        Args:
            significance_level: P-value threshold for detecting drift
        """
        self.significance_level = significance_level

    def kolmogorov_smirnov_test(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    def population_stability_index(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1)).tolist()

        # Ensure unique breakpoints
        breakpoints = sorted(set(breakpoints))

        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        curr_counts, _ = np.histogram(current, bins=breakpoints)

        # Add small constant to avoid division by zero
        epsilon = 1e-10
        ref_percents = (ref_counts + epsilon) / (
            len(reference) + epsilon * len(breakpoints)
        )
        curr_percents = (curr_counts + epsilon) / (
            len(current) + epsilon * len(breakpoints)
        )

        # Calculate PSI
        psi = np.sum(
            (curr_percents - ref_percents) * np.log(curr_percents / ref_percents)
        )

        return float(psi)

    def detect_feature_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame, feature: str
    ) -> Dict:
        """
        Detect drift for a single feature.

        Args:
            reference: Reference dataset
            current: Current dataset
            feature: Feature name to check

        Returns:
            Dictionary with drift detection results
        """
        ref_data = reference[feature].dropna().values
        curr_data = current[feature].dropna().values

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = self.kolmogorov_smirnov_test(ref_data, curr_data)

        # PSI
        psi = self.population_stability_index(ref_data, curr_data)

        # Statistical comparison
        ref_mean = float(np.mean(ref_data))
        curr_mean = float(np.mean(curr_data))
        mean_diff = abs(curr_mean - ref_mean)
        mean_diff_percent = (mean_diff / ref_mean) * 100 if ref_mean != 0 else 0

        # Drift decision
        drift_detected = (
            ks_pvalue < self.significance_level or psi >= 0.2 or mean_diff_percent > 20
        )

        return {
            "feature": feature,
            "drift_detected": drift_detected,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "psi": psi,
            "reference_mean": ref_mean,
            "current_mean": curr_mean,
            "mean_difference_percent": mean_diff_percent,
        }

    def detect_dataset_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> Dict:
        """
        Detect drift across entire dataset.

        Args:
            reference: Reference dataset
            current: Current dataset

        Returns:
            Dictionary with comprehensive drift report
        """
        logger.info("Starting drift detection")

        feature_columns = [
            col
            for col in reference.columns
            if col
            in [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
        ]

        feature_drifts = []
        drifted_features = []

        for feature in feature_columns:
            if feature in current.columns:
                drift_result = self.detect_feature_drift(reference, current, feature)
                feature_drifts.append(drift_result)

                if drift_result["drift_detected"]:
                    drifted_features.append(feature)
                    logger.warning(
                        f"Drift detected in {feature}: "
                        f"PSI={drift_result['psi']:.3f}, "
                        f"p-value={drift_result['ks_pvalue']:.4f}"
                    )

        # Overall drift assessment
        drift_detected = len(drifted_features) > 0
        drift_severity = len(drifted_features) / len(feature_columns)

        # Recommendation
        if drift_severity >= 0.5:
            recommendation = "immediate_retraining"
        elif drift_severity >= 0.25:
            recommendation = "schedule_retraining"
        else:
            recommendation = "monitor"

        result = {
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
            "drift_severity": drift_severity,
            "recommendation": recommendation,
            "feature_drift_details": feature_drifts,
            "reference_samples": len(reference),
            "current_samples": len(current),
        }

        logger.info(
            f"Drift detection complete: "
            f"drift_detected={drift_detected}, "
            f"severity={drift_severity:.2f}, "
            f"recommendation={recommendation}"
        )

        return result
