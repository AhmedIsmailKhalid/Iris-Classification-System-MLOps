"""
Synthetic data generation for testing drift detection.
"""

import logging
from typing import Dict, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic iris data for drift simulation.
    """

    def __init__(self):
        # Reference statistics from original Iris dataset
        self.feature_stats = {
            "sepal length (cm)": {"mean": 5.84, "std": 0.83},
            "sepal width (cm)": {"mean": 3.05, "std": 0.43},
            "petal length (cm)": {"mean": 3.76, "std": 1.76},
            "petal width (cm)": {"mean": 1.20, "std": 0.76},
        }

        self.class_distribution = {
            "setosa": 0.33,
            "versicolor": 0.33,
            "virginica": 0.34,
        }

    def generate_normal_data(self, n_samples: int = 50) -> pd.DataFrame:
        """
        Generate synthetic data following original distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic samples
        """
        logger.info(f"Generating {n_samples} normal samples")

        samples = []

        for _ in range(n_samples):
            # Generate features from normal distribution
            sample = {}
            for feature, stats in self.feature_stats.items():
                value = np.random.normal(stats["mean"], stats["std"])
                # Clip to reasonable bounds
                value = max(0.1, value)  # No negative values
                sample[feature] = round(value, 1)

            # Assign class based on distribution
            sample["species"] = np.random.choice(
                list(self.class_distribution.keys()),
                p=list(self.class_distribution.values()),
            )

            samples.append(sample)

        df = pd.DataFrame(samples)
        logger.info(f"Generated {len(df)} normal samples")

        return df

    def generate_drifted_data(
        self,
        n_samples: int = 50,
        drift_type: Literal["shift", "scale", "extreme"] = "shift",
        drift_magnitude: float = 2.0,
    ) -> pd.DataFrame:
        """
        Generate synthetic data with distribution drift.

        Args:
            n_samples: Number of samples to generate
            drift_type: Type of drift to introduce
                - 'shift': Mean shift
                - 'scale': Variance change
                - 'extreme': Introduce outliers
            drift_magnitude: How much drift to introduce (multiplier)

        Returns:
            DataFrame with drifted samples
        """
        logger.info(
            f"Generating {n_samples} drifted samples "
            f"(type={drift_type}, magnitude={drift_magnitude})"
        )

        samples = []

        for _ in range(n_samples):
            sample = {}

            for feature, stats in self.feature_stats.items():
                if drift_type == "shift":
                    # Shift mean
                    mean = stats["mean"] + (stats["std"] * drift_magnitude)
                    std = stats["std"]
                elif drift_type == "scale":
                    # Increase variance
                    mean = stats["mean"]
                    std = stats["std"] * drift_magnitude
                elif drift_type == "extreme":
                    # Mix of normal and extreme values
                    if np.random.random() < 0.3:  # 30% outliers
                        mean = stats["mean"] + (stats["std"] * drift_magnitude * 3)
                        std = stats["std"] * 0.5
                    else:
                        mean = stats["mean"]
                        std = stats["std"]
                else:
                    mean = stats["mean"]
                    std = stats["std"]

                value = np.random.normal(mean, std)
                value = max(0.1, value)
                sample[feature] = round(value, 1)

            # Drifted class distribution (shift toward virginica)
            sample["species"] = np.random.choice(
                ["setosa", "versicolor", "virginica"], p=[0.15, 0.25, 0.60]
            )

            samples.append(sample)

        df = pd.DataFrame(samples)
        logger.info(f"Generated {len(df)} drifted samples")

        return df

    def get_feature_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate feature statistics for a dataset.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary of feature statistics
        """
        stats = {}

        for feature in self.feature_stats.keys():
            if feature in data.columns:
                stats[feature] = {
                    "mean": float(data[feature].mean()),
                    "std": float(data[feature].std()),
                    "min": float(data[feature].min()),
                    "max": float(data[feature].max()),
                    "median": float(data[feature].median()),
                }

        return stats
