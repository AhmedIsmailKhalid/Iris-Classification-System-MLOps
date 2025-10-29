"""
Data logging for monitoring and retraining.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLogger:
    """
    Log prediction data for monitoring and retraining.
    """

    def __init__(self, log_dir: Path = Path("data/monitoring")):
        """
        Initialize data logger.

        Args:
            log_dir: Directory to store logged data
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.prediction_log_path = self.log_dir / "predictions.csv"
        self.new_data_path = self.log_dir / "new_data.csv"

        logger.info(f"DataLogger initialized: {self.log_dir}")

    def log_prediction(
        self,
        features: Dict[str, float],
        prediction: str,
        confidence: float,
        user_feedback: Optional[str] = None,
    ) -> None:
        """
        Log a single prediction.

        Args:
            features: Input features
            prediction: Model prediction
            confidence: Prediction confidence
            user_feedback: Optional user feedback on correctness
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **features,
            "prediction": prediction,
            "confidence": confidence,
            "user_feedback": user_feedback,
        }

        # Append to CSV
        df = pd.DataFrame([log_entry])

        if self.prediction_log_path.exists():
            df.to_csv(self.prediction_log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.prediction_log_path, index=False)

        logger.info(f"Logged prediction: {prediction} (confidence: {confidence:.2f})")

    def save_new_data(self, data: pd.DataFrame) -> None:
        """
        Save new data for potential retraining.

        Args:
            data: New data samples
        """
        if self.new_data_path.exists():
            # Append to existing data
            existing = pd.read_csv(self.new_data_path)
            combined = pd.concat([existing, data], ignore_index=True)
            combined.to_csv(self.new_data_path, index=False)
        else:
            data.to_csv(self.new_data_path, index=False)

        logger.info(f"Saved {len(data)} new samples to {self.new_data_path}")

    def load_new_data(self) -> Optional[pd.DataFrame]:
        """
        Load accumulated new data.

        Returns:
            DataFrame of new data, or None if no data exists
        """
        if not self.new_data_path.exists():
            logger.info("No new data found")
            return None

        df = pd.read_csv(self.new_data_path)
        logger.info(f"Loaded {len(df)} new samples")
        return df

    def clear_new_data(self) -> None:
        """
        Clear new data after retraining.
        """
        if self.new_data_path.exists():
            self.new_data_path.unlink()
            logger.info("Cleared new data")

    def get_prediction_count(self) -> int:
        """
        Get total number of logged predictions.

        Returns:
            Number of predictions
        """
        if not self.prediction_log_path.exists():
            return 0

        df = pd.read_csv(self.prediction_log_path)
        return len(df)
