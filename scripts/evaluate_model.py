"""
Model evaluation script.
Evaluates a saved model on the test set and prints detailed metrics.
"""

import logging
import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import get_iris_data
from src.data.preprocess import prepare_train_test_split
from src.models.train import IrisModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    logger.info("=" * 80)
    logger.info("Model Evaluation Script")
    logger.info("=" * 80)

    try:
        # Load model
        model_path = Path("models/iris_classifier.joblib")
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.info(
                "Please train a model first using: python scripts/train_model.py"
            )
            return 1

        logger.info(f"\nLoading model from {model_path}")
        model, preprocessor, metadata = IrisModelTrainer.load_model(model_path)

        # Print model info
        logger.info(f"\nModel Type: {metadata['model_type']}")
        logger.info(f"Trained: {metadata['timestamp']}")

        # Load and prepare data
        logger.info("\nLoading test data...")
        X, y = get_iris_data(use_csv=True)
        _, X_test, _, y_test, _ = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42, scale_features=True
        )

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)

        # Calculate metrics
        class_names = metadata["class_names"]

        logger.info("\n" + "=" * 80)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 80)
        print(classification_report(y_test, y_pred, target_names=class_names))

        logger.info("=" * 80)
        logger.info("CONFUSION MATRIX")
        logger.info("=" * 80)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"\n{conf_matrix}\n")
        logger.info(f"Classes: {class_names}")

        # Per-class accuracy
        logger.info("=" * 80)
        logger.info("PER-CLASS ACCURACY")
        logger.info("=" * 80)
        for i, class_name in enumerate(class_names):
            class_accuracy = conf_matrix[i, i] / conf_matrix[i].sum()
            logger.info(f"{class_name}: {class_accuracy:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Evaluation completed")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
