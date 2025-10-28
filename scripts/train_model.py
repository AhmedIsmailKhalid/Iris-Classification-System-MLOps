"""
Training script for Iris classification model.
Run this script to train and save a new model.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import get_iris_data
from src.data.preprocess import prepare_train_test_split
from src.models.train import IrisModelTrainer, update_model_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    logger.info("=" * 80)
    logger.info("Starting Iris Model Training Pipeline")
    logger.info("=" * 80)
    
    try:
        # 1. Load data
        logger.info("\n[1/5] Loading data...")
        X, y = get_iris_data(use_csv=True)
        logger.info(f"Loaded {len(X)} samples")
        
        # 2. Prepare train/test split
        logger.info("\n[2/5] Preparing train/test split...")
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            scale_features=True
        )
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # 3. Train model
        logger.info("\n[3/5] Training model...")
        trainer = IrisModelTrainer(
            model_type="logistic_regression",
            random_state=42,
            max_iter=200
        )
        
        metrics = trainer.train(X_train, y_train, X_test, y_test)
        
        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Train Accuracy: {metrics['train']['accuracy']:.4f}")
        logger.info(f"Train F1 (macro): {metrics['train']['f1_macro']:.4f}")
        logger.info(f"Test Accuracy:  {metrics['test']['accuracy']:.4f}")
        logger.info(f"Test F1 (macro): {metrics['test']['f1_macro']:.4f}")
        logger.info("=" * 80)
        
        # 4. Save model
        logger.info("\n[4/5] Saving model...")
        model_path = Path("models/iris_classifier.joblib")
        trainer.save_model(model_path, preprocessor)
        logger.info(f"Model saved to {model_path}")
        
        # 5. Update model registry
        logger.info("\n[5/5] Updating model registry...")
        update_model_registry(model_path, metrics)
        logger.info("Model registry updated")
        
        # Success
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training pipeline completed successfully!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())