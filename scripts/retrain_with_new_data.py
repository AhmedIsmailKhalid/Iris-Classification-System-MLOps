"""
Script to retrain model with accumulated new data.
Triggered by GitHub Actions or manually.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.load_data import load_iris_from_sklearn
from src.data.preprocess import prepare_train_test_split
from src.models.train import IrisModelTrainer, update_model_registry
from src.monitoring.data_logger import DataLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retrain_model(min_samples: int = 30, force: bool = False) -> bool:
    """
    Retrain model with new data if conditions are met.
    
    Args:
        min_samples: Minimum new samples required for retraining
        force: Force retraining even if not enough samples
        
    Returns:
        True if retraining was performed, False otherwise
    """
    logger.info("="*80)
    logger.info("AUTOMATED MODEL RETRAINING")
    logger.info("="*80)
    
    # Initialize data logger
    data_logger = DataLogger()
    
    # Load new data
    new_data = data_logger.load_new_data()
    
    if new_data is None or len(new_data) == 0:
        logger.info("❌ No new data available for retraining")
        return False
    
    logger.info(f"📊 Found {len(new_data)} new samples")
    
    # Check if we have enough samples
    if len(new_data) < min_samples and not force:
        logger.info(f"⚠️  Not enough samples for retraining (need {min_samples}, have {len(new_data)})")
        return False
    
    logger.info(f"✅ Proceeding with retraining (force={force})")
    
    # Load original data
    logger.info("Loading original Iris dataset...")
    original_X, original_y = load_iris_from_sklearn()
    
    # Merge with new data
    logger.info("Merging original and new data...")
    
    # Ensure new data has the same structure
    if 'species' not in new_data.columns:
        logger.warning("New data missing 'species' column - cannot use for supervised retraining")
        # For now, we'll assign random species based on features (simplified)
        # In production, you'd need labeled data
        def assign_species(row):
            if row['petal length (cm)'] < 2.5:
                return 'setosa'
            elif row['petal length (cm)'] < 5.0:
                return 'versicolor'
            else:
                return 'virginica'
        
        new_data['species'] = new_data.apply(assign_species, axis=1)
        logger.info("Assigned pseudo-labels based on feature rules")
    
    # Combine datasets
    combined_X = pd.concat([original_X, new_data], ignore_index=True)
    
    # Extract target
    combined_y = combined_X['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
    
    logger.info(f"Combined dataset: {len(combined_X)} samples (original: {len(original_X)}, new: {len(new_data)})")
    
    # Prepare train/test split
    logger.info("Preparing train/test split...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
        combined_X, combined_y, test_size=0.2, random_state=42
    )
    
    # Train model
    logger.info("Training new model...")
    trainer = IrisModelTrainer(random_state=42)
    metrics = trainer.train(X_train, y_train, X_test, y_test)
    
    # Log results
    logger.info("="*80)
    logger.info("TRAINING RESULTS")
    logger.info("="*80)
    logger.info(f"Train Accuracy: {metrics['train']['accuracy']:.4f}")
    logger.info(f"Test Accuracy:  {metrics['test']['accuracy']:.4f}")
    logger.info(f"F1 Score:       {metrics['test']['f1_macro']:.4f}")
    
    # Save model
    model_path = Path("models/iris_classifier.joblib")
    logger.info(f"Saving model to {model_path}...")
    trainer.save_model(model_path, preprocessor)
    
    # Update registry
    logger.info("Updating model registry...")
    update_model_registry(model_path, metrics)
    
    # Clear new data after successful retraining
    logger.info("Clearing processed new data...")
    data_logger.clear_new_data()
    
    logger.info("="*80)
    logger.info("✅ RETRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return True


def main():
    """Main entry point for retraining script."""
    parser = argparse.ArgumentParser(description='Retrain Iris classifier with new data')
    parser.add_argument(
        '--min-samples',
        type=int,
        default=30,
        help='Minimum number of new samples required for retraining'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if not enough samples'
    )
    
    args = parser.parse_args()
    
    try:
        success = retrain_model(min_samples=args.min_samples, force=args.force)
        
        if success:
            logger.info("Model retraining completed successfully")
            sys.exit(0)
        else:
            logger.info("Model retraining skipped")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()