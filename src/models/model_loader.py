"""
Model loader module implementing singleton pattern.
Ensures only one model is loaded in memory for API serving.
"""

import logging
from pathlib import Path
from typing import Optional

from src.models.predict import IrisPredictor
from src.models.train import IrisModelTrainer

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton model loader for production serving.
    """
    
    _instance: Optional['ModelLoader'] = None
    _predictor: Optional[IrisPredictor] = None
    _model_path: Optional[Path] = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: Path, force_reload: bool = False) -> IrisPredictor:
        """
        Load model from disk (singleton).
        
        Args:
            model_path: Path to saved model
            force_reload: Force reload even if model already loaded
            
        Returns:
            IrisPredictor instance
        """
        # Check if model already loaded
        if self._predictor is not None and \
           self._model_path == model_path and \
           not force_reload:
            logger.info("Using cached model")
            return self._predictor
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model, preprocessor, metadata = IrisModelTrainer.load_model(model_path)
        
        # Create predictor
        self._predictor = IrisPredictor(
            model=model,
            preprocessor=preprocessor,
            class_names=metadata["class_names"]
        )
        
        self._model_path = model_path
        
        logger.info(
            f"Model loaded successfully. Type: {metadata['model_type']}, "
            f"Test Accuracy: {metadata['training_metrics']['test']['accuracy']:.4f}"
        )
        
        return self._predictor
    
    def get_predictor(self) -> Optional[IrisPredictor]:
        """
        Get loaded predictor instance.
        
        Returns:
            IrisPredictor if loaded, None otherwise
        """
        return self._predictor
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        self._predictor = None
        self._model_path = None
        logger.info("Model unloaded from memory")


# Global instance
model_loader = ModelLoader()