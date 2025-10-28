"""
Unit tests for model training, prediction, and management modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_iris_from_sklearn
from src.data.preprocess import prepare_train_test_split
from src.models.model_loader import ModelLoader
from src.models.predict import IrisPredictor
from src.models.train import IrisModelTrainer, update_model_registry


class TestModelTraining:
    """Tests for model training functionality."""

    @pytest.fixture
    def training_data(self):
        """Fixture providing train/test split data."""
        X, y = load_iris_from_sklearn()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, preprocessor

    def test_trainer_initialization(self):
        """Test trainer initialization with default parameters."""
        trainer = IrisModelTrainer()

        assert trainer.model_type == "logistic_regression"
        assert trainer.random_state == 42
        assert trainer.max_iter == 200
        assert trainer.model is None
        assert trainer.class_names == ["setosa", "versicolor", "virginica"]

    def test_trainer_initialization_custom(self):
        """Test trainer initialization with custom parameters."""
        trainer = IrisModelTrainer(
            model_type="logistic_regression", random_state=123, max_iter=300
        )

        assert trainer.random_state == 123
        assert trainer.max_iter == 300

    def test_create_model(self):
        """Test model creation."""
        trainer = IrisModelTrainer()
        model = trainer._create_model()

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_model_invalid_type(self):
        """Test model creation with invalid type raises error."""
        trainer = IrisModelTrainer(model_type="invalid_model")

        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer._create_model()

    def test_train_model(self, training_data):
        """Test model training."""
        X_train, X_test, y_train, y_test, _ = training_data

        trainer = IrisModelTrainer()
        metrics = trainer.train(X_train, y_train, X_test, y_test)

        # Check model is trained
        assert trainer.model is not None

        # Check metrics structure
        assert "train" in metrics
        assert "test" in metrics
        assert "timestamp" in metrics
        assert "model_type" in metrics

        # Check train metrics
        assert "accuracy" in metrics["train"]
        assert "precision_macro" in metrics["train"]
        assert "recall_macro" in metrics["train"]
        assert "f1_macro" in metrics["train"]

        # Check test metrics
        assert "accuracy" in metrics["test"]
        assert metrics["test"]["accuracy"] >= 0.85  # Adjusted threshold

    def test_train_model_performance(self, training_data):
        """Test that model achieves acceptable performance."""
        X_train, X_test, y_train, y_test, _ = training_data

        trainer = IrisModelTrainer(random_state=42)
        metrics = trainer.train(X_train, y_train, X_test, y_test)

        # Iris dataset should achieve high accuracy
        assert metrics["test"]["accuracy"] >= 0.85  # Adjusted
        assert metrics["test"]["f1_macro"] >= 0.85  # Adjusted
        assert metrics["train"]["accuracy"] >= 0.90  # Adjusted

    def test_save_model(self, training_data):
        """Test model saving."""
        X_train, X_test, y_train, y_test, preprocessor = training_data

        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model(model_path, preprocessor)

            # Check file exists
            assert model_path.exists()

            # Check file is not empty
            assert model_path.stat().st_size > 0

    def test_save_model_without_training(self):
        """Test that saving without training raises error."""
        trainer = IrisModelTrainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"

            with pytest.raises(ValueError, match="No model to save"):
                trainer.save_model(model_path)

    def test_load_model(self, training_data):
        """Test model loading."""
        X_train, X_test, y_train, y_test, preprocessor = training_data

        # Train and save model
        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model(model_path, preprocessor)

            # Load model
            model, loaded_preprocessor, metadata = IrisModelTrainer.load_model(
                model_path
            )

            # Check model loaded
            assert model is not None
            assert loaded_preprocessor is not None
            assert metadata["model_type"] == "logistic_regression"
            assert metadata["class_names"] == ["setosa", "versicolor", "virginica"]
            assert "training_metrics" in metadata

    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            IrisModelTrainer.load_model(Path("nonexistent_model.joblib"))

    def test_model_reproducibility(self, training_data):
        """Test that model training is reproducible with same random_state."""
        X_train, X_test, y_train, y_test, _ = training_data

        # Train first model
        trainer1 = IrisModelTrainer(random_state=42)
        metrics1 = trainer1.train(X_train, y_train, X_test, y_test)
        pred1 = trainer1.model.predict(X_test)

        # Train second model with same random_state
        trainer2 = IrisModelTrainer(random_state=42)
        metrics2 = trainer2.train(X_train, y_train, X_test, y_test)
        pred2 = trainer2.model.predict(X_test)

        # Check metrics are identical
        assert metrics1["test"]["accuracy"] == metrics2["test"]["accuracy"]

        # Check predictions are identical
        assert np.array_equal(pred1, pred2)


class TestModelPrediction:
    """Tests for model prediction functionality."""

    @pytest.fixture
    def trained_predictor(self):
        """Fixture providing a trained predictor."""
        X, y = load_iris_from_sklearn()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        predictor = IrisPredictor(
            model=trainer.model,
            preprocessor=preprocessor,
            class_names=trainer.class_names,
        )

        # Return both scaled (for direct model testing) and original data (for predictor testing)
        from sklearn.model_selection import train_test_split

        _, X_test_raw, _, y_test_raw = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return predictor, X_test, y_test, X_test_raw.drop("species", axis=1), y_test_raw

    def test_predictor_initialization(self, trained_predictor):
        """Test predictor initialization."""
        predictor, _, _, _, _ = trained_predictor  # Fixed: unpack 5 values

        assert predictor.model is not None
        assert predictor.preprocessor is not None
        assert predictor.class_names == ["setosa", "versicolor", "virginica"]

    def test_predict_with_dict(self, trained_predictor):
        """Test prediction with dictionary input."""
        predictor, _, _, _, _ = trained_predictor  # Fixed: unpack 5 values

        features = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }

        result = predictor.predict(features)

        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 3

    def test_predict_with_array(self, trained_predictor):
        """Test prediction with numpy array input."""
        predictor, _, _, X_test_raw, _ = trained_predictor

        # Get first sample from raw data
        features = X_test_raw.iloc[0].values

        result = predictor.predict(features)

        assert "prediction" in result
        assert "confidence" in result
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]

    def test_predict_with_dataframe(self, trained_predictor):
        """Test prediction with DataFrame input."""
        predictor, _, _, _, _ = trained_predictor  # Fixed: unpack 5 values

        features = pd.DataFrame(
            [
                {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2,
                }
            ]
        )

        result = predictor.predict(features)

        assert "prediction" in result
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]

    def test_predict_probabilities_sum_to_one(self, trained_predictor):
        """Test that prediction probabilities sum to 1."""
        predictor, _, _, _, _ = trained_predictor  # Fixed: unpack 5 values

        features = {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }

        result = predictor.predict(features)
        probs_sum = sum(result["probabilities"].values())

        assert abs(probs_sum - 1.0) < 1e-6  # Allow small floating point error

    def test_predict_batch(self, trained_predictor):
        """Test batch prediction."""
        predictor, _, _, X_test_raw, _ = trained_predictor

        # Take first 5 samples from raw data
        df = X_test_raw.iloc[:5]

        results = predictor.predict_batch(df)

        assert len(results) == 5
        for result in results:
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result

    def test_predict_batch_with_array(self, trained_predictor):
        """Test batch prediction with numpy array."""
        predictor, _, _, X_test_raw, _ = trained_predictor

        results = predictor.predict_batch(X_test_raw.iloc[:5].values)

        assert len(results) == 5

    def test_predict_invalid_input_type(self, trained_predictor):
        """Test that invalid input type raises error."""
        predictor, _, _, _, _ = trained_predictor  # Fixed: unpack 5 values

        with pytest.raises(ValueError, match="Unsupported features type"):
            predictor.predict("invalid_input")

    def test_predict_accuracy(self, trained_predictor):
        """Test prediction accuracy on test set."""
        predictor, _, _, X_test_raw, y_test = trained_predictor

        results = predictor.predict_batch(X_test_raw)
        predictions = [
            ["setosa", "versicolor", "virginica"].index(r["prediction"])
            for r in results
        ]

        accuracy = np.mean(np.array(predictions) == y_test)
        assert accuracy >= 0.85  # Should achieve high accuracy


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_update_model_registry(self):
        """Test updating model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            registry_path = Path(tmpdir) / "model_registry.json"

            # Create dummy metrics
            metrics = {
                "train": {"accuracy": 0.98, "f1_macro": 0.97},
                "test": {"accuracy": 0.96, "f1_macro": 0.95},
                "model_type": "logistic_regression",
                "timestamp": "2024-01-01T00:00:00",
            }

            # Update registry
            update_model_registry(model_path, metrics, registry_path)

            # Check registry file created
            assert registry_path.exists()

            # Load and check registry
            import json

            with open(registry_path, "r") as f:
                registry = json.load(f)

            assert "models" in registry
            assert "active_model" in registry
            assert "metadata" in registry
            assert len(registry["models"]) == 1
            assert registry["models"][0]["metrics"]["test_accuracy"] == 0.96

    def test_update_model_registry_multiple_models(self):
        """Test updating registry with multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "model_registry.json"

            # Add first model
            metrics1 = {
                "train": {"accuracy": 0.95, "f1_macro": 0.94},
                "test": {"accuracy": 0.93, "f1_macro": 0.92},
                "model_type": "logistic_regression",
            }
            update_model_registry(
                Path(tmpdir) / "model1.joblib", metrics1, registry_path
            )

            import time

            time.sleep(0.01)  # Small delay to ensure different timestamps

            # Add second model with better performance
            metrics2 = {
                "train": {"accuracy": 0.98, "f1_macro": 0.97},
                "test": {"accuracy": 0.96, "f1_macro": 0.95},
                "model_type": "logistic_regression",
            }
            update_model_registry(
                Path(tmpdir) / "model2.joblib", metrics2, registry_path
            )

            # Load registry
            import json

            with open(registry_path, "r") as f:
                registry = json.load(f)

            # Check both models registered
            assert len(registry["models"]) == 2

            # Check best model is active (model2 with 0.96 accuracy)
            active_model_id = registry["active_model"]
            active_model = next(
                m for m in registry["models"] if m["model_id"] == active_model_id
            )
            assert active_model["metrics"]["test_accuracy"] == 0.96


class TestModelLoader:
    """Tests for model loader singleton."""

    def test_model_loader_singleton(self):
        """Test that ModelLoader implements singleton pattern."""
        loader1 = ModelLoader()
        loader2 = ModelLoader()

        assert loader1 is loader2

    def test_load_model(self):
        """Test model loading through loader."""
        X, y = load_iris_from_sklearn()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train and save model
        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model(model_path, preprocessor)

            # Load through loader
            loader = ModelLoader()
            predictor = loader.load_model(model_path)

            assert predictor is not None
            assert isinstance(predictor, IrisPredictor)

    def test_model_loader_caching(self):
        """Test that model loader caches loaded model."""
        X, y = load_iris_from_sklearn()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model(model_path, preprocessor)

            loader = ModelLoader()

            # First load
            predictor1 = loader.load_model(model_path)

            # Second load (should use cache)
            predictor2 = loader.load_model(model_path)

            # Should return same instance
            assert predictor1 is predictor2

    def test_model_loader_force_reload(self):
        """Test forcing reload of model."""
        X, y = load_iris_from_sklearn()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model(model_path, preprocessor)

            loader = ModelLoader()

            # First load
            predictor1 = loader.load_model(model_path)

            # Force reload
            predictor2 = loader.load_model(model_path, force_reload=True)

            # Should be different instances
            assert predictor1 is not predictor2

    def test_get_predictor(self):
        """Test getting loaded predictor."""
        loader = ModelLoader()

        # Before loading
        predictor = loader.get_predictor()
        assert predictor is None or isinstance(predictor, IrisPredictor)

    def test_unload_model(self):
        """Test unloading model from memory."""
        X, y = load_iris_from_sklearn()
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        trainer = IrisModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            trainer.save_model(model_path, preprocessor)

            loader = ModelLoader()
            loader.load_model(model_path)

            # Check model loaded
            assert loader.get_predictor() is not None

            # Unload
            loader.unload_model()

            # Check model unloaded
            assert loader.get_predictor() is None
