# Version 1.0
- Implementeed `create_structure.bat` to create the directory structure
- Implemented `deployment/docker-compose.yml`
- Implemented `deployment/render.yaml`
= Implemented `pyproject.toml`
- Implemented `.gitignore`
- Implemented `.env.example`
- Implemented `src/data/load_data.py`
- Implemented `src/data/preprocess.py`
- Implemented `scripts/generate_iris_data.py`
- Implemented `models/model_registry.json`
- Implemented `tests/test_data.py`
- Implemented `tests/conftest.py`
- Implemented `src/models/train.py`
- Implemented `scripts/train_model.py`
- Implemented `src/models/predict.py`
- Implemented `src/models/model_loader.py`
- Implemented `scripts/evaluate_model.py`
- Implemented `tests/test_model.py`
- **Version 1.0 ZIP File Created**
- Pushed to Github

---

# Version 1.1
- Implemented `src/core/config.py`
- Implemented `src/core/logging.py`
- Implemented `src/api/schemas/iris.py`
- Implemented `src/api/routes/health.py`
- Implemented `src/api/routes/predict.py`
- Implemented `src/api/main.py`
- Implemented `tests/test_api.py`
- **Version 1.1 ZIP File Created**
- Pushed to Github `feature/fastapi-setup` branch
- Created a PR to merge into `develop` branch from `feature/fastapi-setup` branch

---

# Version 1.2
- Implemented `.github/workflows/backend-ci.yml`
- Implemented `.github/workflows/docker-publish.yml`
- Implemented `deployment/Dockerfile`
- Implemented `deployment/docker-compose.yml`
- Implemented `deployment/render.yaml`
- Updated `.dockerignore` to remove poetry.lock file which was causing issues with docker build
- Implemented `.github/workflows/backend-cd.yml`
- Updated `deployment/Dockerfile` to install poetry in the final image to resolve poetry.lock file not being found issue
- **Version 1.2 ZIP File Created**
- Pushed to Github

---

# Version 1.3
- Updated `src/models/predict.py` to calculate SHAP values
- Updated `tests/test_model.py` to add test for SHAP values
- Updated `tests/test_data.py` to add more tests for 80% target coverage
- Updated `src/api/routes/predict.py` to add new endpoint for getting model info
- **Version 1.3 ZIP File Created**
- Puhsed to Github

---

# Version 1.4
- Implemented `frontend/tailwind.config.js`
- Implemented `frontend/src/index.css`
- Implemented `frontend/.env.development`
- Implemented `frontend/.env.production`
- Implemented `frontend/src/services/api.js`
- Implemented `frontend/src/utils/validation.js`
- Implemented `frontend/src/utils/examples.js`
- Implemented `frontend/src/utils/storage.js`
- Implemented `frontend/src/components/Header.jsx`
- Implemented `frontend/src/components/LoadingSpinner.jsx`
- Implemented `frontend/src/components/FeatureInput.jsx`
- Implemented `frontend/src/components/QuickFillDropdown.jsx`
- Implemented `frontend/src/components/PredictionForm.jsx`
- Implemented `frontend/src/components/ResultDisplay.jsx`
- Implemented `frontend/src/components/FlowerVisualization.jsx`
- Implemented `frontend/src/components/FeatureContributionChart.jsx`
- Implemented `frontend/src/components/PredictionHistory.jsx`
- Implemented `frontend/src/components/ModelPerformanceDashboard.jsx`
- Implemented `frontend/src/components/ErrorMessage.jsx`
- Implemented `frontend/src/App.jsx`
- Implemented `frontend/src/main.jsx`
- Implemented `frontend/index.html`
- Implemented `frontend/vite.config.js`
- Implemented `frontend/package.json`
- Implemented `frontend/eslint.config.js`
- Implemented `rontend/postcss.config.js`
- Updated `src/api/routes/predict.py` to add feature importance support
- Updated `src/api/schemas/iris.py` to add feature importance as optional 
- Updated `frontend/src/App.jsx` to display historical predictions when prediction from history list is clicked
- Updated `frontend/src/utils/storage.js` to save feature contributions
- Updated `frontend/src/components/PredictionHistory.jsx`
- Implemented `frontend/src/App.jsx` to fix handling reload from history
- **Version 1.4 ZIP File Created**
- Pushed to Github
- **NOTE : THIS IS PRODUCTION READY, IS THE MOST STABLE AND HAS EVERY CORE FUNCTIONALITY**
---

# Version 1.5

- Implemented `src/monitoring/__init__.py`
- Implemented `src/monitoring/data_generator.py`
- Implemented `src/monitoring/drift_detector.py`
- Implemented `src/monitoring/data_logger.py`
- Implemented `src/api/routes/monitoring.py`
- Updated `src/api/main.py` to register new routes
- Updated `frontend/src/services/api.js` to add monitoring api services
- Implemented `frontend/src/components/DriftMonitor.jsx` to add drift status widget
- Updated `frontend/src/App.jsx` to include drift monitor
- **Version 1.5.1 ZIP File Created**
- Implemented `scripts/retrain_with_new_data.py`
- Implemented `.github/workflows/automated-retraining.yml`
- Updated `src/api/routes/monitoring.py` to add api endpoints for retraining
- Updated `src/core/config.py` to trigger github actions
- Implemented `src/monitoring/github_integration.py`
- Updated `src/api/routes/monitoring.py` to use GitHub integration
- Implemented `.env`
- **Version 1.5.2 ZIP File Created**
- Implemented `tests/test_monitoring_data_generator.py`
- Implemented `tests/test_monitoring_drift_detector.py`
- Implemented `tests/test_monitoring_data_logger.py`
- Implemented `tests/test_monitoring_api.py`
- Updated `frontend/src/services/api.js` to add retraining functions
- Updated `frontend/src/components/DriftMonitor.jsx` to add retraining support
- Updated `tests/test_api.py` to properly handle retraining changes prediction
**Version 1.5 ZIP File Created**
- Pushed to Github

---

# Version 1.6
- Implemented `frontend/src/components/WorkflowStatusViewer.jsx`
- Updated `frontend/src/App.jsx` to add workflow status viewer
- Implemented `frontend/src/components/TabNavigation.jsx` to add tabs for navigation
- Implemented `frontend/src/components/MLOpsDashboard.jsx`
- Updated `frontend/src/App.jsx` to add tabs
- Updated `src/api/routes/monitoring.py` to add endpoints for model registry
- Updated `frontend/src/services/api.js` to add api service in frontend
- Implemented `frontend/src/components/ModelVersionTimeline.jsx`
- Updated `frontend/src/components/MLOpsDashboard.jsx` to update model version timeline
- Implemented `frontend/src/components/DriftVisualization.jsx`
- Updated `frontend/src/components/MLOpsDashboard.jsx` to add drift detection
- **Version 1.6 ZIP File Created**
- Pushed to Github
