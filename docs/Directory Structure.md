# Directory Structure

```
Iris Classifier with Automated Testing
├── .coverage
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── create_structure.bat
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
├── Version History.md
├── .github
│   └── workflows
│       ├── automated-retraining.yml
│       ├── backend-cd.yml
│       ├── backend-ci.yml
│       └── docker-publish.yml
├── .pytest_cache
│   ├── .gitignore
│   ├── CACHEDIR.TAG
│   ├── README.md
│   └── v
│       └── cache
├── data
│   ├── monitoring
│   │   └── new_data.csv
│   ├── processed
│   │   └── .gitkeep
│   └── raw
│       └── iris.csv
├── deployment
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── render.yaml
├── docs
│   ├── Data Sources.md
│   ├── Deployment & Infrastructure.md
│   ├── Directory Structure.md
│   ├── Implementation Plan & Roadmap.md
│   ├── Success Criteria.md
│   ├── System Design Decisions.md
│   └── Technology Stack.md
├── frontend
│   ├── .env.development
│   ├── .env.example
│   ├── .env.production
│   ├── .gitignore
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── README.md
│   ├── tailwind.config.js
│   ├── vercel.json
│   ├── vite.config.js
│   ├── .github
│   │   └── workflows
│   ├── public
│   │   ├── favicon.ico
│   │   ├── robots.txt
│   │   └── vite.svg
│   └── src
│       ├── App.css
│       ├── App.jsx
│       ├── config.js
│       ├── index.css
│       ├── main.jsx
│       ├── assets
│       ├── components
│       ├── services
│       ├── styles
│       └── utils
├── models
│   ├── .gitkeep
│   ├── iris_classifier.joblib
│   └── model_registry.json
├── scripts
│   ├── evaluate_model.py
│   ├── generate_iris_data.py
│   ├── retrain_with_new_data.py
│   └── train_model.py
├── src
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes
│   │   └── schemas
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── predict.py
│   │   └── train.py
│   └── monitoring
│       ├── __init__.py
│       ├── data_generator.py
│       ├── data_logger.py
│       ├── drift_detector.py
│       └── github_integration.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_data.py
    ├── test_model.py
    ├── test_monitoring_api.py
    ├── test_monitoring_data_generator.py
    ├── test_monitoring_data_logger.py
    └── test_monitoring_drift_detector.py
```