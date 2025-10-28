# Directory Structure

```
iris-ml-pipeline/
│
├── .github/
│   └── workflows/
│       ├── backend-ci.yml           # Backend CI
│       ├── backend-cd.yml           # Backend CD
│       ├── docker-publish.yml       # Docker image publishing
│       ├── frontend-ci.yml          # Frontend CI
│       └── frontend-deploy.yml      # Vercel deployment
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py            # Health check endpoints
│   │   │   └── predict.py           # Prediction endpoints
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── iris.py              # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration management
│   │   └── logging.py               # Logging setup
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── model_loader.py          # Singleton model loader
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── data/
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│       └── .gitkeep
│
├── models/
│   ├── .gitkeep
│   └── model_registry.json
│
├── scripts/
│   ├── train_model.py               # Training script
│   └── evaluate_model.py            # Evaluation script
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── render.yaml
│
├── frontend/                         # React frontend
│   ├── .github/
│   │   └── workflows/
│   │       └── vercel-deploy.yml    # Optional: Vercel deployment override
│   │
│   ├── public/
│   │   ├── favicon.ico
│   │   └── robots.txt
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── PredictionForm.jsx
│   │   │   ├── ResultDisplay.jsx
│   │   │   ├── LoadingSpinner.jsx
│   │   │   └── ErrorMessage.jsx
│   │   ├── services/
│   │   │   └── api.js               # Axios API calls
│   │   ├── utils/
│   │   │   └── validation.js
│   │   ├── styles/
│   │   │   └── index.css
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── config.js                # API endpoints config
│   │
│   ├── .env.example
│   ├── .env.local                   # Local development
│   ├── .gitignore
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── eslint.config.js
│   ├── vercel.json                  # Vercel configuration
│   └── README.md
│
├── docs/
│   ├── Data Sources.md
│   ├── Deployment & Infrastructure.md
│   ├── Directory Strucutre.md
│   ├── Implementation Plan & Roadmap.md
│   ├── Success Criteria.md
│   ├── System Design Decisions.md
│   └── Technology Stack.md
│
├── .env.example
├── .gitignore
├── .dockerignore
├── pyproject.toml                   # Poetry config (Backend)
├── poetry.lock
├── LICENSE                          # MIT License
└── README.md                        # Main project README
```

## Benefits of This Structure

| Benefit | Explanation |
|---------|-------------|
| **Cleaner root** | Backend is the "main" project, frontend is a submodule |
| **Simpler imports** | `from src.api.main import app` instead of `from backend.src.api.main import app` |
| **Standard Python** | Follows conventional Python project structure |
| **Docker context** | Easier Dockerfile paths (no `backend/` prefix) |
| **CI/CD simplicity** | Workflows reference paths without extra nesting |