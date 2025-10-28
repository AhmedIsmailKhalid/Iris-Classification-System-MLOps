# **4. Technology Stack**

## **Core ML Stack:**
```toml
# pyproject.toml (Poetry)
[tool.poetry]
name = "iris-ml-pipeline"
version = "0.1.0"
description = "Production-grade ML classification with automated deployment"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
python = "^3.11.2"

[tool.poetry.dependencies]
python = "^3.11.2"
scikit-learn = "^1.5.2"
pandas = "^2.2.3"
numpy = "^1.26.4"
fastapi = "^0.115.4"
uvicorn = {extras = ["standard"], version = "^0.32.0"}
pydantic = "^2.9.2"
pydantic-settings = "^2.6.0"
python-multipart = "^0.0.12"
joblib = "^1.4.2"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.3"
pytest-cov = "^5.0.0"
pytest-asyncio = "^0.24.0"
httpx = "^0.27.2"
flake8 = "^7.1.1"
black = "^24.10.0"
mypy = "^1.13.0"
isort = "^5.13.2"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --cov=src --cov-report=html --cov-report=term-missing"
```

## Frontend Stack:
**Choice: React with Vite** (over Vue)

**Reasoning:**
- More industry demand (better for resume)
- Larger ecosystem
- Better job market alignment with FAANG

```json
// package.json (Frontend)
{
  "name": "iris-ml-pipeline-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext js,jsx --report-unused-disable-directives --max-warnings 0"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "axios": "^1.7.7",
    "recharts": "^2.12.7"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.2",
    "vite": "^5.4.8",
    "eslint": "^8.57.1",
    "eslint-plugin-react": "^7.37.1",
    "eslint-plugin-react-hooks": "^4.6.2",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.13"
  }
}
```

## **CI/CD Tools:**
- **GitHub Actions:** Free for public repos, 2000 min/month for private
- **Docker:** Containerization
- **GitHub Container Registry:** Free, integrated with Actions

## **Why These Choices?**

| Tool | Reason |
|------|--------|
| **scikit-learn** | Industry standard, simple, well-documented |
| **FastAPI** | Modern, fast, auto-generates API docs |
| **pytest** | Most popular Python testing framework |
| **black** | Opinionated formatter (no debates) |
| **Render** | Easiest free deployment with custom domains |