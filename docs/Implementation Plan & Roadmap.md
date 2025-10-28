### Phase 1: Backend Foundation

**1.1 Project Setup**
- [ ] Create GitHub repository: `iris-ml-pipeline`
- [ ] Initialize Poetry in `backend/` directory
- [ ] Set up Python 3.11.2 environment
- [ ] Install core dependencies
- [ ] Create directory structure
- [ ] Add MIT License
- [ ] Create `.gitignore` for Python

```bash
# Commands
mkdir iris-ml-pipeline && cd iris-ml-pipeline
mkdir backend frontend docs
cd backend
poetry init
poetry add scikit-learn pandas numpy fastapi uvicorn pydantic joblib
poetry add --group dev pytest pytest-cov httpx flake8 black mypy isort
poetry shell
```

**1.2 Data Pipeline**
- [ ] Create data loading module
- [ ] Implement preprocessing functions
- [ ] Add data validation
- [ ] Write unit tests for data functions
- [ ] Generate iris.csv from sklearn

**1.3 Model Training**
- [ ] Implement training script
- [ ] Train Logistic Regression model
- [ ] Calculate and log metrics
- [ ] Save model with joblib
- [ ] Create model registry JSON
- [ ] Write model tests

**1.4 Basic API**
- [ ] Set up FastAPI application
- [ ] Create Pydantic schemas
- [ ] Implement health check endpoint
- [ ] Implement prediction endpoint
- [ ] Add CORS middleware
- [ ] Test locally with uvicorn

---

### **Phase 2: Backend Testing & CI/CD**

**2.1 Testing Suite**
- [ ] Write comprehensive pytest tests
  - Data loading and preprocessing
  - Model prediction
  - API endpoints (integration)
- [ ] Set up pytest configuration
- [ ] Achieve >80% code coverage
- [ ] Add pre-commit hooks

**2.2 Backend CI/CD**
- [ ] Create GitHub Actions workflow for CI
  - Install Poetry
  - Run linting (flake8, black, isort)
  - Run type checking (mypy)
  - Run tests with coverage
- [ ] Create CD workflow
  - Train model on push to main
  - Build Docker image
  - Push to GitHub Container Registry
  - Deploy to Render

**Dockerfile (multi-stage):**
```dockerfile
# Build stage
FROM python:3.11.2-slim as builder

RUN pip install poetry==1.8.3

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Runtime stage
FROM python:3.11.2-slim

WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### **Phase 3: Frontend Development**

**3.1 Frontend Setup (30 mins)**
- [ ] Initialize React with Vite
- [ ] Install dependencies (axios, recharts, tailwind)
- [ ] Configure Tailwind CSS
- [ ] Set up ESLint
- [ ] Create component structure

```bash
# Commands
cd ../frontend
npm create vite@latest . -- --template react
npm install axios recharts
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**3.2 Component Development**
- [ ] Create Header component
- [ ] Create PredictionForm component with validation
- [ ] Create ResultDisplay component (with visualization)
- [ ] Create LoadingSpinner component
- [ ] Create ErrorMessage component
- [ ] Style with Tailwind CSS

**3.3 API Integration**
- [ ] Set up axios instance
- [ ] Create API service functions
- [ ] Handle CORS properly
- [ ] Add error handling
- [ ] Add loading states

**3.4 Testing & Polish**
- [ ] Test form validation
- [ ] Test API calls with mock data
- [ ] Add responsive design
- [ ] Optimize for mobile

---

### **Phase 4: Deployment**

**4.1 Backend Deployment**
- [ ] Create Render account
- [ ] Create new Web Service
- [ ] Connect GitHub repository
- [ ] Configure environment variables
- [ ] Set up health check endpoint
- [ ] Deploy and verify

**render.yaml:**
```yaml
services:
  - type: web
    name: iris-ml-pipeline-api
    env: docker
    repo: https://github.com/yourusername/iris-ml-pipeline
    dockerfilePath: ./backend/deployment/Dockerfile
    dockerContext: ./backend
    envVars:
      - key: API_HOST
        value: 0.0.0.0
      - key: API_PORT
        value: 8000
      - key: ENVIRONMENT
        value: production
    healthCheckPath: /health
```

**4.2 Frontend Deployment**
- [ ] Create Vercel account
- [ ] Import GitHub repository
- [ ] Configure build settings
- [ ] Set environment variables (API URL)
- [ ] Deploy and verify
- [ ] Test end-to-end flow

**vercel.json:**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

---

### **Phase 5: Documentation & Polish**

**5.1 Documentation**
- [ ] Write comprehensive README for main repo
- [ ] Write README for backend
- [ ] Write README for frontend
- [ ] Create architecture diagram
- [ ] Document API endpoints
- [ ] Add usage examples

**5.2 Portfolio Polish**
- [ ] Add badges (build status, coverage, license)
- [ ] Create demo GIF/screenshots
- [ ] Add live demo links
- [ ] Create resume bullet points
- [ ] Add contribution guidelines
- [ ] Final testing of all features