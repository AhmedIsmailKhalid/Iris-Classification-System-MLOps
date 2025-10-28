### **Architecture Diagram:**

```
┌──────────────────────────────────────────────────────────────┐
│                     User's Browser                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Vercel (Frontend Hosting)                       │
│  - React SPA                                                 │
│  - Auto-deploy from GitHub                                   │
│  - CDN for static assets                                     │
│  - Custom domain support                                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ HTTPS/CORS
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Render (Backend Hosting)                        │
│  - FastAPI REST API                                          │
│  - Docker container                                          │
│  - Auto-deploy from GitHub                                   │
│  - Health checks enabled                                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│           ML Model (Joblib serialized)                       │
│  - Loaded into memory on startup                             │
│  - Versioned in GitHub                                       │
└──────────────────────────────────────────────────────────────┘
```

### **CI/CD Pipeline:**

```
┌──────────────────────────────────────────────────────────────┐
│                    GitHub Repository                         │
└───────┬──────────────────────────────────────────┬───────────┘
        │                                          │
        ▼                                          ▼
┌───────────────────┐                    ┌───────────────────┐
│  Backend CI/CD    │                    │  Frontend CI/CD   │
│                   │                    │                   │
│ 1. Run pytest     │                    │ 1. Run ESLint     │
│ 2. Run flake8     │                    │ 2. Build React    │
│ 3. Run black      │                    │ 3. Run tests      │
│ 4. Train model    │                    │ 4. Deploy Vercel  │
│ 5. Build Docker   │                    │                   │
│ 6. Push to GHCR   │                    │                   │
│ 7. Deploy Render  │                    │                   │
└───────────────────┘                    └───────────────────┘
```

### **Environment Variables:**

**Backend (.env):**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false

# CORS Configuration
ALLOWED_ORIGINS=https://iris-ml-pipeline.vercel.app,http://localhost:5173

# Model Configuration
MODEL_PATH=models/iris_classifier.joblib
MODEL_VERSION=v1.0.0

# Logging
LOG_LEVEL=INFO

# Deployment
ENVIRONMENT=production
```

**Frontend (.env.local):**
```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=5000

# Production (.env.production)
VITE_API_BASE_URL=https://iris-ml-pipeline.onrender.com
VITE_API_TIMEOUT=5000
```