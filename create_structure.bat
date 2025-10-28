@echo off
echo Creating iris-ml-pipeline structure...

REM ==================== GITHUB WORKFLOWS ====================
mkdir .github
mkdir .github\workflows
type nul > .github\workflows\backend-ci.yml
type nul > .github\workflows\backend-cd.yml
type nul > .github\workflows\docker-publish.yml
type nul > .github\workflows\frontend-ci.yml
type nul > .github\workflows\frontend-deploy.yml


REM ==================== BACKEND SRC ====================
mkdir src
mkdir src\api\routes
mkdir src\api\schemas
mkdir src\core
mkdir src\data
mkdir src\models
mkdir src\utils

type nul > src\__init__.py
type nul > src\api\__init__.py
type nul > src\api\main.py
type nul > src\api\routes\__init__.py
type nul > src\api\routes\health.py
type nul > src\api\routes\predict.py
type nul > src\api\schemas\__init__.py
type nul > src\api\schemas\iris.py

type nul > src\core\__init__.py
type nul > src\core\config.py
type nul > src\core\logging.py

type nul > src\data\__init__.py
type nul > src\data\load_data.py
type nul > src\data\preprocess.py

type nul > src\models\__init__.py
type nul > src\models\train.py
type nul > src\models\predict.py
type nul > src\models\model_loader.py

type nul > src\utils\__init__.py
type nul > src\utils\metrics.py


REM ==================== TESTS ====================
mkdir tests
type nul > tests\__init__.py
type nul > tests\conftest.py
type nul > tests\test_data.py
type nul > tests\test_model.py
type nul > tests\test_api.py


REM ==================== DATA ====================
mkdir data
mkdir data\raw
mkdir data\processed
type nul > data\raw\iris.csv
type nul > data\processed\.gitkeep


REM ==================== MODEL ARTIFACTS ====================
mkdir models
type nul > models\.gitkeep
type nul > models\model_registry.json


REM ==================== SCRIPTS ====================
mkdir scripts
type nul > scripts\train_model.py
type nul > scripts\evaluate_model.py


REM ==================== DEPLOYMENT ====================
mkdir deployment
type nul > deployment\Dockerfile
type nul > deployment\docker-compose.yml
type nul > deployment\render.yaml


REM ==================== FRONTEND ====================
mkdir frontend
mkdir frontend\.github\workflows
mkdir frontend\public
mkdir frontend\src\components
mkdir frontend\src\services
mkdir frontend\src\utils
mkdir frontend\src\styles

type nul > frontend\.github\workflows\vercel-deploy.yml

type nul > frontend\public\favicon.ico
type nul > frontend\public\robots.txt

type nul > frontend\src\components\Header.jsx
type nul > frontend\src\components\PredictionForm.jsx
type nul > frontend\src\components\ResultDisplay.jsx
type nul > frontend\src\components\LoadingSpinner.jsx
type nul > frontend\src\components\ErrorMessage.jsx

type nul > frontend\src\services\api.js
type nul > frontend\src\utils\validation.js
type nul > frontend\src\styles\index.css
type nul > frontend\src\App.jsx
type nul > frontend\src\main.jsx
type nul > frontend\src\config.js

type nul > frontend\.env.example
type nul > frontend\.env.local
type nul > frontend\.gitignore
type nul > frontend\index.html
type nul > frontend\package.json
type nul > frontend\package-lock.json
type nul > frontend\vite.config.js
type nul > frontend\tailwind.config.js
type nul > frontend\postcss.config.js
type nul > frontend\eslint.config.js
type nul > frontend\vercel.json
type nul > frontend\README.md


REM ==================== DOCS ====================
mkdir docs
type nul > docs\architecture.md
type nul > docs\api_documentation.md
type nul > docs\deployment_guide.md


REM ==================== PROJECT ROOT FILES ====================
type nul > .env.example
type nul > .gitignore
type nul > .dockerignore
type nul > pyproject.toml
type nul > poetry.lock
type nul > LICENSE
type nul > README.md


echo ✅ Directory structure created successfully!
pause
