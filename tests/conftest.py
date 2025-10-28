"""
Pytest configuration and shared fixtures.
"""

import logging
import sys
from pathlib import Path

import pytest

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root_path):
    """Provide data directory path."""
    return project_root_path / "data"


@pytest.fixture(scope="session")
def models_dir(project_root_path):
    """Provide models directory path."""
    return project_root_path / "models"
