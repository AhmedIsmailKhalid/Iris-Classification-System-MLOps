"""
Application configuration module.
Loads settings from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    api_title: str = "Iris ML Pipeline API"
    api_version: str = "1.0.0"
    api_description: str = (
        "Production-grade ML API for Iris flower classification with "
        "automated drift detection and model retraining"
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # CORS Configuration
    allowed_origins: Union[List[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse comma-separated string or list."""
        if isinstance(v, str):
            # Split by comma and strip whitespace
            return [origin.strip() for origin in v.split(",")]
        return v

    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Model Configuration
    model_path: Path = Path("models/iris_classifier.joblib")
    model_version: str = "v1.0.0"

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Environment
    environment: str = "development"

    # GitHub Configuration (for automated retraining)
    github_token: str = ""
    github_repo: str = ""
    github_workflow_id: str = "automated-retraining.yml"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


# Create global settings instance
settings = Settings()