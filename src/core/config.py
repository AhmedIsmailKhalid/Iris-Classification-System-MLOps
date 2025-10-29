"""
Configuration management using Pydantic Settings.
Loads configuration from environment variables.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_title: str = "Iris ML Pipeline API"
    api_version: str = "1.0.0"
    api_description: str = "Production-grade ML classification API for Iris dataset"

    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative frontend port
    ]

    # Model Configuration
    model_path: Path = Path("models/iris_classifier.joblib")
    model_version: str = "v1.0.0"

    # Logging Configuration
    log_level: str = "INFO"

    # Environment
    environment: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def cors_allow_credentials(self) -> bool:
        """Allow credentials in CORS."""
        return True

    @property
    def cors_allow_methods(self) -> List[str]:
        """Allowed HTTP methods."""
        return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    @property
    def cors_allow_headers(self) -> List[str]:
        """Allowed HTTP headers."""
        return ["*"]


# Global settings instance
settings = Settings()
