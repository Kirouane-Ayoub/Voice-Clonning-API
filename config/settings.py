from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    """Application settings."""

    # Server settings
    max_workers: int = 4

    # Model settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Audio processing settings
    max_file_size_mb: int = 100
    supported_formats: list = [".wav", ".mp3"]

    # Logging settings
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
