"""
Environment-driven configuration via pydantic-settings.

Env vars: ENGINE, ECHO_DELAY_MS, LOG_LEVEL, FINISH_WATCHDOG_S
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, populated from environment variables."""

    # Engine selection: "echo" | "fake_model" | "model"
    engine: str = "echo"

    # Echo engine: delay in milliseconds before replaying audio
    echo_delay_ms: int = 2000

    # Logging level
    log_level: str = "INFO"

    # Watchdog: seconds after input_end before forcing session_complete
    finish_watchdog_s: float = 25.0

    # Server host/port (used by uvicorn via main.py)
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "", "case_sensitive": False}


# Singleton – import this everywhere
settings = Settings()
