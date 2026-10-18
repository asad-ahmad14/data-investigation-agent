"""Application configuration, loaded from environment variables (and .env in development).

Settings are validated at startup so a bad configuration fails fast with a clear message.
Secrets are held as SecretStr so they do not appear in repr() or accidental logs.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # Database
    postgres_db: str = "retail_investigator"
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("")
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)

    app_db_user: str = "app_rw"
    app_db_password: SecretStr = SecretStr("")
    analyst_db_user: str = "analyst_ro"
    analyst_db_password: SecretStr = SecretStr("")

    # LLM
    llm_provider: Literal["mock", "gemini"] = "mock"
    llm_model: str = "gemini-3.6-flash"
    llm_api_key: SecretStr = SecretStr("")
    llm_timeout_seconds: int = Field(default=30, ge=1, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=5)  # transport retries per model call
    llm_max_repairs: int = Field(default=1, ge=0, le=3)  # re-asks after invalid structured output
    llm_max_output_tokens: int = Field(default=4096, ge=256, le=32768)

    # Investigation limits
    max_iterations: int = Field(default=3, ge=1, le=5)
    max_steps_per_plan: int = Field(default=6, ge=1, le=10)
    max_total_queries: int = Field(default=12, ge=1, le=30)
    max_llm_calls: int = Field(default=8, ge=1, le=20)
    max_result_rows: int = Field(default=500, ge=1, le=5000)
    query_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    investigation_timeout_seconds: int = Field(default=90, ge=5, le=600)
    max_question_length: int = Field(default=500, ge=20, le=2000)
    max_active_investigations: int = Field(default=4, ge=1, le=50)  # queued + running at once
    rate_limit_per_minute: int = Field(default=20, ge=0, le=10000)  # new investigations per client
    max_body_bytes: int = Field(default=16384, ge=1024, le=1_048_576)

    @model_validator(mode="after")
    def _check_llm_settings(self) -> "Settings":
        if self.llm_provider != "mock" and not self.llm_api_key.get_secret_value():
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider} requires LLM_API_KEY. "
                "Set it in your .env file, or use LLM_PROVIDER=mock."
            )
        return self

    def _db_url(self, user: str, password: SecretStr) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=user,
            password=password.get_secret_value() or None,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    @property
    def admin_db_url(self) -> URL:
        """Superuser URL. Used only for migrations, role creation and seeding."""
        return self._db_url(self.postgres_user, self.postgres_password)

    @property
    def app_db_url(self) -> URL:
        return self._db_url(self.app_db_user, self.app_db_password)

    @property
    def analyst_db_url(self) -> URL:
        return self._db_url(self.analyst_db_user, self.analyst_db_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()
