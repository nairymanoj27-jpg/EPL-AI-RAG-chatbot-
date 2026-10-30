"""Typed application configuration.

All secrets come from environment variables or a local ``.env`` file.  Nothing
in this module logs or returns a secret value.  Importing the app without a
usable configuration raises :class:`ConfigurationError` with the names of the
missing variables, never their values.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    """Application settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    # Application -----------------------------------------------------------
    app_name: str = "Premier League Stats RAG Chatbot"
    app_version: str = "1.0.0"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # Pinecone --------------------------------------------------------------
    pinecone_api_key: SecretStr
    pinecone_index_name: str
    pinecone_host: str | None = None
    pinecone_namespace: str = ""
    pinecone_embed_model: str = "multilingual-e5-large"
    pinecone_text_field: str = "text"
    pinecone_text_field_fallbacks: str = "chunk_text,content,page_content"
    embedding_dimension: int | None = 1024
    expected_metric: str | None = "cosine"

    # Gemini ----------------------------------------------------------------
    gemini_api_key: SecretStr
    gemini_model: str = "gemini-flash-latest"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 1024

    # Retrieval / prompt ----------------------------------------------------
    top_k: int = 5
    min_relevance_score: float | None = None
    max_context_chunks: int = 5
    max_context_characters: int = 12_000
    max_question_characters: int = 2_000
    max_top_k: int = 20
    request_timeout_seconds: float = 30.0
    max_request_bytes: int = 65_536

    # HTTP / security -------------------------------------------------------
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    app_api_key: SecretStr | None = None
    enable_retrieve_endpoint: bool = True
    serve_frontend: bool = True

    # Conversation memory ---------------------------------------------------
    conversation_storage: Literal["sqlite", "none"] = "sqlite"
    conversation_db_path: str = "data/conversations.sqlite3"
    conversation_max_turns: int = 6
    conversation_max_characters: int = 6_000

    # Readiness -------------------------------------------------------------
    readiness_cache_seconds: int = 60

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator(
        "pinecone_index_name",
        "pinecone_embed_model",
        "pinecone_text_field",
        "gemini_model",
        "app_name",
        "app_version",
        "conversation_db_path",
    )
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name.upper()} must not be blank")
        return value

    @field_validator("pinecone_api_key", "gemini_api_key")
    @classmethod
    def _required_secret(cls, value: SecretStr, info) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError(f"{info.field_name.upper()} must not be blank")
        return value

    @field_validator("pinecone_host")
    @classmethod
    def _normalise_host(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().rstrip("/")
        if not value:
            return None
        if "://" not in value:
            value = f"https://{value}"
        if not value.lower().startswith("https://"):
            raise ValueError("PINECONE_HOST must use https://")
        if len(value) <= len("https://"):
            raise ValueError("PINECONE_HOST is not a valid host")
        return value

    @field_validator("pinecone_namespace")
    @classmethod
    def _normalise_namespace(cls, value: str) -> str:
        value = value.strip()
        # The SDK represents the default namespace as an empty string.
        return "" if value in {"", "__default__"} else value

    @field_validator("pinecone_text_field_fallbacks")
    @classmethod
    def _normalise_fallbacks(cls, value: str) -> str:
        fields = [part.strip() for part in value.split(",") if part.strip()]
        return ",".join(dict.fromkeys(fields))

    @field_validator("min_relevance_score", mode="before")
    @classmethod
    def _optional_score(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "none", "off", "null"}:
            return None
        return value

    @field_validator("min_relevance_score")
    @classmethod
    def _finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("MIN_RELEVANCE_SCORE must be a finite number")
        return value

    @field_validator("gemini_temperature")
    @classmethod
    def _temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("GEMINI_TEMPERATURE must be between 0 and 2")
        return value

    @field_validator("log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        value = value.upper().strip()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if value not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def _timeout(cls, value: float) -> float:
        if not 1.0 <= value <= 180.0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be between 1 and 180")
        return value

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def _optional_dimension(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "none", "off", "null"}:
            return None
        return value

    @field_validator("embedding_dimension")
    @classmethod
    def _dimension(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 20_000:
            raise ValueError("EMBEDDING_DIMENSION must be between 1 and 20000")
        return value

    @field_validator("expected_metric", mode="before")
    @classmethod
    def _optional_metric(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "none", "off", "null"}:
            return None
        return value

    @field_validator("expected_metric")
    @classmethod
    def _known_metric(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        allowed = {"cosine", "dotproduct", "euclidean"}
        if value not in allowed:
            raise ValueError(f"EXPECTED_METRIC must be one of {sorted(allowed)} or none")
        return value

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        if self.max_context_chunks < 1:
            raise ValueError("MAX_CONTEXT_CHUNKS must be at least 1")
        if self.max_context_characters < 500:
            raise ValueError("MAX_CONTEXT_CHARACTERS must be at least 500")
        if self.max_top_k < 1 or self.max_top_k > 100:
            raise ValueError("MAX_TOP_K must be between 1 and 100")
        if not 1 <= self.top_k <= self.max_top_k:
            raise ValueError("TOP_K must be between 1 and MAX_TOP_K")
        if not 1 <= self.conversation_max_turns <= 50:
            raise ValueError("CONVERSATION_MAX_TURNS must be between 1 and 50")
        if self.conversation_max_characters < 500:
            raise ValueError("CONVERSATION_MAX_CHARACTERS must be at least 500")
        if self.max_request_bytes < 1_024:
            raise ValueError("MAX_REQUEST_BYTES must be at least 1024")
        if not 1 <= self.max_question_characters <= 2_000:
            raise ValueError("MAX_QUESTION_CHARACTERS must be between 1 and 2000")
        if not 1 <= self.gemini_max_output_tokens <= 65_536:
            raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be between 1 and 65536")
        if self.readiness_cache_seconds < 0:
            raise ValueError("READINESS_CACHE_SECONDS must be >= 0")
        return self

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def namespace(self) -> str:
        """Namespace passed to Pinecone (``""`` means the default namespace)."""
        return self.pinecone_namespace

    @property
    def text_field_fallbacks(self) -> list[str]:
        return [f for f in self.pinecone_text_field_fallbacks.split(",") if f]

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        # Never allow the wildcard to be silently used together with credentials.
        return origins or ["http://localhost:8000", "http://127.0.0.1:8000"]

    @property
    def database_path(self) -> Path:
        path = Path(self.conversation_db_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def api_key(self) -> str | None:
        if self.app_api_key is None:
            return None
        value = self.app_api_key.get_secret_value().strip()
        return value or None

    def startup_warnings(self) -> list[str]:
        """Non-fatal configuration concerns, safe to log at startup."""
        warnings: list[str] = []
        if self.min_relevance_score is None:
            warnings.append(
                "MIN_RELEVANCE_SCORE is unset: every top-k hit will be sent to Gemini. "
                "Calibrate a threshold for your index/embedding model."
            )
        if self.app_env == "production" and self.enable_retrieve_endpoint:
            warnings.append(
                "POST /retrieve is enabled in production and can expose indexed document text. "
                "Set ENABLE_RETRIEVE_ENDPOINT=false or protect it with APP_API_KEY."
            )
        if self.app_env == "production" and not self.api_key:
            warnings.append(
                "APP_API_KEY is unset: the API has no application-level authentication. "
                "Use a reverse proxy with authentication for public deployments."
            )
        if self.conversation_storage == "sqlite":
            warnings.append(
                "Conversation memory uses a local SQLite file. Conversation IDs act like bearer "
                "tokens; anyone who knows an ID can continue that conversation."
            )
        if "*" in self.cors_origin_list:
            warnings.append(
                "CORS_ORIGINS contains a wildcard. Any website can call the API from a browser; "
                "list explicit origins unless this is intentional."
            )
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings.

    Raises:
        ConfigurationError: if required settings are missing or invalid.  The
            message lists only variable names, never values.
    """
    try:
        return Settings()
    except ValidationError as exc:
        fields: list[str] = []
        messages: list[str] = []
        for error in exc.errors():
            location = error.get("loc", ())
            if location:
                name = str(location[0]).upper()
                if name not in fields:
                    fields.append(name)
            else:
                message = str(error.get("msg", "")).removeprefix("Value error, ").strip()
                if message and message not in messages:
                    messages.append(message)
        names = ", ".join(fields) if fields else "unknown fields"
        detail = f" {'; '.join(messages)}" if messages else ""
        raise ConfigurationError(
            "Invalid or missing configuration for: "
            f"{names}.{detail} Copy .env.example to .env and fill in the required values."
        ) from None


def clear_settings_cache() -> None:
    """Clear the cached settings (used by tests and local tooling)."""
    get_settings.cache_clear()
