"""Configuration validation and safety checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import ConfigurationError, Settings, clear_settings_cache, get_settings


def base(**overrides):
    values = {
        "_env_file": None,
        "pinecone_api_key": "pc-key",
        "pinecone_index_name": "epl",
        "gemini_api_key": "g-key",
    }
    values.update(overrides)
    return values


def test_missing_required_secret_names_the_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, pinecone_index_name="epl", gemini_api_key="g-key")
    assert "pinecone_api_key" in str(excinfo.value)
    assert "pc-key" not in str(excinfo.value)


def test_blank_secrets_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(pinecone_api_key="   "))


def test_blank_index_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(pinecone_index_name="  "))


def test_host_is_normalised_to_https() -> None:
    settings = Settings(**base(pinecone_host="my-index.svc.pinecone.io/"))
    assert settings.pinecone_host == "https://my-index.svc.pinecone.io"


def test_host_can_be_none() -> None:
    assert Settings(**base(pinecone_host="")).pinecone_host is None


def test_plain_http_host_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(pinecone_host="http://my-index.svc.pinecone.io"))


def test_default_namespace_is_normalised() -> None:
    assert Settings(**base(pinecone_namespace="__default__")).namespace == ""
    assert Settings(**base(pinecone_namespace="  ")).namespace == ""
    assert Settings(**base(pinecone_namespace=" epl ")).namespace == "epl"


def test_text_field_fallbacks_are_trimmed_and_deduplicated() -> None:
    settings = Settings(
        **base(pinecone_text_field_fallbacks=" chunk_text, content ,chunk_text, ")
    )
    assert settings.text_field_fallbacks == ["chunk_text", "content"]


def test_top_k_cannot_exceed_max_top_k() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(top_k=10, max_top_k=5))


def test_temperature_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(gemini_temperature=3.0))


def test_timeout_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(request_timeout_seconds=0.1))


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = Settings(**base(cors_origins=" http://a.test , http://b.test ,"))
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_default_cors_origins_when_blank() -> None:
    settings = Settings(**base(cors_origins="  "))
    assert settings.cors_origin_list == [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def test_optional_checks_can_be_disabled_explicitly() -> None:
    settings = Settings(
        **base(
            embedding_dimension="none",
            expected_metric="off",
            min_relevance_score="none",
        )
    )
    assert settings.embedding_dimension is None
    assert settings.expected_metric is None
    assert settings.min_relevance_score is None


def test_metric_is_validated_and_normalised() -> None:
    assert Settings(**base(expected_metric=" COSINE ")).expected_metric == "cosine"
    with pytest.raises(ValidationError):
        Settings(**base(expected_metric="manhattan"))


def test_text_field_and_database_path_must_not_be_blank() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(pinecone_text_field="  "))
    with pytest.raises(ValidationError):
        Settings(**base(conversation_db_path="  "))


def test_database_path_is_absolute() -> None:
    settings = Settings(**base(conversation_db_path="data/test.sqlite3"))
    assert settings.database_path.is_absolute()
    assert settings.database_path.name == "test.sqlite3"


def test_api_key_property_handles_blank() -> None:
    assert Settings(**base()).api_key is None
    assert Settings(**base(app_api_key="  ")).api_key is None
    assert Settings(**base(app_api_key=" secret ")).api_key == "secret"


def test_startup_warnings_cover_risky_production_defaults() -> None:
    settings = Settings(
        **base(
            app_env="production",
            min_relevance_score=None,
            enable_retrieve_endpoint=True,
        )
    )
    warnings = settings.startup_warnings()
    joined = " ".join(warnings)
    assert "MIN_RELEVANCE_SCORE" in joined
    assert "POST /retrieve" in joined
    assert "APP_API_KEY" in joined
    assert "SQLite" in joined


def test_get_settings_wraps_validation_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    for name in (
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    # A developer's real .env must not make this test pass or fail differently.
    monkeypatch.setitem(
        Settings.model_config, "env_file", tmp_path / "missing.env"
    )
    clear_settings_cache()
    try:
        with pytest.raises(ConfigurationError) as excinfo:
            get_settings()
        message = str(excinfo.value)
        assert "PINECONE_API_KEY" in message
        assert "GEMINI_API_KEY" in message
        assert ".env.example" in message
    finally:
        clear_settings_cache()


def test_invalid_max_question_characters_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(max_question_characters=0))


def test_invalid_conversation_limits_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(**base(conversation_max_turns=0))
    with pytest.raises(ValidationError):
        Settings(**base(conversation_max_characters=10))


def test_wildcard_cors_origin_produces_warning() -> None:
    settings = Settings(**base(cors_origins="*"))
    assert any("wildcard" in warning.lower() for warning in settings.startup_warnings())
