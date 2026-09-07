from src.config.settings import Settings, get_settings


def test_settings_use_default_values() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.openai_api_key is None


def test_settings_read_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-api-key"


def test_get_settings_returns_cached_instance() -> None:
    get_settings.cache_clear()

    first_settings = get_settings()
    second_settings = get_settings()

    assert first_settings is second_settings

    get_settings.cache_clear()