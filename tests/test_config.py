import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_use_mock_provider():
    s = Settings(_env_file=None)
    assert s.llm_provider == "mock"
    assert s.max_iterations == 3


def test_real_provider_requires_api_key():
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, llm_provider="gemini", llm_api_key="")
    assert "LLM_API_KEY" in str(exc.value)


def test_real_provider_with_key_is_accepted():
    s = Settings(_env_file=None, llm_provider="gemini", llm_api_key="test-key")
    assert s.llm_provider == "gemini"


def test_limits_are_bounded():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_iterations=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_result_rows=10**6)


def test_secrets_not_in_repr():
    s = Settings(_env_file=None, llm_provider="gemini", llm_api_key="super-secret-value")
    assert "super-secret-value" not in repr(s)


def test_db_urls_use_distinct_users():
    s = Settings(_env_file=None, postgres_password="admin-pw", app_db_password="app-pw")
    assert s.admin_db_url.username == "postgres"
    assert s.app_db_url.username == "app_rw"
    assert s.analyst_db_url.username == "analyst_ro"
