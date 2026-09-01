"""config.settings — .env ayrıştırma + sır çözümü (ADR 0016)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from autoragml.config.settings import Settings, parse_dotenv


def test_parse_dotenv_forms() -> None:
    text = "\n".join(
        [
            "# yorum",
            "",
            "PLAIN=abc",
            'QUOTED="a b c"',
            "SINGLE='x y'",
            "export EXPORTED=zzz",
            "WITH_COMMENT=val # inline",
            "EMPTY=",
        ]
    )
    parsed = parse_dotenv(text)
    assert parsed == {
        "PLAIN": "abc",
        "QUOTED": "a b c",
        "SINGLE": "x y",
        "EXPORTED": "zzz",
        "WITH_COMMENT": "val",
        "EMPTY": "",
    }


def test_settings_env_overrides_dotenv(tmp_path: object) -> None:
    envf = tmp_path / ".env"  # type: ignore[operator]
    envf.write_text("SECRET_KEY=from_file\nURI=file_uri\n", encoding="utf-8")
    s = Settings(env_file=envf, environ={"SECRET_KEY": "from_env"})
    assert s.get("SECRET_KEY") == "from_env"
    assert s.get("URI") == "file_uri"


def test_resolve_secret_returns_secretstr() -> None:
    s = Settings(env_file=None, environ={"AZURE_OPENAI_KEY": "sk-123"})
    secret = s.resolve_secret("AZURE_OPENAI_KEY")
    assert isinstance(secret, SecretStr)
    assert secret.get_secret_value() == "sk-123"
    assert s.resolve_secret("MISSING") is None
    assert s.resolve_secret(None) is None


def test_require_secret_raises_when_missing() -> None:
    s = Settings(env_file=None, environ={})
    with pytest.raises(KeyError):
        s.require_secret("NOPE")


def test_repr_hides_values() -> None:
    s = Settings(env_file=None, environ={"SECRET_KEY": "sk-xyz"})
    assert "sk-xyz" not in repr(s)
