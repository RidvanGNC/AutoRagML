"""Settings — sırları `.env` + ortam değişkenlerinden çözer (ADR 0016 · ADR 0008/4).

`RunConfig` yalnız `*_env` **adları** taşır; gerçek değerler runtime'da burada çözülür.
`Settings` **asla serialize edilmez**. `SecretStr` maskeler.

pydantic-settings, her LLM sağlayıcının kendi tipli `BaseSettings` alt sınıfı için
saklı tutulur (v2); v1'de jenerik ad-çözümü yeterli.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr

_DEFAULT_ENV_FILE = Path(".env")


def parse_dotenv(text: str) -> dict[str, str]:
    """Minimal ama sağlam `.env` ayrıştırıcı.

    Destekler: `KEY=value`, `export KEY=value`, tek/çift tırnaklı değerler,
    `#` yorum satırları, boş satırlar, tırnaksız değerlerde ` #` sonrası inline yorum.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = rest.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            val = val[1:-1]
        else:
            hash_idx = val.find(" #")
            if hash_idx != -1:
                val = val[:hash_idx].rstrip()
        values[key] = val
    return values


class Settings:
    """Runtime sır/ortam çözücü. Değerleri saklar ama dışa vermez."""

    __slots__ = ("_values",)

    def __init__(
        self,
        *,
        env_file: str | Path | None = _DEFAULT_ENV_FILE,
        environ: dict[str, str] | None = None,
    ) -> None:
        values: dict[str, str] = {}
        if env_file is not None:
            path = Path(env_file)
            if path.is_file():
                values.update(parse_dotenv(path.read_text(encoding="utf-8")))
        # Gerçek ortam değişkenleri .env'i ezer.
        values.update(os.environ if environ is None else environ)
        self._values = values

    def get(self, env_name: str) -> str | None:
        """Ham değer (sır olmayan: endpoint, uri...)."""
        return self._values.get(env_name)

    def resolve_secret(self, env_name: str | None) -> SecretStr | None:
        """`*_env` adını `SecretStr`'e çöz. Ad yoksa / değer yoksa `None`."""
        if not env_name:
            return None
        raw = self._values.get(env_name)
        return SecretStr(raw) if raw is not None else None

    def require_secret(self, env_name: str) -> SecretStr:
        """Zorunlu sır — yoksa hata."""
        secret = self.resolve_secret(env_name)
        if secret is None:
            msg = f"Gerekli ortam değişkeni tanımlı değil: {env_name}"
            raise KeyError(msg)
        return secret

    def __repr__(self) -> str:
        return f"Settings(keys={len(self._values)})"  # değerler yok
