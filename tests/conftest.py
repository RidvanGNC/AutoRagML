"""Paylaşılan pytest fixture'ları.

`_lean_catalog` (autouse): test oturumu boyunca **ağır-fit** modelleri havuzdan düşürür
(EBM cyclic-boosting, NGBoost saf-Python, SVR/SVC O(n²), AutoTBATS, TabICL ağ-indirmesi).
Bu modeller üretimde açık; test suite'i her e2e/champion koşumunda onları fit etmesin diye
kapatılır. Katalog **girişleri** yerinde kalır — yalnız `enabled` bayrağı override edilir.
`@pytest.mark.full_catalog` ile bir test/modül tam katalogla koşabilir.
"""

from __future__ import annotations

import pytest

_LEAN_DISABLE = ("ebm", "ngboost", "svr", "svc", "auto_tbats", "tabicl")


@pytest.fixture(autouse=True)
def _lean_catalog(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("full_catalog"):
        return
    from autoragml.models import registry

    real = registry._builtin_catalog  # noqa: SLF001

    def _lean() -> dict:
        cat = {k: dict(v) for k, v in real().items()}
        for key in _LEAN_DISABLE:
            if key in cat:
                cat[key]["enabled"] = False
        return cat

    monkeypatch.setattr(registry, "_builtin_catalog", _lean)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "full_catalog: testte ağır modeller de dahil tüm katalog aktif")
