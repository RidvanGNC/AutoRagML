"""AutoRagML — modalite-agnostik, deterministik AutoML çekirdeği (v1: tablo + zaman serisi).

RAG/agent katmanı v2'de ayrı üst katman olarak eklenecektir.

>>> from autoragml import AutoRagML
>>> result = AutoRagML(preset="tabular_fast").fit(df, target="sales")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.0.1"

__all__ = ["AutoRagML", "__version__"]

if TYPE_CHECKING:
    from autoragml.interfaces import AutoRagML


def __getattr__(name: str) -> Any:
    """Lazy export — `import autoragml` hafif kalsın (PEP 562)."""
    if name == "AutoRagML":
        from autoragml.interfaces import AutoRagML

        return AutoRagML
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
