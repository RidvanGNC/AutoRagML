"""explain — şampiyon öznitelik atıfı (ADR 0037).

Kullanıcı-tetikli, yan-etkisiz, salt-okunur. SHAP opsiyonel (`[explain]` extra); yoksa
model-agnostik permutation fallback (çekirdek `scikit-learn`).
"""

from __future__ import annotations

from autoragml.explain.attribution import explain_champion

__all__ = ["explain_champion"]
