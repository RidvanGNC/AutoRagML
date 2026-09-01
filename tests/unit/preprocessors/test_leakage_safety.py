"""preprocessors — leakage-safe by construction (ADR 0011).

Kritik testler: target encoding cross-fitting, fit yalnız train'de, unseen kategori.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoragml.preprocessors.catalog import build_encode
from tests.unit.preprocessors._util import ctx


def _cat_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cat = rng.choice(["a", "b", "c", "d"], n)
    effect = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}
    y = np.array([effect[c] for c in cat]) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"cat": cat, "y": y})


def test_target_encode_fit_transform_differs_from_fit_apply() -> None:
    df = _cat_frame()
    te = build_encode(["cat"], "target_encode")
    _, cross_fitted = te.fit_transform(df, ctx())
    fitted = te.fit(df, ctx())
    full_train = fitted.apply(df)
    # cross-fitting -> train çıktısı full-train kodlamasından farklı olmalı
    assert not np.allclose(cross_fitted["cat"].to_numpy(), full_train["cat"].to_numpy())


def test_target_encode_apply_on_test_uses_full_train() -> None:
    train = _cat_frame(300)
    test = _cat_frame(120)
    te = build_encode(["cat"], "target_encode")
    fitted = te.fit(train, ctx())
    out = fitted.apply(test)
    # test kodlaması train kategori ortalamalarına yakın olmalı (a<b<c<d)
    means = out.groupby(test["cat"])["cat"].mean()
    assert means["a"] < means["b"] < means["c"] < means["d"]


def test_target_encode_does_not_touch_test_target() -> None:
    train = _cat_frame(300)
    test = _cat_frame(120)
    # test'in target'ını uçlara it — kodlamayı değiştirmemeli
    poisoned = test.copy()
    poisoned["y"] = 1e6
    te = build_encode(["cat"], "target_encode")
    fitted = te.fit(train, ctx())
    a = fitted.apply(test)["cat"].to_numpy()
    b = fitted.apply(poisoned)["cat"].to_numpy()
    assert np.allclose(a, b)  # apply test target'ına bakmaz


def test_onehot_unknown_category_safe() -> None:
    train = pd.DataFrame({"c": ["x", "y", "x", "y"], "y": [1.0, 2.0, 1.0, 2.0]})
    test = pd.DataFrame({"c": ["x", "z"], "y": [1.0, 2.0]})  # z eğitimde yok
    oh = build_encode(["c"], "onehot")
    fitted = oh.fit(train, ctx())
    out = fitted.apply(test)
    assert "c" not in out.columns
    assert len(out) == 2  # çökme yok


def test_ordinal_unknown_category_gets_sentinel() -> None:
    train = pd.DataFrame({"c": ["x", "y"], "y": [1.0, 2.0]})
    test = pd.DataFrame({"c": ["z"], "y": [1.0]})
    fitted = build_encode(["c"], "ordinal").fit(train, ctx())
    assert fitted.apply(test)["c"].iloc[0] == -1
