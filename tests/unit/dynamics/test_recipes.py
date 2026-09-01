"""dynamics.recipes — registry: decorator, path yükleme, çakışma, doğrulama (ADR 0015)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoragml.contracts.enums import Provenance
from autoragml.contracts.plan_context import PlanContext
from autoragml.dynamics import recipes as reg
from autoragml.dynamics.recipes import RecipeError
from autoragml.transform import StatelessFitted


class _Fitted:
    provenance_fitted_on = Provenance.TRAIN

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def get_params(self) -> dict[str, object]:
        return {}


class _DemoRecipe:
    name = "demo"

    def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> _Fitted:
        return _Fitted()


def test_register_and_get() -> None:
    reg._REGISTRY.pop("t_demo", None)
    reg.register_recipe("t_demo")(_DemoRecipe)
    assert "t_demo" in reg.list_recipes()
    assert reg.get_recipe("t_demo") is _DemoRecipe


def test_get_unknown_raises() -> None:
    with pytest.raises(RecipeError, match="bulunamadı"):
        reg.get_recipe("yok_boyle")


def test_collision_raises() -> None:
    reg._REGISTRY.pop("t_dup", None)
    reg.register_recipe("t_dup")(_DemoRecipe)

    class _Other:
        def fit(self, frame: pd.DataFrame, ctx: PlanContext) -> _Fitted:
            return _Fitted()

    with pytest.raises(RecipeError, match="çakışması"):
        reg.register_recipe("t_dup")(_Other)


def test_non_transform_rejected() -> None:
    class _NoFit:
        pass

    with pytest.raises(RecipeError, match="protokol"):
        reg.register_recipe("t_nofit")(_NoFit)  # type: ignore[arg-type]


def test_load_from_path(tmp_path: Path) -> None:
    recipe_file = tmp_path / "my_recipe.py"
    recipe_file.write_text(
        "from autoragml.dynamics.recipes import register_recipe\n"
        "@register_recipe('t_from_path')\n"
        "class R:\n"
        "    def fit(self, frame, ctx):\n"
        "        return None\n",
        encoding="utf-8",
    )
    reg._REGISTRY.pop("t_from_path", None)
    reg._paths_loaded.discard(str(tmp_path.resolve()))
    reg.load_recipe_paths([tmp_path])
    assert "t_from_path" in reg.list_recipes()


def test_validate_recipes_fail_fast() -> None:
    with pytest.raises(RecipeError):
        reg.validate_recipes(["definitely_missing"])


def test_stateless_fitted_roundtrip() -> None:
    sf = StatelessFitted(lambda df: df.assign(x2=df["x"] * 2), params={"k": 2})
    out = sf.apply(pd.DataFrame({"x": [1, 2]}))
    assert list(out["x2"]) == [2, 4]
    assert sf.get_params() == {"k": 2}
    assert sf.provenance_fitted_on is Provenance.TRAIN
