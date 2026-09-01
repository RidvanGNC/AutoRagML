"""dynamics — veriye-özel strateji katmanı (ADR 0007 + 0010 + 0015).

- `planner.build_plan(DataProfile, TaskSpec, RunConfig) -> AdaptivePlan`
  (deterministik; `committed_ops` her zaman, `candidate_ops` HPO uzayında; kod üretmez, fit yok)
- `recipes/` — custom transform registry (`@register_recipe`, `recipe_paths`, entry-points)
- `synthesis.py` — v2: LLM recipe üretimi (v1'de boş)
"""

from __future__ import annotations

from autoragml.dynamics.planner import build_plan
from autoragml.dynamics.recipes import (
    RecipeError,
    get_recipe,
    list_recipes,
    load_recipe_paths,
    register_recipe,
    validate_recipes,
)

__all__ = [
    "RecipeError",
    "build_plan",
    "get_recipe",
    "list_recipes",
    "load_recipe_paths",
    "register_recipe",
    "validate_recipes",
]
