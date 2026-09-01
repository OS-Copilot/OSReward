from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JUDGE_DIRS = (
    _REPO_ROOT / "evaluation_examples" / "scale_cua_rl" / "judge_functions",
    _REPO_ROOT / "evaluation_examples" / "scale_cua_generated" / "judge_functions",
)


def _install_osworld_aliases() -> None:
    osworld_evaluators = importlib.import_module("desktop_env.evaluators")
    sys.modules.setdefault("desktop_env.evaluators", osworld_evaluators)

    for package_name in ("metrics", "getters"):
        osworld_package = importlib.import_module(
            f"desktop_env.evaluators.{package_name}"
        )
        sys.modules.setdefault(f"desktop_env.evaluators.{package_name}", osworld_package)


def _load_package(module_name: str, package_dir: Path) -> Optional[ModuleType]:
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        logger.warning("Failed to import ScaleCUA judge package %s: %s", package_dir, exc)
        return None
    return module


def _collect_callables(package_kind: str) -> Dict[str, Callable]:
    _install_osworld_aliases()
    collected: Dict[str, Callable] = {}

    for index, judge_dir in enumerate(_JUDGE_DIRS):
        package_dir = judge_dir / package_kind
        module_name = f"_scalecua_osworld_{index}_{package_kind}"
        module = _load_package(module_name, package_dir)
        if module is None:
            continue

        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name, None)
            if callable(value):
                collected.setdefault(name, value)

    return collected


@lru_cache(maxsize=1)
def _metrics() -> Dict[str, Callable]:
    return _collect_callables("verigen_metrics")


@lru_cache(maxsize=1)
def _getters() -> Dict[str, Callable]:
    return _collect_callables("verigen_getters")


def resolve_metric(name: str) -> Optional[Callable]:
    return _metrics().get(name)


def resolve_getter(getter_type: str) -> Optional[Callable]:
    return _getters().get(f"get_{getter_type}")
