"""Small YAML config loader with a stdlib fallback.

The project config is intentionally plain dictionaries so scripts can run on
Great Lakes without installing a project package first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _minimal_yaml_value(text: str) -> Any:
    value = text.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value.replace("'", '"'))
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path = "configs/phase1_3.yaml") -> dict[str, Any]:
    """Load YAML using PyYAML when available, with a tiny fallback parser.

    The fallback supports the current config shape enough for smoke checks.
    Install/use PyYAML on Great Lakes for full YAML support.
    """
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")
        return data
    except ModuleNotFoundError:
        return _load_minimal_yaml(text)


def _load_minimal_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("- "):
            # The fallback is for smoke checks; PyYAML handles list blocks.
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _minimal_yaml_value(value)
    return root
