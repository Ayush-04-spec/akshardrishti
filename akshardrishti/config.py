"""Config loading with dotted access and CLI overrides.

Keeping experiments as YAML diffs (rather than edited code) is what makes the
paper's numbers reproducible -- every result can be traced to a config hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_CLASS_MAP = _REPO_ROOT / "configs" / "class_map.yaml"


class Config:
    """Thin dict wrapper supporting ``cfg.get("layout.sahi.enabled")``."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # -- access ---------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        val = self.get(dotted, sentinel)
        if val is sentinel:
            raise KeyError(f"missing required config key: {dotted!r}")
        return val

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self.require(key)

    def __contains__(self, dotted: str) -> bool:
        sentinel = object()
        return self.get(dotted, sentinel) is not sentinel

    # -- io -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
        path = Path(path) if path else DEFAULT_CONFIG
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls(data)
        for ov in overrides or []:
            if "=" not in ov:
                raise ValueError(f"override must be key=value, got {ov!r}")
            key, raw = ov.split("=", 1)
            cfg.set(key.strip(), _coerce(raw.strip()))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def hash(self) -> str:
        """Short stable hash -- stamp this on every result row."""
        blob = json.dumps(self._data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self._data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path


def _coerce(raw: str) -> Any:
    """Turn a CLI string into a sensible Python value."""
    low = raw.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


# ------------------------------------------------------------------ class map


class ClassMap:
    """IndicDLP's 42 source classes -> our 11 target classes."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.targets: list[str] = list(data["targets"])
        self.mapping: dict[str, str] = {k.lower(): v for k, v in (data.get("mapping") or {}).items()}
        self.ignore: set[str] = {s.lower() for s in (data.get("ignore") or [])}
        self.on_unmapped: str = data.get("on_unmapped", "fail")
        self._index = {name: i for i, name in enumerate(self.targets)}

        unknown = set(self.mapping.values()) - set(self.targets)
        if unknown:
            raise ValueError(f"class_map maps to targets not in `targets`: {sorted(unknown)}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> ClassMap:
        path = Path(path) if path else DEFAULT_CLASS_MAP
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def index_of(self, target: str) -> int:
        return self._index[target]

    def resolve(self, source_class: str) -> tuple[str, int] | None:
        """Map a source class name. ``None`` means 'drop this annotation'.

        Raises on unknown classes when ``on_unmapped: fail`` -- silently
        dropping unrecognised classes is how you end up with a model that
        mysteriously never predicts tables.
        """
        key = source_class.strip().lower().replace(" ", "_").replace("-", "_")
        if key in self.ignore:
            return None
        if key in self.mapping:
            tgt = self.mapping[key]
            return tgt, self._index[tgt]
        if self.on_unmapped == "fail":
            raise KeyError(
                f"unmapped source class {source_class!r}. Add it to configs/class_map.yaml "
                f"under `mapping` or `ignore`, or set `on_unmapped` to 'text'/'drop'."
            )
        if self.on_unmapped == "text":
            return "text", self._index["text"]
        return None

    def unmapped_report(self, source_classes: list[str]) -> list[str]:
        """Names that would fail -- call this before a long export run."""
        out = []
        for name in source_classes:
            key = name.strip().lower().replace(" ", "_").replace("-", "_")
            if key not in self.mapping and key not in self.ignore:
                out.append(name)
        return sorted(set(out))
