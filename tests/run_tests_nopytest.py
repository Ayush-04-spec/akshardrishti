#!/usr/bin/env python3
"""Fallback test runner for environments without pytest.

`python -m pytest tests/ -v` is the normal way to run the suite. This script
exists so the same test file also runs on a bare Kaggle/Colab image where
pytest is not installed:

    python tests/run_tests_nopytest.py

It implements the small slice of the pytest API the suite actually uses:
``pytest.approx``, ``pytest.raises``, ``pytest.fixture``, and class-based test
collection with fixture injection.
"""

from __future__ import annotations

import inspect
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------- pytest shim


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, actual):
        if self.abs is not None:
            return abs(actual - self.expected) <= self.abs
        tol = max(self.rel * max(abs(self.expected), abs(actual)), 1e-12)
        return abs(actual - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected})"


class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__} but nothing was raised")
        return issubclass(exc_type, self.exc_type)


def _fixture(func=None, **_kwargs):
    def wrap(f):
        f.__is_fixture__ = True
        return f
    return wrap(func) if func else wrap


def _install_shim() -> None:
    if "pytest" in sys.modules:
        return
    mod = types.ModuleType("pytest")
    mod.approx = _Approx
    mod.raises = _Raises
    mod.fixture = _fixture
    mod.mark = types.SimpleNamespace(
        skip=lambda *a, **k: (lambda f: f),
        skipif=lambda *a, **k: (lambda f: f),
        parametrize=lambda *a, **k: (lambda f: f),
    )
    sys.modules["pytest"] = mod


# ------------------------------------------------------------------- runner


def run_module(module) -> tuple[int, int, list[str]]:
    fixtures = {
        name: fn for name, fn in vars(module).items()
        if callable(fn) and getattr(fn, "__is_fixture__", False)
    }

    passed = failed = 0
    failures: list[str] = []

    def call_test(fn, label: str) -> None:
        nonlocal passed, failed
        kwargs = {}
        for param in inspect.signature(fn).parameters:
            if param == "self":
                continue
            if param in fixtures:
                kwargs[param] = fixtures[param]()
            elif param == "tmp_path":
                import tempfile

                kwargs[param] = Path(tempfile.mkdtemp())
        try:
            fn(**kwargs)
            passed += 1
            print(f"  PASS  {label}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {label}")
            failures.append(f"{label}\n{traceback.format_exc()}")

    for name, obj in vars(module).items():
        if isinstance(obj, type) and name.startswith("Test"):
            print(f"\n{name}")
            instance = obj()
            for meth_name in dir(obj):
                if meth_name.startswith("test_"):
                    call_test(getattr(instance, meth_name), meth_name)
        elif callable(obj) and name.startswith("test_"):
            call_test(obj, name)

    return passed, failed, failures


def main() -> int:
    _install_shim()

    import importlib.util

    total_passed = total_failed = 0
    all_failures: list[str] = []

    test_files = sorted(Path(__file__).parent.glob("test_*.py"))
    if not test_files:
        print("no test files found")
        return 1

    for path in test_files:
        print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        p, f, failures = run_module(module)
        total_passed += p
        total_failed += f
        all_failures.extend(failures)

    print(f"\n{'=' * 60}")
    print(f"  {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    for failure in all_failures:
        print(f"\n{'-' * 60}\n{failure}")

    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
