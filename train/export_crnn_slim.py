#!/usr/bin/env python3
"""Export a slim, weights-only CRNN checkpoint from the original training artifact.

Problem
-------
The checkpoint saved by train_crnn.py on Python 3.13 (Google Colab) contains
`pathlib.PosixPath` objects inside the `args` dict.  Python 3.13 stores that
class in `pathlib._local`, a private submodule that does not exist on Python
3.12 or earlier.  Both `torch.load(..., weights_only=False)` and
`torch.load(..., weights_only=True)` therefore fail on local Python 3.12.

Solution
--------
1. Install a shim for `pathlib._local` before unpickling so the
   ModuleNotFoundError is suppressed.
2. Load with `weights_only=False` (safe because we verify SHA-256 first).
3. Save a slim dict containing only the model state, charset_size, and
   provenance metadata — no optimizer, no args, no best_cer.  The slim file
   loads cleanly with `weights_only=True` on any Python version.

Usage
-----
    python train/export_crnn_slim.py
    python train/export_crnn_slim.py --src weights/recognize/crnn_mozhi_himr.pt \
                                     --dst weights/recognize/crnn_mozhi_himr_slim.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import types
import pathlib
from pathlib import Path

# ---------------------------------------------------------------------------
# pathlib._local shim
# ---------------------------------------------------------------------------
# Python 3.13 relocated PosixPath / WindowsPath into a private submodule
# `pathlib._local` that does not exist on Python ≤ 3.12.  When a checkpoint
# pickled on 3.13 is unpickled here, the pickler tries to import that module
# and fails with ModuleNotFoundError.
#
# We register a shim before any torch.load call so the import succeeds.
# Pure* variants are used here because PosixPath cannot be instantiated on
# Windows (it raises NotImplementedError); PurePosixPath / PureWindowsPath
# are cross-platform stand-ins that pickle/unpickle without running
# OS-specific code.  We only need the class objects to exist — their values
# are never used after loading (they live inside `args`, which we discard).
_shim = types.ModuleType("pathlib._local")
_shim.PosixPath    = pathlib.PurePosixPath   # type: ignore[attr-defined]
_shim.WindowsPath  = pathlib.PureWindowsPath  # type: ignore[attr-defined]
_shim.Path         = pathlib.PurePath         # type: ignore[attr-defined]
_shim.PurePath     = pathlib.PurePath
_shim.PurePosixPath   = pathlib.PurePosixPath
_shim.PureWindowsPath = pathlib.PureWindowsPath
sys.modules["pathlib._local"] = _shim

# ---------------------------------------------------------------------------
# torch import AFTER shim registration
# ---------------------------------------------------------------------------
import torch  # noqa: E402  (import after sys.modules manipulation is intentional)

EXPECTED_CRNN_SHA256 = "626E6B86968FA550DD72E3BDBB9E1119F0B7C0195E65A0219B687FDF1151939A"

SLIM_FORMAT = "akshardrishti-crnn-slim-v1"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path,
                   default=Path("weights/recognize/crnn_mozhi_himr.pt"))
    p.add_argument("--dst", type=Path,
                   default=Path("weights/recognize/crnn_mozhi_himr_slim.pt"))
    args = p.parse_args()

    # ------------------------------------------------------------------ verify
    if not args.src.exists():
        sys.exit(f"ERROR: source not found: {args.src}")

    print(f"Source : {args.src}")
    print(f"Dest   : {args.dst}")
    print()

    print("Computing SHA-256 of source …", end=" ", flush=True)
    actual_hash = sha256_of(args.src)
    print(actual_hash)

    if actual_hash != EXPECTED_CRNN_SHA256:
        sys.exit(
            f"ERROR: SHA-256 mismatch.\n"
            f"  expected : {EXPECTED_CRNN_SHA256}\n"
            f"  actual   : {actual_hash}\n"
            f"Aborting — will not load an unexpected file."
        )
    print("SHA-256 verified ✓")
    print()

    # -------------------------------------------------------------------- load
    # weights_only=False is justified here because:
    #   1. We verified the SHA-256 above against the known training artifact.
    #   2. The file is our own checkpoint produced by train_crnn.py.
    print("Loading original checkpoint (weights_only=False) …", end=" ", flush=True)
    state = torch.load(args.src, map_location="cpu", weights_only=False)
    print("OK")

    # Confirm expected keys are present
    required = {"model", "charset_size", "epoch"}
    missing = required - state.keys()
    if missing:
        sys.exit(f"ERROR: checkpoint missing keys: {missing}")

    print(f"  epoch        : {state['epoch']}")
    print(f"  charset_size : {state['charset_size']}")
    print(f"  model keys   : {len(state['model'])}")
    print()

    # -------------------------------------------------------------------- slim
    slim = {
        "model": {k: v.cpu() for k, v in state["model"].items()},
        "charset_size": state["charset_size"],
        "source_sha256": actual_hash,
        "source_epoch": state["epoch"],
        "format": SLIM_FORMAT,
    }

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving slim checkpoint to {args.dst} …", end=" ", flush=True)
    torch.save(slim, args.dst)
    print("OK")

    slim_size_mb = args.dst.stat().st_size / 1024 / 1024
    slim_hash    = sha256_of(args.dst)
    print(f"  size   : {slim_size_mb:.2f} MB")
    print(f"  SHA-256: {slim_hash}")
    print()

    # --------------------------------------------------------- self-verify
    print("Self-verification …")

    # 1. Reload slim with weights_only=True (the whole point)
    slim2 = torch.load(args.dst, map_location="cpu", weights_only=True)
    assert slim2["format"] == SLIM_FORMAT,       "format tag mismatch"
    assert slim2["charset_size"] == state["charset_size"], "charset_size mismatch"
    assert slim2["source_sha256"] == actual_hash,          "source_sha256 mismatch"
    assert slim2["source_epoch"] == state["epoch"],        "source_epoch mismatch"
    print("  weights_only=True reload: OK")

    # 2. Key count and shape check
    orig_keys = set(state["model"].keys())
    slim_keys = set(slim2["model"].keys())
    assert orig_keys == slim_keys, f"key mismatch: {orig_keys ^ slim_keys}"
    print(f"  key count matches: {len(orig_keys)} keys")

    # 3. Tensor equality
    mismatches = []
    for k in orig_keys:
        if not torch.equal(state["model"][k].cpu(), slim2["model"][k]):
            mismatches.append(k)
    if mismatches:
        sys.exit(f"ERROR: tensor mismatch for keys: {mismatches}")
    print(f"  torch.equal True for all {len(orig_keys)} tensors")

    # 4. Parameter count
    total_params = sum(v.numel() for v in slim2["model"].values())
    print(f"  total parameters: {total_params:,}")

    print()
    print("=" * 56)
    print("  export_crnn_slim: DONE")
    print(f"  slim file  : {args.dst}")
    print(f"  size       : {slim_size_mb:.2f} MB")
    print(f"  SHA-256    : {slim_hash}")
    print(f"  params     : {total_params:,}")
    print(f"  charset_sz : {slim2['charset_size']}")
    print("=" * 56)


if __name__ == "__main__":
    main()
