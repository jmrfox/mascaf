#!/usr/bin/env python3
"""Replace a literal string across repo text files. Skips common non-repo dirs.

Usage (from repo root):
  python scripts/rename_string_in_repo.py OLD NEW [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".uv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".ipynb_checkpoints",
        ".rubbish",
    }
)


def _skip_path(rel: Path) -> bool:
    parts = rel.parts
    if any(p.endswith(".egg-info") for p in parts):
        return True
    return any(part in SKIP_DIRS for part in parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    old, new = args.old, args.new
    if not old:
        print("empty old string", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    changed: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if _skip_path(rel):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if old.encode() not in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            print(f"skip (not utf-8): {path.relative_to(root)}", file=sys.stderr)
            continue
        if old not in text:
            continue
        updated = text.replace(old, new)
        print(f"{'would write' if args.dry_run else 'write'}: {rel}")
        changed.append(path)
        if not args.dry_run:
            path.write_text(updated, encoding="utf-8", newline="")

    print(f"total: {len(changed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
