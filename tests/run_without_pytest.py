"""Minimal runner so the suite works before pytest is installed."""

from __future__ import annotations

import inspect
import tempfile
import traceback
from pathlib import Path

import test_rag_components as t


def main() -> int:
    failed = 0
    for name in sorted(dir(t)):
        if not name.startswith("test_"):
            continue
        fn = getattr(t, name)
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print("PASS", name)
        except Exception:
            failed += 1
            print("FAIL", name)
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
