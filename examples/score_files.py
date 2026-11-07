"""Прогон файлов через stylometer: P(написал целевой автор) для каждого.

Запуск: python3 examples/score_files.py --model models/stylometer-you f1.ts f2.py ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeprint.harvest import BACKEND_BY_EXT
from codeprint.stylometer import Stylometer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    stylo = Stylometer(args.model)
    print(f"{'P(автор)':>9s}  файл")
    for f in args.files:
        p = Path(f)
        backend = BACKEND_BY_EXT.get(p.suffix.lower())
        if backend is None:
            print(f"{'—':>9s}  {f} (неизвестное расширение)")
            continue
        score = stylo.score_source(p.read_text(encoding="utf-8"), backend)
        print(f"{score:9.3f}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
