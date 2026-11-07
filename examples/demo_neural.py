"""Демо полного цикла: стираем пустые строки из файла автора,
а NeuralDecider (обученный на его же истории) восстанавливает ритм.

Запуск: python3 examples/demo_neural.py <file.ts> --ckpt models/blanklines/model.pt
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeprint import clike
from codeprint.harvest import extract_blankline_examples
from codeprint.neural import NeuralDecider
from codeprint.profiles import Decider
from codeprint.restyler import restyle


class ZeroDecider(Decider):
    """Стерилизатор: ни одной пустой строки."""

    def blanks_between(self, ctx) -> int:
        return 0


def boundary_labels(source: str) -> dict[tuple[str, str], int]:
    return {
        (e["prev_tail"][-1].strip(), e["next_line"].strip()): min(e["label"], 1)
        for e in extract_blankline_examples(source, clike)
        if e["prev_tail"]
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--ckpt", default="models/blanklines/model.pt")
    args = ap.parse_args()

    original = Path(args.file).read_text(encoding="utf-8")
    stripped = restyle(original, ZeroDecider(), backend=clike)
    decider = NeuralDecider(checkpoint=args.ckpt)
    restored = restyle(stripped, decider, backend=clike)

    orig = boundary_labels(original)
    rest = boundary_labels(restored)
    common = set(orig) & set(rest)
    agree = sum(1 for k in common if orig[k] == rest[k]) / max(len(common), 1)

    print(f"\nграниц в оригинале: {len(orig)}, сопоставлено после рестайла: {len(common)}")
    print(f"совпадение решений модели с автором: {agree:.1%}")

    print("\n--- фрагмент диффа (стерильно -> восстановлено), первые 40 строк ---")
    for i, line in enumerate(difflib.unified_diff(
            stripped.splitlines(), restored.splitlines(),
            fromfile="стерильно", tofile="нейронка", lineterm="", n=1)):
        if i >= 40:
            print("  ...")
            break
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
