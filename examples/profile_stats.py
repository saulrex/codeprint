"""Профиль автора по оси imports: статистика схем из датасета harvest.

Negative result этапа: нейронка не предсказывает схему импортов по контенту
лучше мажоритарного бейзлайна — привычка сортировки/группировки почти не
коррелирует с содержимым блока. Поэтому эта ось работает ПРОФИЛЕМ:
доминантная схема автора применяется детерминированно через apply_scheme.

Запуск: python examples/profile_stats.py datasets/front/2026-07-03/imports.jsonl
Пишет рядом profile-imports.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeprint.imports_axis import SCHEMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="imports.jsonl от harvest")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        labels = [json.loads(l)["label"] for l in f]
    counts = Counter(labels)
    total = sum(counts.values())
    dist = {SCHEMES[i]: round(counts.get(i, 0) / total, 4) for i in range(len(SCHEMES))}
    dominant = max(dist, key=dist.get)

    profile = {
        "axis": "imports",
        "examples": total,
        "scheme_distribution": dist,
        "dominant_scheme": dominant,
    }
    out = Path(args.dataset).with_name("profile-imports.json")
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"сохранено: {out}")
    print(f"\nприменение: apply_scheme(source, {dominant!r}, lang) "
          f"из codeprint.imports_axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
