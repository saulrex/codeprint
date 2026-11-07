#!/usr/bin/env python3
"""similarity — CLI сходства исходников.

  python similarity.py a.ts b.ts          # попарное сравнение
  python similarity.py --restyle FILE --profile you   # сходство до/после
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codeprint.similarity import report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--restyle", metavar="PATH",
                    help="сходство до/после рестайла профилем")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    if args.restyle:
        if not args.profile:
            ap.error("--restyle требует --profile")
        from codeprint.composer import Composer, load_profile
        composer = Composer(load_profile(args.profile))
        p = Path(args.restyle)
        files = [p] if p.is_file() else sorted(
            f for f in p.rglob("*") if f.suffix.lower() in (".py", ".ts", ".tsx",
                                                            ".js", ".jsx"))[:200]
        rows = []
        for f in files:
            try:
                src = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new = composer.apply(src, f.suffix)
            if new == src:
                continue
            rows.append((f.name, report(src, new)))
        for name, r in rows[:15]:
            print(f"  {name}: содержимое_было->стало={r['containment_before_in_after']:.3f} "
                  f"jaccard={r['jaccard']:.3f}")
        if rows:
            print(f"\nсредний containment(было->стало): "
                  f"{statistics.mean(r['containment_before_in_after'] for _, r in rows):.3f} "
                  f"({len(rows)} изменённых файлов)")
        return 0

    if len(args.files) != 2:
        ap.error("нужны два файла или --restyle")
    a = Path(args.files[0]).read_text(encoding="utf-8")
    b = Path(args.files[1]).read_text(encoding="utf-8")
    r = report(a, b)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
