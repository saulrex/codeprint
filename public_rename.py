#!/usr/bin/env python3
"""public_rename — repo-wide переименование публичных имён (dry-run по умолчанию).

  python public_rename.py --repo ~/myrepo --profile you            # отчёт
  python public_rename.py --repo ~/myrepo --profile you --in-place # ЗАПИСЬ
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codeprint.composer import load_profile
from codeprint.naming_axis import build_vocab_from_model, make_renamer
from codeprint.public_names import run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--report", default=None, help="куда писать json-отчёт")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    naming = profile.get("axes", {}).get("naming")
    if not naming:
        print("в профиле нет оси naming (нужен vocab_model)", file=sys.stderr)
        return 1
    vocab = build_vocab_from_model(
        Path(naming["vocab_model"]) / "meta.json")
    renamer = make_renamer(vocab, seed=naming.get("seed", ""))

    report = run(Path(args.repo), renamer, in_place=args.in_place)
    report["profile"] = args.profile
    report["repo"] = args.repo

    print(f"экспортов найдено: {report['exports_found']}")
    print(f"безопасных: {report['safe']}, переименовано: {report['renamed']}")
    print(f"отклонено: {len(report['rejected'])}")
    for n, r in list(report["rejected"].items())[:10]:
        print(f"  ! {n}: {', '.join(r)}")
    print(f"файлов затронуто: {report['changed_files']}")
    if not args.in_place:
        print("(dry-run: файлы НЕ изменены; --in-place для записи)")

    out = args.report or "reports/public-rename.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
