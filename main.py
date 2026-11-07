#!/usr/bin/env python3
"""codeprint — CLI.

Команды:
  stats    <path>                      снять фингерпринт с репозитория/файла
  restyle  <path> --profile NAME       переставить пустые строки (dry-run по умолчанию)
           [--in-place]                записать изменения в файлы
  profiles                             список доступных профилей
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from codeprint.metrics import Fingerprint, fingerprint_many
from codeprint.profiles import PROFILES
from codeprint.restyler import restyle


def iter_source_files(path: Path):
    """Все файлы поддерживаемых языков (Python + TS/JS)."""
    from codeprint.harvest import BACKEND_BY_EXT

    if path.is_file():
        if path.suffix.lower() in BACKEND_BY_EXT:
            yield path
        return
    skip = {".venv", "venv", "__pycache__", ".git", "node_modules", "dist", "build"}
    yield from sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in BACKEND_BY_EXT and not (set(p.parts) & skip))


def iter_py_files(path: Path):
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    skip = {".venv", "venv", "__pycache__", ".git", "node_modules"}
    yield from sorted(p for p in path.rglob("*.py") if not (set(p.parts) & skip))


def load_sources(path: Path):
    files, sources = [], []
    for f in iter_py_files(path):
        try:
            sources.append(f.read_text(encoding="utf-8"))
            files.append(f)
        except (OSError, UnicodeDecodeError) as e:
            print(f"!! не могу прочитать {f}: {e}", file=sys.stderr)
    return files, sources


def print_fp(label: str, fp: Fingerprint) -> None:
    hist = "[" + " ".join(f"{h:.2f}" for h in fp.hist) + "]"
    print(
        f"{label}: строк={fp.total_lines} плотность_пустых={fp.blank_density:.3f} "
        f"ср._абзац={fp.mean_paragraph:.2f} "
        f"пуст_перед[return]={fp.p_blank_before_return:.2f} "
        f"[loop]={fp.p_blank_before_loop:.2f} [comment]={fp.p_blank_before_comment:.2f} "
        f"гист={hist}"
    )


def cmd_stats(args) -> int:
    _, sources = load_sources(Path(args.path))
    if not sources:
        print("python-файлы не найдены", file=sys.stderr)
        return 1
    print_fp("фингерпринт", fingerprint_many(sources))
    return 0


def cmd_restyle(args) -> int:
    # именованный профиль автора (profiles/<name>.json) — все оси сразу
    profile_path = Path("profiles") / f"{args.profile}.json"
    if profile_path.is_file():
        return cmd_restyle_profile(args)

    decider = PROFILES[args.profile]()
    files, sources = load_sources(Path(args.path))
    if not files:
        print("python-файлы не найдены", file=sys.stderr)
        return 1

    new_sources: list[str] = []
    changed = 0
    for f, src in zip(files, sources):
        try:
            new = restyle(src, decider)
            compile(new, str(f), "exec")  # страховка: никогда не выпускаем битый код
        except Exception as e:
            print(f"!! пропускаю {f}: {e}", file=sys.stderr)
            new = src
        new_sources.append(new)
        if new != src:
            changed += 1
            if not args.in_place:
                sys.stdout.writelines(
                    difflib.unified_diff(
                        src.splitlines(True),
                        new.splitlines(True),
                        fromfile=f"{f} (было)",
                        tofile=f"{f} (стало)",
                    )
                )
            else:
                f.write_text(new, encoding="utf-8")

    before = fingerprint_many(sources)
    after = fingerprint_many(new_sources)
    print_fp("было ", before)
    print_fp("стало", after)
    print(
        f"\nдистанция(было -> стало) = {before.distance(after):.4f} "
        f"({changed} файл(ов) изменено)"
    )
    if not args.in_place and changed:
        print("(dry-run: добавь --in-place, чтобы записать изменения)")
    return 0


def cmd_restyle_profile(args) -> int:
    """Рестайл именованным профилем автора: все оси + замер стилометром."""
    from codeprint.composer import Composer, load_profile

    profile = load_profile(args.profile)
    composer = Composer(profile)
    files = list(iter_source_files(Path(args.path)))
    if not files:
        print("поддерживаемых файлов не найдено", file=sys.stderr)
        return 1
    if args.limit:
        files = files[: args.limit]

    from codeprint.harvest import BACKEND_BY_EXT

    changed = 0
    scores_before, scores_after = [], []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"!! не могу прочитать {f}: {e}", file=sys.stderr)
            continue
        backend = BACKEND_BY_EXT[f.suffix.lower()]
        try:
            new = composer.apply(src, f.suffix)
            if f.suffix == ".py":
                compile(new, str(f), "exec")  # страховка для Python
        except Exception as e:
            print(f"!! пропускаю {f}: {e}", file=sys.stderr)
            continue
        if args.measure:
            scores_before.append(composer.score(src, backend))
            scores_after.append(composer.score(new, backend))
        if new != src:
            changed += 1
            if not args.in_place:
                sys.stdout.writelines(
                    difflib.unified_diff(
                        src.splitlines(True), new.splitlines(True),
                        fromfile=f"{f} (было)", tofile=f"{f} (стало)",
                    )
                )
            else:
                f.write_text(new, encoding="utf-8")

    print(f"\nпрофиль: {args.profile}, файлов изменено: {changed}/{len(files)}")
    if args.measure and scores_before:
        import statistics
        b = statistics.mean(s for s in scores_before if s == s)
        a = statistics.mean(s for s in scores_after if s == s)
        print(f"stylometer P(автор): было {b:.3f} -> стало {a:.3f} "
              f"(сдвиг {a - b:+.3f})")
    if not args.in_place and changed:
        print("(dry-run: добавь --in-place, чтобы записать изменения)")
    return 0


def cmd_profiles(_args) -> int:
    for name, cls in PROFILES.items():
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:10s} {doc}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="codeprint", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="снять фингерпринт")
    p.add_argument("path")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("restyle", help="переставить пустые строки по профилю")
    p.add_argument("path")
    p.add_argument("--profile", default="paragraph",
                   help="ось blanklines (paragraph/compact/keep) или имя "
                        "профиля автора из profiles/<имя>.json (все оси)")
    p.add_argument("--in-place", action="store_true", help="записать в файлы")
    p.add_argument("--limit", type=int, default=None,
                   help="макс. файлов (для профилей автора)")
    p.add_argument("--measure", action="store_true",
                   help="мерить сдвиг stylometer'ом (профили автора)")
    p.set_defaults(fn=cmd_restyle)

    p = sub.add_parser("profiles", help="список профилей")
    p.set_defaults(fn=cmd_profiles)

    return ap.parse_args().fn(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
