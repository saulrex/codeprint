#!/usr/bin/env python3
"""wizard — мини-прога сбора датасета стиля автора.

Проводит по шагам:
  1/4 путь к репозиторию (проверяет, что это git-репо)
  2/4 кто ты: показывает авторов истории с числом коммитов, выбираешь свои подписи
  3/4 дедлайн-дата: коммиты позже неё считаются «нейроночными» и не берутся
  4/4 куда сохранить (по умолчанию ./datasets/<имя-репо>/)

Сохранёнки в целевой папке: blanklines.jsonl (датасет), manifest.json (статистика
harvest), wizard.json (параметры запуска — чтобы повторить «как было»).

Все вопросы можно обойти флагами: --repo/--email/--before/--out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # работает из любого cwd

from codeprint.harvest import main as harvest_main


def run_git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


def list_authors(repo: Path) -> list[tuple[int, str, str]]:
    """(число коммитов, имя, email), по убыванию."""
    counts: dict[tuple[str, str], int] = {}
    for line in run_git(repo, "log", "--format=%an%x09%ae").splitlines():
        if "\t" in line:
            name, email = line.split("\t", 1)
            key = (name, email)
            counts[key] = counts.get(key, 0) + 1
    return sorted(((c, n, e) for (n, e), c in counts.items()), reverse=True)


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or (default or "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wizard", description=__doc__)
    ap.add_argument("--repo", help="путь к репозиторию")
    ap.add_argument("--email", action="append", help="твой email в истории; можно несколько раз")
    ap.add_argument("--before", help="дедлайн YYYY-MM-DD")
    ap.add_argument("--out", help="корень для сохранёнок (по умолчанию ./datasets)")
    args = ap.parse_args(argv)

    print("=== codeprint harvest wizard ===\n")

    # 1/4 репозиторий
    raw_repo = args.repo or ask("1/4 Путь к репозиторию")
    repo = Path(os.path.expanduser(raw_repo.strip('"')))
    try:
        run_git(repo, "rev-parse", "--git-dir")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"не похоже на git-репозиторий: {repo} ({e})", file=sys.stderr)
        return 1

    # 2/4 чей код
    emails = args.email
    if not emails:
        authors = list_authors(repo)
        print("\n2/4 Кто из них ты? (номера через пробел, например: 1 4)")
        for i, (c, name, email) in enumerate(authors[:15], 1):
            print(f"  {i:2d}. {c:6d} коммитов  {name} <{email}>")
        try:
            pick = input("   > ").split()
            emails = sorted({authors[int(p) - 1][2] for p in pick})
        except (ValueError, IndexError):
            print("не понял выбор", file=sys.stderr)
            return 1
        if not emails:
            print("не выбрано ни одной подписи", file=sys.stderr)
            return 1
    print(f"   твои подписи: {', '.join(emails)}")

    # 3/4 дедлайн
    before = args.before
    if not before:
        last = run_git(repo, "log", "-1", "--format=%ad", "--date=short").strip()
        before = ask(f"\n3/4 Дедлайн YYYY-MM-DD: позже неё код считается нейроночным", last)
    try:
        dt.date.fromisoformat(before)
    except ValueError:
        print(f"дата должна быть YYYY-MM-DD, получено {before!r}", file=sys.stderr)
        return 1

    # 4/4 куда сохранять
    if args.out:
        out_root = Path(os.path.expanduser(args.out))
    elif getattr(sys, "frozen", False):
        out_root = Path.cwd() / "datasets"  # exe: рядом с запуском, не в _MEI
    else:
        out_root = Path(__file__).resolve().parent / "datasets"
    out_dir = out_root / repo.name / before  # сохранёнки версионируются дедлайном
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n4/4 Сохранёнки: {out_dir}\n")

    argv_h = ["--repo", str(repo), "--before", before, "--out", str(out_dir)]
    for e in emails:
        argv_h += ["--author", e]
    rc = harvest_main(argv_h)
    if rc != 0:
        return rc

    (out_dir / "wizard.json").write_text(json.dumps({
        "repo": str(repo),
        "emails": emails,
        "before": before,
        "ran_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nСохранено: blanklines.jsonl + manifest.json + wizard.json")
    print("Повторить этот же сбор одной командой:")
    flags = " ".join(f"--email {e}" for e in emails)
    print(f'  python wizard.py --repo "{repo}" --before {before} {flags}')
    print("\nДальше: trainer — дообучение мелкой модели на этом датасете.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
