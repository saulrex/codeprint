"""harvest — сборщик чистого датасета стиля автора.

См. docs/ARCHITECTURE.md, модуль 8. Выгребает из git-репозиториев примеры
«контекст -> решение автора» по стилевым осям, отфильтрованные:

- по автору   (--author): коммитером может быть CI/бот, поэтому author;
- по дедлайну (--before): всё, что закоммичено позже даты, считается
  «загрязнённым» нейрогенерацией и в датасет НЕ попадает.

Выход: <out>/<axis>.jsonl + manifest.json — вход для trainer (модуль 9).
Поставка: PyInstaller --onefile harvest.py (входная точка в корне проекта).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from codeprint import clike, logiclines
from codeprint.profiles import _indent, _is_assign, _kw

#: Бэкенды сканирования по расширению файла.
BACKEND_BY_EXT = {
    ".py": logiclines,
    ".ts": clike, ".tsx": clike, ".js": clike, ".jsx": clike,
    ".mjs": clike, ".cjs": clike,
}
logiclines.LANG = "python"


# ---------- извлечение обучающих примеров ----------

def extract_blankline_examples(source: str, backend=logiclines) -> list[dict]:
    """Для каждой границы логических строк: контекст + сколько пустых строк
    поставил автор (label 0/1/2, 2 = «2 и больше»).

    backend — модуль сканера (logiclines для Python, clike для TS/JS/...).
    Пустые строки внутри многострочных литералов — контент, в разметку не идут.
    """
    try:
        ends = backend.logical_end_lines(source)
        protected = backend.string_body_lines(source)
    except Exception:
        return []  # битый/непарсящийся файл — просто пропускаем

    lines = source.splitlines()
    examples: list[dict] = []
    history: list[str] = []      # недавние непустые физические строки
    pending_end: int | None = None
    blank_count = 0

    for i, line in enumerate(lines):
        lineno = i + 1
        if not line.strip():
            if lineno in protected:
                history.append(line)
            elif pending_end is not None:
                blank_count += 1
            continue

        if pending_end is not None:
            prev = history[-1]
            examples.append({
                "axis": "blanklines",
                "lang": getattr(backend, "LANG", "python"),
                "label": min(blank_count, 2),
                "prev_tail": [l.rstrip() for l in history[-4:]],
                "next_line": line.rstrip(),
                "prev_indent": _indent(prev),
                "next_indent": _indent(line),
                "prev_kw": _kw(prev),
                "next_kw": _kw(line),
                "prev_ends_colon": prev.strip().endswith(":"),
                "next_is_comment": line.strip().startswith("#"),
                "prev_is_assign": _is_assign(prev.strip()),
                "next_is_assign": _is_assign(line.strip()),
            })
            pending_end = None
            blank_count = 0

        history.append(line)
        if lineno in ends:
            pending_end = lineno

    return examples


#: Реестр стилевых осей. Новые оси добавляются сюда.
from codeprint.filefeat_axis import extract_filefeat
from codeprint.comments_axis import extract_comments_examples
from codeprint.idioms_axis import extract_idioms_examples
from codeprint.imports_axis import extract_imports_examples
from codeprint.naming_axis import extract_naming_examples
from codeprint.structure_axis import extract_structure_examples
from codeprint.wrapping_axis import extract_wrapping_examples

AXES = {
    "blanklines": extract_blankline_examples,
    "imports": extract_imports_examples,
    "wrapping": extract_wrapping_examples,
    "filefeat": extract_filefeat,
    "naming": extract_naming_examples,
    "idioms": extract_idioms_examples,
    "comments": extract_comments_examples,
    "structure": extract_structure_examples,
}


# ---------- git-слой ----------

def run_git(args: list[str], cwd: Path | None = None) -> str:
    r = subprocess.run(
        ["git", *args], cwd=cwd,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def with_token(url: str, token: str | None) -> str:
    """Вшивает токен в https-URL (для приватных репо). В манифест не пишется."""
    if token and url.startswith("https://"):
        return "https://x-access-token:" + token + "@" + url[len("https://"):]
    return url


def harvest_repo(url: str, authors: list[str], before: str, token: str | None,
                 workdir: Path, axes: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """Клонирует репозиторий и собирает примеры. Возвращает (примеры по осям, статистику).

    Производительность: вместо ls-tree/show на каждый коммит — один проход
    `git log --raw` (даёт sha блобов изменённых файлов) + чтение блобов через
    один долгоживущий `git cat-file --batch`. На репо с 15k+ коммитов это
    разница между часами и секундами.
    Побочный плюс: --raw показывает только файлы, которые автор реально
    трогал в своих коммитах, — ещё и фильтр атрибуции.
    """
    dest = workdir / "repo"
    # NB: --shallow-since НЕ подходит — он отсекает историю ДО даты,
    # а нам нужно наоборот. Поэтому клонируем полностью и фильтруем логом.
    run_git(["clone", "--quiet", with_token(url, token), str(dest)])

    out = run_git([
        "log", "--no-merges", "--raw", "--format=commit %H",
        # несколько --author у git работают как ИЛИ — покрывает разные
        # подписи одного человека (имя/кириллица/другая почта)
        *[f"--author={a}" for a in authors],
        f"--before={before} 23:59:59",
    ], cwd=dest)

    commits = 0
    seen_blobs: set[str] = set()
    blobs: list[tuple[str, str]] = []  # (sha, ext), в порядке от новых к старым
    occurrences = 0
    for line in out.splitlines():
        if line.startswith("commit "):
            commits += 1
        elif line.startswith(":"):
            # формат: :100644 100644 <old_sha> <new_sha> M\t<path>
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) < 4:
                continue
            sha = parts[3]
            if sha == "0" * 40:  # удалённый файл
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in BACKEND_BY_EXT:
                continue
            occurrences += 1
            if sha not in seen_blobs:
                seen_blobs.add(sha)
                blobs.append((sha, ext))

    examples: dict[str, list[dict]] = {a: [] for a in axes}
    by_ext: dict[str, int] = {}
    files = 0

    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=dest,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    try:
        for sha, ext in blobs:
            proc.stdin.write(sha.encode() + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().decode("ascii", "replace").split()
            if len(header) < 3 or header[1] != "blob":
                continue
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)  # завершающий LF
            try:
                src = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            files += 1
            by_ext[ext] = by_ext.get(ext, 0) + 1
            backend = BACKEND_BY_EXT[ext]
            for axis in axes:
                examples[axis].extend(AXES[axis](src, backend))
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()

    stats = {
        "repo": url,
        "commits": commits,
        "files": files,
        "by_ext": by_ext,
        "dupe_blobs_skipped": occurrences - files,
        "examples": {a: len(examples[a]) for a in axes},
    }
    return examples, stats


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harvest", description=__doc__)
    ap.add_argument("--repo", action="append", required=True,
                    help="URL или локальный путь к репозиторию; можно несколько раз")
    ap.add_argument("--author", action="append", required=True,
                    help="подпись автора (git --author, regex); можно несколько раз "
                         "для разных подписей одного человека: --author me@example.com --author me2@example.com")
    ap.add_argument("--before", required=True,
                    help="дедлайн YYYY-MM-DD: коммиты позже этой даты не берём")
    ap.add_argument("--token", help="GitHub token для приватных репо / лимитов API")
    ap.add_argument("--out", default="dataset", help="куда писать датасет")
    ap.add_argument("--axes", default="blanklines",
                    help="стилевые оси через запятую (пока: blanklines)")
    args = ap.parse_args(argv)

    if shutil.which("git") is None:
        print("git не найден в PATH", file=sys.stderr)
        return 1

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = set(axes) - AXES.keys()
    if unknown:
        print(f"неизвестные оси: {', '.join(sorted(unknown))} "
              f"(доступны: {', '.join(AXES)})", file=sys.stderr)
        return 1

    # ранняя валидация даты, чтобы не клонировать зря
    try:
        dt.date.fromisoformat(args.before)
    except ValueError:
        print(f"--before: ожидалась дата YYYY-MM-DD, получено {args.before!r}",
              file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict]] = {a: [] for a in axes}
    repo_stats = []
    for url in args.repo:
        print(f">> {url}")
        with tempfile.TemporaryDirectory(prefix="harvest-") as tmp:
            try:
                examples, stats = harvest_repo(
                    url, args.author, args.before, args.token, Path(tmp), axes)
            except RuntimeError as e:
                print(f"!! пропускаю {url}: {e}", file=sys.stderr)
                continue
        for a in axes:
            merged[a].extend(examples[a])
        repo_stats.append(stats)
        print(f"   коммитов: {stats['commits']}, файлов: {stats['files']}, "
              f"дублей пропущено: {stats['dupe_blobs_skipped']}, "
              f"примеров: {sum(stats['examples'].values())}")

    for a in axes:
        path = out_dir / f"{a}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in merged[a]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    manifest = {
        "authors": args.author,
        "before": args.before,
        "axes": axes,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repos": repo_stats,
        "total_examples": {a: len(merged[a]) for a in axes},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nготово: {out_dir.resolve()}")
    for a in axes:
        print(f"  {a}.jsonl: {len(merged[a])} примеров")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
