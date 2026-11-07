"""bench — read-only бенчмарк решалок по репозиторию.

Для каждого файла: запоминаем, где автор поставил пустые строки, стираем их
(ZeroDecider), восстанавливаем решалкой и меряем совпадение решений.
Репозиторий только читается — ни один файл не изменяется.

Запуск:
  python bench.py --repo ~/myrepo --ckpt models/blanklines/model.pt [--limit 500]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from codeprint.harvest import BACKEND_BY_EXT, extract_blankline_examples
from codeprint.profiles import Decider, ParagraphDecider
from codeprint.restyler import restyle


class ZeroDecider(Decider):
    """Стерилизатор: ни одной пустой строки."""

    def blanks_between(self, ctx) -> int:
        return 0


def boundary_labels(source: str, backend) -> dict[tuple[str, str], int]:
    return {
        (e["prev_tail"][-1].strip(), e["next_line"].strip()): min(e["label"], 1)
        for e in extract_blankline_examples(source, backend)
        if e["prev_tail"]
    }


def bench_source(source: str, backend, decider) -> tuple[int, int]:
    """-> (сопоставленных границ, совпавших решений)."""
    stripped = restyle(source, ZeroDecider(), backend)
    restored = restyle(stripped, decider, backend)
    a = boundary_labels(source, backend)
    b = boundary_labels(restored, backend)
    common = set(a) & set(b)
    return len(common), sum(1 for k in common if a[k] == b[k])


def iter_tracked_files(repo: Path):
    """Только отслеживаемые файлы (git ls-files) — node_modules не трогаем."""
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    for rel in out.splitlines():
        ext = Path(rel).suffix.lower()
        if ext in BACKEND_BY_EXT:
            yield repo / rel, BACKEND_BY_EXT[ext]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", default="models/blanklines/model.pt",
                    help="чекпоинт trainer'а; если доступен — добавится нейронка")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args(argv)

    deciders: dict[str, Decider] = {"paragraph(эвристика)": ParagraphDecider()}
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        try:
            from codeprint.neural import NeuralDecider
            deciders["neural"] = NeuralDecider(checkpoint=str(ckpt))
        except ImportError:
            print("!! нет ни onnxruntime, ни torch — нейронка пропущена, "
                  "только эвристика", file=sys.stderr)
    else:
        print(f"чекпоинт {ckpt} не найден — только эвристика", file=sys.stderr)

    results = {name: [0, 0] for name in deciders}  # [границы, совпадения]
    files = 0
    t0 = time.time()
    for path, backend in iter_tracked_files(Path(args.repo)):
        if files >= args.limit:
            break
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not src.strip():
            continue
        files += 1
        for name, decider in deciders.items():
            total, hits = bench_source(src, backend, decider)
            results[name][0] += total
            results[name][1] += hits

    elapsed = time.time() - t0
    print(f"\nрепозиторий: {args.repo} (read-only)")
    print(f"файлов обработано: {files} за {elapsed:.1f}с "
          f"({files / max(elapsed, 1e-9):.1f} файлов/с)")
    print(f"{'решалка':22s} {'границ':>8s} {'совпадение с автором':>22s}")
    for name, (total, hits) in results.items():
        pct = hits / total * 100 if total else 0.0
        print(f"{name:22s} {total:8d} {pct:21.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
