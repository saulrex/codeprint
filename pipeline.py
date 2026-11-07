#!/usr/bin/env python3
"""pipeline — весь цикл одной командой: harvest -> train -> bench.

Воспроизводимость «с нуля» на любой машине:
  python pipeline.py --repo <путь|url> --email <твой-email> --before YYYY-MM-DD

Обучение идёт на GPU, если torch видит CUDA, иначе на CPU (медленнее;
для быстрой проверки уменьши: --max-examples 50000 --epochs 1).
Инференсу в bench torch не нужен, если рядом с model.pt есть model.onnx
(trainer экспортирует его автоматически при установленном пакете onnx).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codeprint.bench import main as bench_main
from codeprint.harvest import main as harvest_main
from codeprint.trainer import main as trainer_main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--email", action="append", required=True,
                    help="твой email в истории; можно несколько раз")
    ap.add_argument("--before", required=True, help="дедлайн YYYY-MM-DD")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--bench-limit", type=int, default=500)
    args = ap.parse_args(argv)

    name = Path(args.repo.rstrip("/\\")).name
    data_dir = Path("datasets") / name / args.before
    model_dir = Path("models") / name  # модель привязана к репо-источнику

    print(f"\n=== шаг 1/3: harvest -> {data_dir} ===")
    hargv = ["--repo", args.repo, "--before", args.before, "--out", str(data_dir)]
    for e in args.email:
        hargv += ["--author", e]
    rc = harvest_main(hargv)
    if rc != 0:
        return rc

    print(f"\n=== шаг 2/3: train -> {model_dir} ===")
    targv = ["--data", str(data_dir / "blanklines.jsonl"),
             "--out", str(model_dir), "--epochs", str(args.epochs),
             "--binary"]  # blanklines: класс 2 («2+ пустых») сливаем в 1
    if args.max_examples:
        targv += ["--max-examples", str(args.max_examples)]
    rc = trainer_main(targv)
    if rc != 0:
        return rc

    print(f"\n=== шаг 3/3: bench (read-only) ===")
    return bench_main(["--repo", args.repo,
                       "--ckpt", str(model_dir / "model.pt"),
                       "--limit", str(args.bench_limit)])


if __name__ == "__main__":
    raise SystemExit(main())
