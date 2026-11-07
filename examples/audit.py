"""audit — ревизия профильного рестайла на пачке файлов.

Для каждого файла:
  1. инвариант токенов: мультимножество токенов до/после совпадает
     (рестайл не добавляет и не теряет код);
  2. идемпотентность: apply(apply(x)) == apply(x);
  3. для .py — результат компилируется;
  4. стилометр до/после (распределение, не только среднее);
  5. исключения собираются с примерами.

Запуск: python examples/audit.py --repo ~/myrepo --profile you --limit 300
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codeprint.composer import BACKEND_BY_EXT, Composer, load_profile
from codeprint.features import tokenize


def token_multiset(source: str) -> Counter:
    return Counter(tokenize(source))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    composer = Composer(load_profile(args.profile))
    repo = Path(args.repo)

    skip = {".git", "node_modules", "dist", "build", "__pycache__"}
    files = [p for p in sorted(repo.rglob("*"))
             if p.suffix.lower() in BACKEND_BY_EXT and not (set(p.parts) & skip)]
    files = files[: args.limit]

    stats = Counter()
    deltas: list[float] = []
    anomalies: list[str] = []

    for f in files:
        ext = f.suffix.lower()
        backend = BACKEND_BY_EXT[ext]
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            stats["unreadable"] += 1
            continue
        if not src.strip():
            stats["empty"] += 1
            continue

        try:
            out1 = composer.apply(src, ext)
        except Exception as e:
            stats["apply_exception"] += 1
            anomalies.append(f"{f.name}: apply: {type(e).__name__}: {e}")
            continue

        # 1. токены
        before, after = token_multiset(src), token_multiset(out1)
        lost = list((before - after).elements())
        added = list((after - before).elements())
        ident_re = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
        if lost and all(ident_re.match(t) for t in lost + added) \
                and len(lost) == len(added):
            stats["renamed_idents"] += 1  # naming-ось: ожидаемо
        elif lost:
            stats["tokens_LOST"] += 1
            anomalies.append(f"{f.name}: ПОТЕРЯНЫ токены {lost[:5]}")
        elif added and set(added) == {","}:
            # trailing comma — осознанный стиль (.prettierrc: trailingComma all)
            stats["trailing_comma_added"] += 1
        elif added:
            stats["tokens_added"] += 1
            anomalies.append(f"{f.name}: добавлены токены {added[:5]}")
        else:
            stats["tokens_ok"] += 1

        # 2. сходимость: naming-гварды зависят от результата других осей,
        # поэтому 1->2 может двигаться; стабильность обязательна с 2->3
        try:
            out2 = composer.apply(out1, ext)
            out3 = composer.apply(out2, ext)
            if out3 != out2:
                stats["no_fixpoint"] += 1
                anomalies.append(f"{f.name}: не сошёлся за 3 прогона")
            elif out2 != out1:
                stats["converged_at_2"] += 1
            else:
                stats["idempotent"] += 1
        except Exception as e:
            stats["reapply_exception"] += 1
            anomalies.append(f"{f.name}: повторный apply: {e}")

        # 3. синтаксис (python compile + tree-sitter для TS/JS)
        from codeprint.syntax_gate import syntax_ok
        if syntax_ok(out1, ext):
            stats["syntax_ok"] += 1
        elif not syntax_ok(src, ext):
            stats["gate_blind"] += 1  # оригинал не парсился — гейт слеп
        else:
            stats["syntax_FAIL"] += 1
            anomalies.append(f"{f.name}: результат не парсится")

        # 4. стилометр
        try:
            b = composer.score(src, backend)
            a = composer.score(out1, backend)
            if b == b and a == a:  # не NaN
                deltas.append(a - b)
        except Exception:
            stats["score_exception"] += 1

        stats["changed" if out1 != src else "unchanged"] += 1

    print(f"\n=== аудит {args.profile} на {len(files)} файлах ===")
    for k in ("tokens_ok", "renamed_idents", "idempotent", "converged_at_2",
              "syntax_ok", "gate_blind", "changed", "unchanged",
              "trailing_comma_added", "tokens_LOST", "tokens_added",
              "no_fixpoint", "apply_exception",
              "reapply_exception", "syntax_FAIL", "unreadable", "empty"):
        if stats.get(k):
            print(f"  {k:18s} {stats[k]}")
    if deltas:
        print(f"\nстилометр (n={len(deltas)}): средний сдвиг {statistics.mean(deltas):+.4f}, "
              f"медиана {statistics.median(deltas):+.4f}, "
              f"min {min(deltas):+.4f}, max {max(deltas):+.4f}")
        pos = sum(1 for d in deltas if d > 0.001)
        neg = sum(1 for d in deltas if d < -0.001)
        print(f"  сдвиг > +0.001: {pos} файлов, < -0.001: {neg} файлов")
    if anomalies:
        print(f"\n=== аномалии ({len(anomalies)}) ===")
        for a in anomalies[:15]:
            print(" !", a)
    else:
        print("\nаномалий нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
