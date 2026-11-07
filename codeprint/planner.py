"""planner — adversarial-луп: сам подбирает комбинацию осей под цель.

Жадный forward selection: стартуем с пустого набора осей, на каждом шаге
пробуем добавить каждую неиспользованную ось и оставляем ту, что даёт
лучший прирост целевой функции:

  score = Δстилометр - λ · доля изменённых строк

Останавливаемся, когда ни одна ось не улучшает score. Это первая версия
«атакующего» контура: дальше сюда же садится bandit поверх вариантов
конфигураций осей (пороги, схемы).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from codeprint.composer import BACKEND_BY_EXT, Composer, load_profile

LAMBDA = 0.1      # штраф за долю изменённых строк
EPSILON = 0.001   # минимальный значимый прирост


def _diff_ratio(before: str, after: str) -> float:
    bl = before.splitlines()
    al = after.splitlines()
    changed = sum(1 for a, b in itertools.zip_longest(bl, al) if a != b)
    return changed / max(len(bl), 1)


def evaluate(composer_cls, base_profile: dict, axes_subset: dict,
             source: str, ext: str, backend) -> tuple[float, float, float]:
    """-> (score, delta_stylo, diff_ratio)."""
    prof = dict(base_profile)
    prof["axes"] = axes_subset
    c = Composer(prof)
    before = composer_score(c, source, backend)
    new = c.apply(source, ext)
    after = composer_score(c, new, backend)
    if before != before or after != after:  # NaN
        delta = 0.0
    else:
        delta = after - before
    dr = _diff_ratio(source, new)
    return delta - LAMBDA * dr, delta, dr


def composer_score(c: Composer, source: str, backend) -> float:
    try:
        return c.score(source, backend)
    except Exception:
        return float("nan")


def plan(profile_name: str, source: str, ext: str) -> dict:
    backend = BACKEND_BY_EXT.get(ext.lower())
    base = load_profile(profile_name)
    available = dict(base.get("axes", {}))

    chosen: dict = {}
    best_score, _, _ = evaluate(Composer, base, {}, source, ext, backend)
    history = [{"step": 0, "axes": [], "score": round(best_score, 4)}]

    while True:
        best_axis, best_cfg = None, None
        for name, cfg in available.items():
            if name in chosen:
                continue
            trial = dict(chosen)
            trial[name] = cfg
            score, delta, dr = evaluate(Composer, base, trial, source, ext, backend)
            if score > best_score + EPSILON:
                best_score, best_axis, best_cfg = score, name, cfg
        if best_axis is None:
            break
        chosen[best_axis] = best_cfg
        history.append({"step": len(history), "axis": best_axis,
                        "score": round(best_score, 4)})

    # жадный выбор не видит взаимодействий осей — проверяем полный набор
    full_score, _, _ = evaluate(Composer, base, dict(available), source, ext,
                                backend)
    history.append({"step": len(history), "axis": "ВСЕ",
                    "score": round(full_score, 4)})
    if full_score > best_score:
        chosen = dict(available)
        best_score = full_score

    return {
        "chosen_axes": chosen,
        "final_score": best_score,
        "history": history,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="planner — автоподбор осей")
    ap.add_argument("--file", required=True)
    ap.add_argument("--profile", required=True)
    args = ap.parse_args()

    p = Path(args.file)
    source = p.read_text(encoding="utf-8")
    result = plan(args.profile, source, p.suffix)

    print(f"\nфайл: {p.name}, профиль: {args.profile}")
    for h in result["history"]:
        axis = h.get("axis", "—")
        print(f"  шаг {h['step']}: +{axis:12s} score={h['score']:.4f}")
    print(f"\nвыбранные оси: {list(result['chosen_axes'])}")
    print(f"финальный score: {result['final_score']:.4f}")
    return 0


