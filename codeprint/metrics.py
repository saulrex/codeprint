"""Фингерпринт вертикального ритма: вектор признаков и расстояние.

Это v1-стилометр: грубые статистики, которые позже заменит/дополнит
обученный классификатор. Уже сейчас он отвечает на главный вопрос —
«насколько расстановка пустых строк сдвинулась после рестайла».
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Fingerprint:
    total_lines: int = 0
    blank_density: float = 0.0          # доля пустых строк
    mean_paragraph: float = 0.0         # средняя длина «абзаца» кода
    p_blank_before_return: float = 0.0  # P(пустая строка перед return/yield/raise)
    p_blank_before_loop: float = 0.0    # P(пустая строка перед for/while)
    p_blank_before_comment: float = 0.0 # P(пустая строка перед комментарием)
    hist: tuple = (0.0, 0.0, 0.0, 0.0)  # длины абзацев: 1-2 / 3-5 / 6-10 / 11+

    def vector(self) -> list[float]:
        return [
            self.blank_density,
            self.mean_paragraph / 20.0,  # грубая нормализация к ~0..1
            self.p_blank_before_return,
            self.p_blank_before_loop,
            self.p_blank_before_comment,
            *self.hist,
        ]

    def distance(self, other: "Fingerprint") -> float:
        """Косинусное расстояние: 0 = стили совпадают, ~1 = максимально различны."""
        a, b = self.vector(), other.vector()
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return 1.0 - dot / (na * nb)


def fingerprint(source: str) -> Fingerprint:
    lines = source.splitlines()
    total = len(lines)
    blanks = sum(1 for l in lines if not l.strip())

    paragraphs: list[int] = []
    run = 0
    for l in lines:
        if l.strip():
            run += 1
        elif run:
            paragraphs.append(run)
            run = 0
    if run:
        paragraphs.append(run)

    def _rate(prefixes: tuple[str, ...]) -> float:
        hit = tot = 0
        for i, l in enumerate(lines):
            if l.strip().startswith(prefixes):
                tot += 1
                if i > 0 and not lines[i - 1].strip():
                    hit += 1
        return hit / tot if tot else 0.0

    buckets = [0, 0, 0, 0]
    for p in paragraphs:
        buckets[0 if p <= 2 else 1 if p <= 5 else 2 if p <= 10 else 3] += 1
    s = sum(buckets) or 1

    return Fingerprint(
        total_lines=total,
        blank_density=blanks / total if total else 0.0,
        mean_paragraph=sum(paragraphs) / len(paragraphs) if paragraphs else 0.0,
        p_blank_before_return=_rate(("return", "yield", "raise")),
        p_blank_before_loop=_rate(("for ", "while ")),
        p_blank_before_comment=_rate(("#",)),
        hist=tuple(b / s for b in buckets),
    )


def fingerprint_many(sources: list[str]) -> Fingerprint:
    """Взвешенный по строкам агрегат по нескольким файлам (репозиторию)."""
    fps = [fingerprint(s) for s in sources]
    w = [max(fp.total_lines, 1) for fp in fps]
    total_w = sum(w) or 1

    def avg(attr: str) -> float:
        return sum(getattr(fp, attr) * wi for fp, wi in zip(fps, w)) / total_w

    return Fingerprint(
        total_lines=sum(fp.total_lines for fp in fps),
        blank_density=avg("blank_density"),
        mean_paragraph=avg("mean_paragraph"),
        p_blank_before_return=avg("p_blank_before_return"),
        p_blank_before_loop=avg("p_blank_before_loop"),
        p_blank_before_comment=avg("p_blank_before_comment"),
        hist=tuple(
            sum(fp.hist[i] * wi for fp, wi in zip(fps, w)) / total_w for i in range(4)
        ),
    )
