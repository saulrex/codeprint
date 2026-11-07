"""similarity — поверхностное сходство исходников (модуль 12).

Winnowing (как в MOSS): токены -> k-граммы -> хэши -> в каждом окне из w
хэшей берётся минимальный. Полученное множество отпечатков устойчиво к
мелким правкам и нечувствительно к переименованиям/переносам частично.

Метрики:
- containment(A в B) = |fp(A) ∩ fp(B)| / |fpA| — сколько A видно в B;
- jaccard = |∩| / |∪|.

Применение: после рестайла containment(оригинал -> результат) должен падать
(сходство отмыто), но не до нуля (логика та же). Для поиска заимствований:
containment(подозрительный -> источник) высокий = покрытие.
"""
from __future__ import annotations

import zlib

from codeprint.features import tokenize

K = 5   # длина k-граммы в токенах
W = 4   # окно винновинга


def _hashes(tokens: list[str]) -> list[int]:
    return [zlib.crc32(t.encode()) for t in tokens]


def fingerprint(source: str, k: int = K, w: int = W) -> frozenset[int]:
    # токены кода (слова + пунктуация), lowercase — регистр не важен
    tokens = tokenize(source)
    hashes = _hashes(tokens)
    if len(hashes) < k:
        return frozenset(hashes)
    grams = []
    for i in range(len(hashes) - k + 1):
        g = 0
        for h in hashes[i:i + k]:
            g = ((g << 8) ^ h) & 0xFFFFFFFFFFFFFFFF
        grams.append(g)
    if len(grams) < w:
        return frozenset(grams)
    fp = set()
    for i in range(len(grams) - w + 1):
        fp.add(min(grams[i:i + w]))
    return frozenset(fp)


def containment(a: str, b: str) -> float:
    """Доля отпечатков a, видимых в b: 1.0 = a полностью покрыт b."""
    fa, fb = fingerprint(a), fingerprint(b)
    if not fa:
        return 1.0
    return len(fa & fb) / len(fa)


def jaccard(a: str, b: str) -> float:
    fa, fb = fingerprint(a), fingerprint(b)
    union = fa | fb
    if not union:
        return 1.0
    return len(fa & fb) / len(union)


def report(before: str, after: str) -> dict:
    return {
        "containment_before_in_after": round(containment(before, after), 4),
        "containment_after_in_before": round(containment(after, before), 4),
        "jaccard": round(jaccard(before, after), 4),
    }
