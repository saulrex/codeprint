"""filefeat — файл-уровневые мульти-осевые признаки для stylometer v3.

Вместо классификации отдельных границ (v2) — один вектор на файл:
как автор расставляет пустые строки, когда переносит длинные строки,
как раскладывает импорты. Признаки интерпретируемые, утечек формата нет.

Один пример = один файл (blob). Регистрируется в harvest как ось filefeat.
"""
from __future__ import annotations

import math
from statistics import mean

#: Фиксированный порядок признаков — контракт между train и inference.
FEATURE_KEYS = [
    "log_lines",          # log1p(строк в файле)
    "blank_rate",         # доля границ с пустой строкой
    "p_blank_return",     # P(пустая перед return/yield/raise)
    "p_blank_loop",       # P(пустая перед for/while)
    "p_blank_comment",    # P(пустая перед комментарием)
    "wrap_rate",          # доля перенесённых стейтментов
    "wrap_60_90",         # P(wrap) для 60-90 символов
    "wrap_90_120",        # P(wrap) для 90-120
    "wrap_120p",          # P(wrap) для 120+
    "mean_stmt_chars",    # средняя длина стейтмента (норм. /200)
    "imports_dominant",   # доминантная схема импортов /3 (или -0.33 если нет)
    "log_import_blocks",  # log1p(числа блоков импортов)
]

_RETURN_KW = {"return", "yield", "raise"}
_LOOP_KW = {"for", "while"}


def extract_filefeat(source: str, backend) -> list[dict]:
    """Один пример на файл; пустой список — если файл бедный на границы."""
    from codeprint.harvest import extract_blankline_examples  # лениво: циклы
    from codeprint.imports_axis import extract_imports_examples
    from codeprint.wrapping_axis import extract_wrapping_examples

    bl = extract_blankline_examples(source, backend)
    if len(bl) < 5:
        return []
    wr = extract_wrapping_examples(source, backend)
    im = extract_imports_examples(source, backend)

    def rate(exs, pred=lambda e: True):
        sub = [e for e in exs if pred(e)]
        return float(mean(min(e["label"], 1) for e in sub)) if sub else 0.0

    def wrap_bucket(lo, hi):
        sub = [e for e in wr if lo <= e["n_chars"] < hi]
        return float(mean(e["label"] for e in sub)) if sub else 0.0

    scheme_counts = [0] * 4
    for e in im:
        scheme_counts[e["label"]] += 1
    dominant = scheme_counts.index(max(scheme_counts)) / 3.0 if any(scheme_counts) \
        else -1 / 3.0

    features = {
        "log_lines": math.log1p(len(source.splitlines())),
        "blank_rate": rate(bl),
        "p_blank_return": rate(bl, lambda e: e["next_kw"] in _RETURN_KW),
        "p_blank_loop": rate(bl, lambda e: e["next_kw"] in _LOOP_KW),
        "p_blank_comment": rate(bl, lambda e: e["next_is_comment"]),
        "wrap_rate": rate(wr) if wr else 0.0,
        "wrap_60_90": wrap_bucket(60, 90),
        "wrap_90_120": wrap_bucket(90, 120),
        "wrap_120p": wrap_bucket(120, 10**9),
        "mean_stmt_chars": min(mean(e["n_chars"] for e in wr) / 200.0, 1.0)
        if wr else 0.0,
        "imports_dominant": dominant,
        "log_import_blocks": math.log1p(len(im)),
    }
    return [{"axis": "filefeat", "lang": getattr(backend, "LANG", "python"),
             "label": 0, "features": features}]
