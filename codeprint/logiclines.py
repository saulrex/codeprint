"""Анализ физических/логических строк исходника через stdlib tokenize.

Здесь НЕТ стилевых решений — только факты о файле:
- где заканчиваются логические строки (там и только там можно ставить разрывы);
- какие строки лежат внутри многострочных строковых литералов (их трогать нельзя).
"""
from __future__ import annotations

import io
import tokenize


def logical_end_lines(source: str) -> set[int]:
    """1-based номера физических строк, где заканчивается логическая строка.

    Именно после этих строк разрешено вставлять/удалять пустые строки:
    гарантируется, что мы не разорвём выражение в скобках или продолжение
    через backslash.
    """
    ends: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NEWLINE:
            ends.add(tok.end[0])
    return ends


def string_body_lines(source: str) -> set[int]:
    """1-based номера строк, покрытых многострочными строковыми литералами.

    Пустая строка внутри triple-quoted строки — это КОНТЕНТ, а не стиль.
    Такие строки restyler обязан сохранить как есть.
    """
    protected: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0], tok.end[0] + 1))
    return protected
