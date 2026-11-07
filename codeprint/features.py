"""Признаки границы логических строк для нейронной решалки.

Вынесено отдельно от trainer'а, чтобы не тянуть torch: этот модуль
zero-dep и используется на обоих концах — при обучении (trainer) и
при инференсе (neural.NeuralDecider).
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MAX_LEN = 96   # токенов на пример (берём хвост — ближе к границе важнее)
N_NUM = 9      # число структурных признаков (9-й — длина текста, добавлен в wrapping)


def tokenize(line: str) -> list[str]:
    return TOKEN_RE.findall(line.lower())


def ex_from_context(ctx) -> dict:
    """Собирает «пример harvest'а» из BoundaryContext рестайлера."""
    from codeprint.profiles import _indent, _is_assign

    prev = ctx.prev_lines[-1] if ctx.prev_lines else ""
    return {
        "prev_tail": ctx.prev_lines,
        "next_line": ctx.next_line,
        "prev_indent": _indent(prev),
        "next_indent": _indent(ctx.next_line),
        "prev_ends_colon": prev.strip().endswith(":"),
        "next_is_comment": ctx.next_line.strip().startswith(("#", "//")),
        "prev_is_assign": _is_assign(prev.strip()),
        "next_is_assign": _is_assign(ctx.next_line.strip()),
        "label": 0,  # на инференсе метки нет
    }


def encode_example(ex: dict, vocab: dict[str, int]) -> tuple[list[int], list[float], int]:
    """Пример -> (token ids[MAX_LEN], числовые признаки[N_NUM], label).

    Label возвращается как есть; слияние редких классов — забота trainer'а
    (флаг --binary), чтобы encoder был общим для всех осей.
    """
    toks: list[str] = []
    for l in ex["prev_tail"]:
        toks += tokenize(l)
    toks.append("<sep>")
    toks += tokenize(ex["next_line"])
    ids = [vocab.get(t, 1) for t in toks][-MAX_LEN:]
    ids = [0] * (MAX_LEN - len(ids)) + ids
    text_len = ex.get("n_chars") or (
        sum(len(l) for l in ex["prev_tail"]) + len(ex["next_line"]))
    nums = [
        ex["prev_indent"] / 40.0,
        ex["next_indent"] / 40.0,
        float(ex["prev_ends_colon"]),
        float(ex["next_is_comment"]),
        float(ex["prev_is_assign"]),
        float(ex["next_is_assign"]),
        min(len(ex["prev_tail"]), 8) / 8.0,
        min(len(toks), MAX_LEN) / MAX_LEN,
        min(text_len, 400) / 400.0,
    ]
    return ids, nums, int(ex["label"])
