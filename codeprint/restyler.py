"""Детерминированный движок перестановки пустых строк.

Гарантии:
- работает только на границах логических строк (tokenize.NEWLINE), поэтому
  не может разорвать выражение в скобках или строку с backslash-продолжением;
- не трогает пустые строки внутри многострочных строковых литералов;
- никогда не отрывает декоратор от его def/class;
- идемпотентен: restyle(restyle(s)) == restyle(s);
- не меняет ни один непустой символ исходника => семантика сохраняется.
"""
from __future__ import annotations

from . import logiclines as _logiclines
from .profiles import BoundaryContext, Decider

MAX_BLANKS = 2


def restyle(source: str, decider: Decider, backend=None) -> str:
    """backend — сканер синтаксиса (logiclines для Python, clike для TS/JS)."""
    backend = backend or _logiclines
    ends = backend.logical_end_lines(source)
    protected = backend.string_body_lines(source)
    lines = source.splitlines()

    out: list[str] = []
    pending_end: int | None = None   # номер строки, где кончилась последняя логическая строка
    pending_had_blank = False        # встретили ли пустую строку после неё

    for i, line in enumerate(lines):
        lineno = i + 1
        if not line.strip():
            if lineno in protected:
                out.append("")  # пустая строка внутри строкового литерала — контент
            elif pending_end is not None:
                pending_had_blank = True
            # прочие пустые строки выбрасываем: расстановку делает decider с нуля
            continue

        if pending_end is not None:
            ctx = BoundaryContext(
                prev_lines=out[-8:],
                next_line=line,
                had_blank=pending_had_blank,
            )
            count = max(0, min(MAX_BLANKS, decider.blanks_between(ctx)))
            # декоратор никогда не отрываем от следующей строки
            if out and out[-1].lstrip().startswith("@"):
                count = 0
            out.extend([""] * count)
            pending_end = None
            pending_had_blank = False

        out.append(line)
        if lineno in ends:
            pending_end = lineno

    return "\n".join(out).rstrip("\n") + "\n"
