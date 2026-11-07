"""Эвристический сканер C-подобного синтаксиса (TS/TSX/JS/JSX) для оси blanklines.

Заменяет logiclines (который завязан на Python tokenize) для языков со
скобочным синтаксисом. Интерфейс тот же: logical_end_lines + string_body_lines.

«Конец логической строки» — строка, последний значащий символ которой
`;`, `{` или `}` при нулевой глубине () и [] и вне строк/комментариев.
Защищёнными (контентными) считаются строки внутри template-литералов (`` ` ``)
и блочных комментариев /* */.

Известные ограничения (приняты сознательно: это разметка датасета, не компилятор):
- regex-литералы (/[;}]/ и т.п.) не распознаются и могут дать шумовую границу;
- разделённые backslash'ом строки не склеиваются.
"""
from __future__ import annotations

LANG = "clike"

_END_CHARS = ";{}"


def _scan(source: str) -> tuple[set[int], set[int]]:
    ends: set[int] = set()
    protected: set[int] = set()

    state = "normal"  # normal | block_comment | sq | dq | template
    paren = 0         # глубина () и []
    brace = 0         # глубина {}
    tstack: list[int] = []  # глубины {} при входе в ${...} внутри template

    for lineno, line in enumerate(source.split("\n"), start=1):
        start_state = state
        last_sig = ""
        i, m = 0, len(line)

        while i < m:
            c = line[i]
            nxt = line[i + 1] if i + 1 < m else ""

            if state == "normal":
                if c == "/" and nxt == "/":
                    break  # остаток строки — строчный комментарий
                if c == "/" and nxt == "*":
                    state = "block_comment"; i += 2; continue
                if c == "'":
                    state = "sq"
                elif c == '"':
                    state = "dq"
                elif c == "`":
                    state = "template"
                elif c in "([":
                    paren += 1
                elif c in ")]":
                    paren = max(0, paren - 1)
                elif c == "{":
                    brace += 1
                elif c == "}":
                    if tstack and brace == tstack[-1] + 1:
                        # закрытие интерполяции ${...} — возвращаемся в template
                        brace -= 1
                        tstack.pop()
                        state = "template"
                        i += 1
                        continue
                    brace = max(0, brace - 1)
                if not c.isspace():
                    last_sig = c
            elif state == "block_comment":
                if c == "*" and nxt == "/":
                    state = "normal"; i += 2; continue
            elif state in ("sq", "dq"):
                if c == "\\":
                    i += 2; continue
                if (state == "sq" and c == "'") or (state == "dq" and c == '"'):
                    state = "normal"
                    last_sig = c
            elif state == "template":
                if c == "\\":
                    i += 2; continue
                if c == "`":
                    state = "normal"
                    last_sig = c
                elif c == "$" and nxt == "{":
                    tstack.append(brace)
                    brace += 1
                    state = "normal"
                    i += 2
                    continue
            i += 1

        # ' и " не живут через перевод строки — восстанавливаемся
        if state in ("sq", "dq"):
            state = "normal"

        if start_state in ("template", "block_comment") or state in ("template", "block_comment"):
            protected.add(lineno)

        if state == "normal" and paren == 0 and last_sig and last_sig in _END_CHARS:
            ends.add(lineno)

    return ends, protected


def logical_end_lines(source: str) -> set[int]:
    return _scan(source)[0]


def string_body_lines(source: str) -> set[int]:
    return _scan(source)[1]
