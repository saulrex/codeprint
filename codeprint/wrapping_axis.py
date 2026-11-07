"""wrapping — третья стилевая ось: перенос длинных логических строк.

Пример для обучения: одна логическая строка (стейтмент).
label: 1 — автор разбил её на несколько физических строк, 0 — оставил в одну.

Ключевой признак — длина в символах, но не только: вложенность, вид
конструкции, присваивание vs вызов. Поэтому ось нейронная, а не пороговая.

Ограничения v1:
- TS/JS: объекты/интерфейсы, открывающиеся `{`, считаются однострочными
  (у clike `{` — конец логической строки); переносы ловятся в вызовах,
  цепочках, условиях и т.п.;
- стейтменты длиннее 1000 символов пропускаем (генерация/минификация).
"""
from __future__ import annotations

import re

from codeprint.profiles import _indent, _is_assign, _kw

MAX_STMT_CHARS = 1000


def extract_wrapping_examples(source: str, backend) -> list[dict]:
    lang = getattr(backend, "LANG", "python")
    try:
        ends = backend.logical_end_lines(source)
        protected = backend.string_body_lines(source)
    except Exception:
        return []

    lines = source.splitlines()
    out: list[dict] = []
    start: int | None = None  # 0-based первая физическая строка стейтмента

    for i, line in enumerate(lines):
        lineno = i + 1
        s = line.strip()
        if not s or s.startswith(("#", "//")):
            continue  # пустые и comment-only строки не участвуют
        if lineno in protected:
            start = None  # многострочный литерал/блочный комментарий: шум
            continue
        if start is None:
            start = i
        if lineno in ends:
            stmt = lines[start:i + 1]
            start = None
            first = stmt[0]
            # Нормализация к «гипотетической однострочной» форме: склеиваем
            # стейтмент в одну строку. Иначе модель учит утечки формата
            # («первая строка без ';' == перенос»), а не решение автора.
            joined = " ".join(l.strip() for l in stmt)
            if len(joined) > MAX_STMT_CHARS:
                continue
            out.append({
                "axis": "wrapping",
                "lang": lang,
                "label": int(len(stmt) > 1),
                "prev_tail": [joined],
                "next_line": joined,
                "prev_indent": _indent(first),
                "next_indent": _indent(first),
                "prev_ends_colon": first.strip().endswith((":", "{")),
                "next_is_comment": False,
                "prev_is_assign": _is_assign(first.strip()),
                "next_is_assign": False,
                "n_chars": len(joined),
                "first_kw": _kw(first),
            })
    return out


# ---------- применение (только тесты/bench; репозитории читаем) ----------

def detect_indent_unit(source: str) -> int:
    """2 или 4 пробела — по первой indented-строке."""
    for l in source.splitlines():
        if l.strip():
            ind = _indent(l)
            if ind > 0:
                return 2 if ind % 4 != 0 else 4
    return 4


def _split_top_level(inner: str) -> list[str]:
    """Разбить содержимое скобок по запятым верхнего уровня (строки пропускаем)."""
    parts, buf, depth = [], [], 0
    i, n = 0, len(inner)
    quote = ""
    while i < n:
        c = inner[i]
        if quote:
            buf.append(c)
            if c == "\\":
                i += 1
                if i < n:
                    buf.append(inner[i])
            elif c == quote:
                quote = ""
        elif c in "'\"`":
            quote = c
            buf.append(c)
        elif c in "([{":
            depth += 1
            buf.append(c)
        elif c in ")]}":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


# Слова перед "(" означают управляющую конструкцию или группировку,
# а не вызов: там trailing comma — синтаксическая ошибка (JS) или
# кортеж-бомба (Python: `if (a,):` — всегда истина!)
_CONTROL_BEFORE_PAREN = {
    "if", "for", "while", "switch", "catch", "return", "throw", "typeof",
    "case", "delete", "void", "in", "of", "await", "yield", "elif", "not",
    "and", "or", "assert", "with", "lambda",
}


def wrap_statement(line: str, unit: int) -> list[str] | None:
    """Длинную однострочную строку -> перенос по запятым внешних скобок ВЫЗОВА.

    None — если переносить не за что или скобки не вызов (if/for/return/...).
    """
    s = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    depth = 0
    open_idx = close_idx = None
    quote = ""
    i = 0
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = ""
        elif c in "'\"`":
            quote = c
        elif c == "(":
            if depth == 0 and open_idx is None:
                open_idx = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and open_idx is not None:
                if s[open_idx + 1:i].strip():
                    close_idx = i  # первая НЕПУСТАЯ пара скобок
                    break
                open_idx = None  # пустые скобки (factory()...) — ищем дальше
        i += 1
    if open_idx is None or close_idx is None:
        return None
    # скобки — это список аргументов? слово перед ними не должно быть
    # управляющим ключевым словом; «foo(», «)(», «](» — вызовы
    head = s[:open_idx]
    m = re.search(r"[A-Za-z_$][\w$]*$", head)
    if m and m.group(0) in _CONTROL_BEFORE_PAREN:
        return None
    if not m and not head.rstrip().endswith((")", "]")):
        return None
    parts = _split_top_level(s[open_idx + 1:close_idx])
    if not parts:
        return None
    pad = indent + " " * unit
    out = [indent + s[: open_idx + 1]]
    out += [pad + p + "," for p in parts]
    out.append(indent + s[close_idx:])
    return out


def apply_wrapping(source: str, backend, predict) -> str:
    """predict(ex_dict) -> 0/1. Стейтменты переносятся/склеиваются по решению.

    Консервативно: склейка пропускается при комментариях/литералах внутри,
    перенос — только по запятым внешних круглых скобок.
    """
    lang = getattr(backend, "LANG", "python")
    try:
        ends = backend.logical_end_lines(source)
        protected = backend.string_body_lines(source)
    except Exception:
        return source

    comment_mark = "#" if lang == "python" else "//"
    unit = detect_indent_unit(source)
    lines = source.splitlines()
    out: list[str] = []
    stmt_start: int | None = None

    def flush_statement(end_i: int) -> None:
        nonlocal stmt_start
        s0 = stmt_start
        stmt_start = None
        stmt = lines[s0:end_i + 1]
        joined = " ".join(l.strip() for l in stmt)
        if len(joined) > MAX_STMT_CHARS:
            out.extend(stmt)
            return
        first = stmt[0]
        ex = {
            "prev_tail": [joined], "next_line": joined,
            "prev_indent": _indent(first), "next_indent": _indent(first),
            "prev_ends_colon": first.strip().endswith((":", "{")),
            "next_is_comment": False,
            "prev_is_assign": _is_assign(first.strip()),
            "next_is_assign": False,
            "n_chars": len(joined), "label": 0,
        }
        pred = predict(ex)
        if len(stmt) == 1 and pred == 1:
            wrapped = wrap_statement(first, unit)
            out.extend(wrapped if wrapped else stmt)
        elif len(stmt) > 1 and pred == 0:
            # склейка безопасна только без литералов/комментариев внутри
            safe = all(
                (s0 + k + 1) not in protected and comment_mark not in l
                for k, l in enumerate(stmt)
            )
            if safe:
                # убираем артефакты разбивки: "( x", висячая запятая перед ")"
                joined = re.sub(r",\s*\)", ")", joined)
                joined = re.sub(r"\(\s+", "(", joined)
                out.append((" " * _indent(first)) + joined)
            else:
                out.extend(stmt)
        else:
            out.extend(stmt)

    for i, line in enumerate(lines):
        lineno = i + 1
        s = line.strip()
        if not s or s.startswith(("#", "//")):
            if stmt_start is None:
                out.append(line)
            else:
                # комментарий/пустая внутри стейтмента: не склеиваем через неё
                out.extend(lines[stmt_start:i])
                stmt_start = None
                out.append(line)
            continue
        if lineno in protected:
            if stmt_start is None:
                out.append(line)
            else:
                out.extend(lines[stmt_start:i])
                stmt_start = None
                out.append(line)
            continue
        if stmt_start is None:
            stmt_start = i
        if lineno in ends:
            flush_statement(i)
    if stmt_start is not None:
        out.extend(lines[stmt_start:])
    return "\n".join(out) + "\n"
