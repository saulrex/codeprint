"""idioms — ось идиом: тернарник ↔ ранний return (v1, return-формы).

Prettier не трогает выбор конструкции — это личная привычка автора.
В данных front: if-else с двумя return не встречается вообще (0), ранний
return — 321 раз, тернарник — 801. Значит, конвертируемая пара такая:

  if (C) { return A; }      <->   return C ? A : B;   (TS/JS)
  return B;

  if C: return A            <->   return A if C else B (Python)
  return B

Гварды: optional chaining (?.), вложенные тернарники, многострочные ветки —
пропускаем. Ветки — только чистые выражения return.
"""
from __future__ import annotations

import re

# TS/JS: if (C) { return A; }\n return B;  — ранняя форма (без else)
_TS_IFELSE_RE = re.compile(
    r"(?P<indent>[ \t]*)if\s*\((?P<cond>[^(){}]+)\)\s*\{\s*"
    r"return\s+(?P<a>[^;{}]+?);\s*\}\s*\n\s*return\s+(?P<b>[^;{}]+?);")
# TS/JS: return C ? A : B;  (без ?. и без вложенных ?)
_TS_TERNARY_RE = re.compile(
    r"(?P<indent>[ \t]*)return\s+(?P<cond>[^;{}?]+?)\s*\?\s*"
    r"(?P<a>[^;{}:?]+?)\s*:\s*(?P<b>[^;{}]+?);")

# Python: if C:\n  return A\n  return B  (без else)
_PY_IFELSE_RE = re.compile(
    r"(?P<indent>[ \t]*)if\s+(?P<cond>[^\n:]+):\s*\n[ \t]*return\s+(?P<a>[^\n]+)\n"
    r"[ \t]*return\s+(?P<b>[^\n]+)")
# Python: return A if C else B
_PY_TERNARY_RE = re.compile(
    r"(?P<indent>[ \t]*)return\s+(?P<a>[^\n]+?)\s+if\s+(?P<c>[^\n]+?)\s+else\s+(?P<b>[^\n]+)")


def _ex(axis_label: int, cond: str, a: str, b: str, indent: int, lang: str) -> dict:
    return {
        "axis": "idioms",
        "lang": lang,
        "label": axis_label,  # 1 = тернарник, 0 = if-else
        "prev_tail": [f"if ({cond.strip()})"],
        "next_line": f"return {a.strip()}",
        "prev_indent": indent,
        "next_indent": indent,
        "prev_ends_colon": False,
        "next_is_comment": False,
        "prev_is_assign": False,
        "next_is_assign": False,
        "n_chars": len(cond) + len(a) + len(b),
        "first_kw": "if",
    }


def _line_of(source: str, pos: int) -> int:
    return source.count("\n", 0, pos) + 1


def _protected(source: str, backend) -> set[int]:
    try:
        return backend.string_body_lines(source)
    except Exception:
        return set()


def extract_idioms_examples(source: str, backend) -> list[dict]:
    lang = getattr(backend, "LANG", "python")
    protected = _protected(source, backend)
    out = []
    patterns = []
    if lang == "python":
        patterns = [(_PY_IFELSE_RE, 0, "cond"), (_PY_TERNARY_RE, 1, "c")]
    else:
        patterns = [(_TS_IFELSE_RE, 0, "cond"), (_TS_TERNARY_RE, 1, "cond")]
    for rx, label, ckey in patterns:
        for m in rx.finditer(source):
            span = {_line_of(source, m.start()), _line_of(source, m.end())}
            if span & protected:
                continue  # внутри template/строки — не трогаем
            if label == 1 and "?." in m.group(0):
                continue  # optional chaining — не наш паттерн
            out.append(_ex(label, m[ckey], m["a"], m["b"], len(m["indent"]), lang))
    return out


def apply_idioms(source: str, backend, ternary_max_chars: int) -> str:
    """Переносит идиомы в стиль автора.

    ternary_max_chars: если суммарная длина (cond+A+B) <= порога — тернарник,
    иначе if-else. Порог из профиля автора (examples/profile_idioms.py).
    """
    lang = getattr(backend, "LANG", "python")
    protected = _protected(source, backend)

    def blocked(m) -> bool:
        span = {_line_of(source, m.start()), _line_of(source, m.end())}
        return bool(span & protected)

    def to_ternary(cond, a, b, ind, lang):
        if lang == "python":
            return f"{ind}return {a.strip()} if {cond.strip()} else {b.strip()}"
        return f"{ind}return {cond.strip()} ? {a.strip()} : {b.strip()};"

    def to_ifelse(cond, a, b, ind, lang):
        inner = ind + ("    " if lang == "python" else "  ")
        if lang == "python":
            return (f"{ind}if {cond.strip()}:\n"
                    f"{inner}return {a.strip()}\n"
                    f"{ind}return {b.strip()}")
        return (f"{ind}if ({cond.strip()}) {{\n"
                f"{inner}return {a.strip()};\n"
                f"{ind}}}\n"
                f"{ind}return {b.strip()};")

    if lang == "python":
        def sub_if(m):
            if blocked(m):
                return m.group(0)
            total = len(m["cond"]) + len(m["a"]) + len(m["b"])
            if total <= ternary_max_chars:
                return to_ternary(m["cond"], m["a"], m["b"], m["indent"], lang)
            return m.group(0)

        def sub_tern(m):
            if blocked(m):
                return m.group(0)
            total = len(m["c"]) + len(m["a"]) + len(m["b"])
            if total <= ternary_max_chars:
                return m.group(0)
            return to_ifelse(m["c"], m["a"], m["b"], m["indent"], lang)

        source = _PY_IFELSE_RE.sub(sub_if, source)
        source = _PY_TERNARY_RE.sub(sub_tern, source)
    else:
        def sub_if(m):
            if blocked(m):
                return m.group(0)
            total = len(m["cond"]) + len(m["a"]) + len(m["b"])
            if total <= ternary_max_chars:
                return to_ternary(m["cond"], m["a"], m["b"], m["indent"], lang)
            return m.group(0)

        def sub_tern(m):
            if blocked(m) or "?." in m.group(0):
                return m.group(0)
            total = len(m["cond"]) + len(m["a"]) + len(m["b"])
            if total <= ternary_max_chars:
                return m.group(0)
            return to_ifelse(m["cond"], m["a"], m["b"], m["indent"], lang)

        source = _TS_IFELSE_RE.sub(sub_if, source)
        source = _TS_TERNARY_RE.sub(sub_tern, source)
    return source
