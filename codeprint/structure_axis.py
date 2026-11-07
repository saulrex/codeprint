"""structure — ось структуры: порядок методов в классах (v1, консервативно).

Семантически безопасно: порядок методов класса не влияет на поведение.
Гварды:
- классы с полями-инициализаторами пропускаем (порядок полей значим);
- constructor/__init__ всегда остаётся первым;
- геттер/сеттер двигаем как обычные методы;
- статические блоки и декораторы не трогаем (Python: декорированный метод
  переносится вместе со своими декораторами).

Схемы: "alpha" (по алфавиту), "ctor_first" (конструктор + остальные как были),
"private_last" (приватные в конец).
"""
from __future__ import annotations

import re

from codeprint.naming_axis import mask_code
from codeprint.profiles import _indent

SCHEMES = ["alpha", "ctor_first", "private_last"]

_TS_METHOD_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+)*"
    r"([a-zA-Z_$][\w$]*)\s*\(.*\)\s*[:{]")
_PY_METHOD_RE = re.compile(r"^(\s+)(?:@\w+\s*\n\s*)*def\s+(\w+)\s*\(")


def _brace_depths(lines: list[str]) -> list[int]:
    """Глубина {} на конец каждой строки (строки/комментарии вырезаны маской)."""
    depths = []
    d = 0
    for line in lines:
        code = mask_code(line)
        d += code.count("{") - code.count("}")
        depths.append(d)
    return depths


def _find_classes_ts(source: str) -> list[dict]:
    lines = source.splitlines()
    depths = _brace_depths(lines)
    classes = []
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+", lines[i])
        if not m:
            i += 1
            continue
        base = depths[i - 1] if i > 0 else 0
        # конец класса: строка, после которой глубина возвращается к base
        j = i + 1
        while j < len(lines) and depths[j] > base:
            j += 1
        # методы: строки на глубине base+1, начинающиеся с метода
        methods = []
        k = i + 1
        while k < j:
            if depths[k - 1] == base + 1 or (k == i + 1 and depths[k] > base):
                mm = _TS_METHOD_RE.match(lines[k])
                if mm and not mm.group(1).startswith(("if", "for", "while")):
                    # конец метода: строка, вернувшая глубину на base+1 (включая её)
                    e = k + 1
                    while e < j and depths[e] > base + 1:
                        e += 1
                    methods.append({"name": mm.group(1), "start": k, "end": e + 1})
                    k = e + 1
                    continue
            k += 1
        if methods:
            classes.append({"start": i, "end": j, "methods": methods})
        i = j + 1
    return classes


def _find_classes_py(source: str) -> list[dict]:
    lines = source.splitlines()
    classes = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)class\s+\w+", lines[i])
        if not m:
            i += 1
            continue
        base_ind = len(m.group(1))
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s and _indent(lines[j]) <= base_ind:
                break
            j += 1
        methods = []
        k = i + 1
        while k < j:
            dm = re.match(r"^(\s+)def\s+(\w+)\s*\(", lines[k])
            if dm and len(dm.group(1)) == base_ind + 4:
                # захватываем декораторы выше
                start = k
                while start > i + 1 and lines[start - 1].strip().startswith("@"):
                    start -= 1
                e = k + 1
                while e < j:
                    s = lines[e].strip()
                    if s and _indent(lines[e]) <= base_ind + 4:
                        break
                    e += 1
                methods.append({"name": dm.group(2), "start": start, "end": e})
                k = e
                continue
            k += 1
        if methods:
            classes.append({"start": i, "end": j, "methods": methods})
        i = j + 1
    return classes


def find_classes(source: str, backend) -> list[dict]:
    lang = getattr(backend, "LANG", "python")
    return _find_classes_py(source) if lang == "python" else _find_classes_ts(source)


def _is_private(name: str, lang: str) -> bool:
    return name.startswith("_") or name.startswith("#")


def scheme_of_class(cls: dict, lang: str) -> str:
    names = [m["name"] for m in cls["methods"]]
    rest = [n for n in names if n not in ("constructor", "__init__")]
    if rest == sorted(rest):
        return "alpha"
    if names and names[0] in ("constructor", "__init__"):
        return "ctor_first"
    privates_last = all(
        not _is_private(n, lang) for n in rest[: len(rest) - sum(
            _is_private(n, lang) for n in rest)])
    return "private_last" if privates_last else "custom"


def extract_structure_examples(source: str, backend) -> list[dict]:
    lang = getattr(backend, "LANG", "python")
    out = []
    for cls in find_classes(source, backend):
        scheme = scheme_of_class(cls, lang)
        if scheme == "custom":
            continue  # произвольный порядок — не обучающий пример
        names = [m["name"] for m in cls["methods"]]
        out.append({
            "axis": "structure",
            "lang": lang,
            "label": SCHEMES.index(scheme),
            "prev_tail": [f"class ... {names[0]}()"],
            "next_line": f"{names[-1]}()",
            "prev_indent": 0,
            "next_indent": 0,
            "prev_ends_colon": False,
            "next_is_comment": False,
            "prev_is_assign": False,
            "next_is_assign": False,
            "n_chars": sum(len(n) for n in names),
            "n_methods": len(names),
        })
    return out


def apply_structure(source: str, backend, scheme: str) -> str:
    """Переставляет методы классов по схеме. ctor/__init__ всегда первым."""
    lang = getattr(backend, "LANG", "python")
    lines = source.splitlines()
    for cls in reversed(find_classes(source, backend)):
        methods = cls["methods"]
        ctor = [m for m in methods if m["name"] in ("constructor", "__init__")]
        rest = [m for m in methods if m not in ctor]
        if scheme == "alpha":
            rest.sort(key=lambda m: m["name"])
        elif scheme == "private_last":
            rest.sort(key=lambda m: _is_private(m["name"], lang))
        # ctor_first: rest as-is
        ordered = ctor + rest
        if ordered == methods:
            continue
        # методы в классе идут подряд? собираем новый блок по кускам
        block_lines: list[str] = []
        cursor = methods[0]["start"]
        chunks = {m["start"]: lines[m["start"]:m["end"]] for m in methods}
        for m in ordered:
            block_lines += chunks[m["start"]]
        lines[methods[0]["start"]:methods[-1]["end"]] = block_lines
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")
