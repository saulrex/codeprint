"""imports — вторая стилевая ось: раскладка блоков импортов.

Схема автора по блоку (label):
  0 flat_asis      — без пустых строк внутри блока, порядок как попало
  1 flat_alpha     — без пустых строк, алфавитный порядок
  2 grouped_alpha  — группы через пустую строку, внутри алфавит
  3 grouped_asis   — группы через пустую строку, порядок как попало

Трансформация semantics-safe: меняются только порядок import-строк и
пустые строки между ними; сами импорты не трогаем. Применение — ТОЛЬКО
в тестах/бенче (репозитории пока читаем).
"""
from __future__ import annotations

import re

SCHEMES = ["flat_asis", "flat_alpha", "grouped_alpha", "grouped_asis"]

_PY_STDLIB = {
    "os", "sys", "re", "io", "json", "math", "time", "datetime", "pathlib",
    "typing", "collections", "functools", "itertools", "subprocess", "tempfile",
    "argparse", "dataclasses", "abc", "copy", "enum", "hashlib", "logging",
    "unittest", "contextlib", "shutil", "random", "string", "warnings",
}


def is_import_line(line: str, lang: str) -> bool:
    s = line.strip()
    if lang == "python":
        return s.startswith("import ") or (s.startswith("from ") and " import " in s)
    return s.startswith("import ")  # TS/JS: import X from '...', import '...'


def import_key(line: str, lang: str) -> str:
    """Модуль/путь импорта — ключ сортировки и группировки."""
    s = line.strip()
    if lang == "python":
        m = re.match(r"from\s+([\w.]+)", s) or re.match(r"import\s+([\w.]+)", s)
        return (m.group(1) if m else s).lower()
    m = re.search(r"from\s+['\"]([^'\"]+)['\"]", s) or re.match(r"import\s+['\"]([^'\"]+)['\"]", s)
    return (m.group(1) if m else s).lower()


def import_group(key: str, lang: str) -> int:
    """0 — внешнее/стандартное, 1 — локальное (./ ../ или относительный from)."""
    if lang == "python":
        return 0 if key.split(".")[0] in _PY_STDLIB else 1
    return 1 if key.startswith(".") else 0


def find_import_blocks(source: str, lang: str) -> list[dict]:
    """Блоки импортов: серии import-строк, возможно разбитые пустыми строками.

    Возвращает [{start, end, lines, group_breaks}] (end — exclusive,
    захватывает внутренние пустые строки). Блоки из <3 импортов пропускаем.
    """
    lines = source.splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if not is_import_line(lines[i], lang):
            i += 1
            continue
        start = i
        cur: list[str] = []
        groups: list[list[str]] = [[]]
        while i < n:
            if is_import_line(lines[i], lang):
                cur.append(lines[i])
                groups[-1].append(lines[i])
                i += 1
                continue
            if not lines[i].strip():
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and is_import_line(lines[j], lang):
                    groups.append([])
                    i = j
                    continue
            break
        if len(cur) >= 3:
            blocks.append({"start": start, "end": i, "lines": cur,
                           "groups": groups, "group_breaks": len(groups) - 1})
    return blocks


def scheme_of(block: dict, lang: str) -> str:
    # алфавитность проверяем ВНУТРИ групп: «stdlib, потом внешние» — это
    # осмысленная группировка, а не хаотичный порядок
    alpha = all(
        [import_key(l, lang) for l in g] == sorted(import_key(l, lang) for l in g)
        for g in block["groups"] if g
    )
    if block["group_breaks"] > 0:
        return "grouped_alpha" if alpha else "grouped_asis"
    return "flat_alpha" if alpha else "flat_asis"


def apply_scheme(source: str, scheme: str, lang: str) -> str:
    """Перекладывает все блоки импортов в заданную схему (in-memory)."""
    lines = source.splitlines()
    for b in reversed(find_import_blocks(source, lang)):
        keys = [import_key(l, lang) for l in b["lines"]]
        grouped = scheme.startswith("grouped")
        alpha = scheme.endswith("alpha")
        if grouped:
            # сначала внешние/стандартные, потом локальные; внутри — по схеме
            order = sorted(
                range(len(keys)),
                key=lambda i: (import_group(keys[i], lang),
                               keys[i] if alpha else i))
        else:
            order = sorted(range(len(keys)),
                           key=lambda i: keys[i]) if alpha else list(range(len(keys)))
        new_block: list[str] = []
        prev_group = None
        for idx in order:
            g = import_group(keys[idx], lang) if grouped else 0
            if prev_group is not None and g != prev_group:
                new_block.append("")
            new_block.append(b["lines"][idx])
            prev_group = g
        lines[b["start"]:b["end"]] = new_block
    return "\n".join(lines) + "\n"


# ---------- извлечение примеров для harvest ----------

def extract_imports_examples(source: str, backend) -> list[dict]:
    lang = getattr(backend, "LANG", "python")
    out = []
    for b in find_import_blocks(source, lang):
        lines = b["lines"]
        out.append({
            "axis": "imports",
            "lang": lang,
            "label": SCHEMES.index(scheme_of(b, lang)),
            # поля под общий encoder (features.encode_example)
            "prev_tail": lines[:4],
            "next_line": lines[min(4, len(lines) - 1)],
            "prev_indent": 0,
            "next_indent": 0,
            "prev_ends_colon": False,
            "next_is_comment": False,
            "prev_is_assign": False,
            "next_is_assign": False,
            "n_imports": len(lines),
        })
    return out
