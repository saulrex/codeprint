"""comments — ось комментариев: позиция (своя строка ↔ хвостом за кодом).

Только ПЕРЕМЕЩЕНИЕ существующих комментариев. Генерации нет по принципу.
Блочные /* */ в v1 считаем «своими строками» и не двигаем.

  const x = 1;  // пояснение   <->   // пояснение
                                     const x = 1;

Профиль автора: доля trailing-комментариев и порог длины.
"""
from __future__ import annotations

import io
import re
import tokenize

from codeprint.naming_axis import mask_strings
from codeprint.profiles import _indent


def _comments_python(source: str) -> list[dict]:
    """Комментарии через tokenize: (строка, col, текст, trailing?)."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                before = tok.line[: tok.start[1]]
                out.append({
                    "lineno": tok.start[0],
                    "col": tok.start[1],
                    "text": tok.string.lstrip("#").strip(),
                    "trailing": bool(before.strip()),
                    "line": tok.line,
                })
    except Exception:
        pass
    return out


def _comments_clike(source: str) -> list[dict]:
    try:
        from codeprint.clike import string_body_lines
        protected = string_body_lines(source)
    except Exception:
        protected = set()
    out = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in protected:
            continue  # внутри template-литерала '//' — это контент (CSS и т.п.)
        masked = mask_strings(line)
        m = re.search(r"//", masked)
        if not m:
            continue
        before = masked[: m.start()]
        out.append({
            "lineno": lineno,
            "col": m.start(),
            "text": line[m.start() + 2:].strip(),
            "trailing": bool(before.strip()),
            "line": line,
        })
    return out


def extract_comments_examples(source: str, backend) -> list[dict]:
    """label: 1 = комментарий хвостом за кодом, 0 = своей строкой."""
    lang = getattr(backend, "LANG", "python")
    comments = _comments_python(source) if lang == "python" \
        else _comments_clike(source)
    out = []
    for c in comments:
        code_part = c["line"][:c["col"]].strip() if c["trailing"] else ""
        out.append({
            "axis": "comments",
            "lang": lang,
            "label": int(c["trailing"]),
            "prev_tail": [code_part or "//"],
            "next_line": "// " + c["text"][:60],
            "prev_indent": _indent(c["line"]),
            "next_indent": _indent(c["line"]),
            "prev_ends_colon": code_part.endswith((":", "{")),
            "next_is_comment": True,
            "prev_is_assign": "=" in code_part,
            "next_is_assign": False,
            "n_chars": len(c["line"]),
            "comment_len": len(c["text"]),
        })
    return out


def apply_comments(source: str, backend, trailing_max_chars: int) -> str:
    """Переносит комментарии под профиль автора.

    trailing_max_chars: хвостовой комментарий допустим, если вся строка
    короче порога; длинные уезжают на свою строку выше. Своя строка
    подтягивается в хвост, если влезает в порог.
    """
    lang = getattr(backend, "LANG", "python")
    mark = "#" if lang == "python" else "//"
    try:
        protected = backend.string_body_lines(source)
    except Exception:
        protected = set()
    lines = source.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if (i + 1) in protected:
            out.append(line)  # контент многострочных литералов не трогаем
            continue
        masked = mask_strings(line)  # строки занулены для обоих языков
        m = re.search(re.escape(mark), masked)

        if not m:
            out.append(line)
            continue

        before = line[: m.start()].rstrip()
        comment = line[m.start():].strip()
        if before:
            if len(line) > trailing_max_chars:
                ind = " " * _indent(line)
                out.append(f"{ind}{comment}")
                out.append(before)
            else:
                out.append(line)
        else:
            # своя строка -> в хвост предыдущей, если та код и влезает
            if out and out[-1].strip() and not out[-1].strip().startswith(
                    ("#", "//")) and len(out[-1]) + len(comment) + 2 <= trailing_max_chars:
                out[-1] = out[-1].rstrip() + "  " + comment
            else:
                out.append(line)
    return "\n".join(out) + ("\n" if source.endswith("\n") else "")
