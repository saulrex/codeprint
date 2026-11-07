"""naming — ось имён идентификаторов.

Prettier не трогает имена — поэтому словарь автора (любимые слова, длины,
сокращения) это то место, где личный почерк выживает в отформатированном
репозитории. Классическая стилометрия: имена — сильнейший сигнал авторства.

Пример = один файл: мультимножество слов идентификаторов + скалярные черты.
Слова из строковых литералов и комментариев исключаются (это контент, не имена).
Идентификаторы после точки (obj.prop) пропускаем: это чужие API, не выбор автора.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import keyword
import re
import tokenize
from pathlib import Path

IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
CAMEL_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

PY_STOP = set(keyword.kwlist) | set(dir(builtins))
JS_STOP = {
    "break", "case", "catch", "class", "const", "continue", "debugger",
    "default", "delete", "do", "else", "export", "extends", "finally", "for",
    "function", "if", "import", "in", "instanceof", "new", "return", "super",
    "switch", "this", "throw", "try", "typeof", "var", "void", "while",
    "with", "yield", "let", "static", "enum", "await", "implements",
    "interface", "package", "private", "protected", "public", "null",
    "true", "false", "undefined", "of", "as", "from", "type", "readonly",
    # частые DOM/JS-глобалы — не выбор автора
    "console", "window", "document", "Math", "JSON", "Object", "Array",
    "Promise", "String", "Number", "Boolean", "Date", "Error", "Map", "Set",
}

_STRING_RE = re.compile(
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`")


def mask_code(line: str) -> str:
    """Строка той же длины, где строковые литералы и // комментарии занулены."""
    masked = _STRING_RE.sub(lambda m: "\x00" * len(m.group(0)), line)
    return re.sub(r"//.*$", lambda m: "\x00" * len(m.group(0)), masked)


def split_words(ident: str) -> list[str]:
    """fetchTemplateParts / fetch_template_parts -> [fetch, template, parts]."""
    words: list[str] = []
    for chunk in ident.replace("$", "_").split("_"):
        words.extend(m.group(0) for m in CAMEL_SPLIT.finditer(chunk))
    return [w.lower() for w in words if w]


def _identifiers_python(source: str) -> list[str]:
    """NAME-токены через tokenize: строки и комментарии отсекаются нативно."""
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except Exception:
        return out
    for i, tok in enumerate(toks):
        if tok.type != tokenize.NAME or tok.string in PY_STOP:
            continue
        # пропускаем obj.attr — атрибут чужого API
        line_prefix = tok.line[: tok.start[1]].rstrip()
        if line_prefix.endswith("."):
            continue
        out.append(tok.string)
    return out


def _identifiers_clike(source: str, protected: set[int]) -> list[str]:
    out = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in protected:
            continue  # многострочный литерал/комментарий
        s = line.strip()
        if s.startswith(("//", "/*", "*")):
            continue
        cleaned = _STRING_RE.sub('""', line)  # строковые литералы -> ""
        cleaned = re.sub(r"//.*$", "", cleaned)
        for m in IDENT_RE.finditer(cleaned):
            w = m.group(0)
            if w in JS_STOP:
                continue
            if m.start() > 0 and cleaned[m.start() - 1] == ".":
                continue  # obj.prop
            out.append(w)
    return out


def extract_naming_examples(source: str, backend) -> list[dict]:
    lang = getattr(backend, "LANG", "python")
    if lang == "python":
        idents = _identifiers_python(source)
    else:
        try:
            protected = backend.string_body_lines(source)
        except Exception:
            return []
        idents = _identifiers_clike(source, protected)
    if len(idents) < 10:
        return []

    words: dict[str, int] = {}
    for ident in idents:
        for w in split_words(ident):
            words[w] = words.get(w, 0) + 1

    n = len(idents)
    return [{
        "axis": "naming",
        "lang": lang,
        "label": 0,
        "words": words,
        "n_idents": n,
        "mean_len": sum(len(i) for i in idents) / n,
        "single_letter_rate": sum(1 for i in idents if len(i) == 1) / n,
        "camel_rate": sum(1 for i in idents
                          if re.search(r"[a-z][A-Z]", i)) / n,
        "snake_rate": sum(1 for i in idents if "_" in i) / n,
    }]


# ---------- переименование локальных имён ----------

import ast
import hashlib


def build_vocab_from_model(meta_path) -> list[str]:
    """Словарь автора из meta.json naming-стилометра (топ дискриминативных слов)."""
    import json
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return meta["vocab"]


def make_renamer(vocab_words: list[str], seed: str = "", core_size: int = 150):
    """Детерминированный генератор имён в словаре автора.

    seed — это заодно ключ watermark'а: та же seed => та же карта имён.
    Стабилизация: генерация из core_size самых характерных слов автора;
    имя, целиком составленное из них, не трогаем — повторный прогон стабилен.
    """
    words = [w for w in vocab_words if w.isalpha() and 2 <= len(w) <= 12]
    core = words[:core_size] or ["data"]
    core_set = set(core)

    def renamer(old: str, taken: set[str]) -> str:
        old_words = split_words(old)
        if old_words and all(w in core_set for w in old_words):
            return old
        n_words = max(len(old_words), 1)
        h = int(hashlib.md5((seed + old).encode()).hexdigest(), 16)
        for attempt in range(50):
            parts = [core[(h + attempt * 7919 + i * 13) % len(core)]
                     for i in range(n_words)]
            # сохраняем кэйсинг-конвенцию оригинала
            if re.fullmatch(r"[A-Z0-9_]+", old):        # SCREAMING_SNAKE
                new = "_".join(p.upper() for p in parts)
            elif "_" in old:                            # snake_case
                new = "_".join(parts)
            elif old[:1].isupper():                     # PascalCase
                new = "".join(p.capitalize() for p in parts)
            else:                                       # camelCase
                new = parts[0] + "".join(p.capitalize() for p in parts[1:])
            if new != old and new not in taken and len(new) >= 3:
                return new
        return old

    return renamer


def rename_locals_python(source: str, renamer) -> str:
    """Переименование локальных переменных через точные позиции ast.

    Переименовываются только присваиваемые локали функций (не параметры,
    не module/class-уровень, не global/nonlocal, не имена, которые
    перекрываются во вложенных функциях). Строки, комментарии, ключи dict,
    атрибуты и keyword-args не затрагиваются — ast это гарантирует.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    # все идентификаторы файла — для избежания коллизий
    taken = {t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
             if t.type == tokenize.NAME}

    edits: list[tuple[int, int, int, int, str]] = []

    def handle_scope(fn) -> None:
        skip: set[str] = set()
        for sub in ast.walk(fn):
            if isinstance(sub, (ast.Global, ast.Nonlocal)):
                skip.update(sub.names)
        # имена, объявленные во вложенных областях (shadowing)
        nested_locals: set[str] = set()
        for sub in ast.walk(fn):
            if sub is fn:
                continue
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                args = sub.args
                nested_locals.update(
                    a.arg for a in (*args.posonlyargs, *args.args,
                                    *args.kwonlyargs)
                    if isinstance(a, ast.arg))
                for s in ast.walk(sub):
                    if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store):
                        nested_locals.add(s.id)
        skip |= nested_locals

        stores: dict[str, list] = {}
        loads: list = []
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name):
                if isinstance(sub.ctx, ast.Store):
                    stores.setdefault(sub.id, []).append(sub)
                elif isinstance(sub.ctx, ast.Load):
                    loads.append(sub)

        for name, nodes in stores.items():
            if name in skip or name.startswith("_"):
                continue
            new = renamer(name, taken)
            if new == name:
                continue
            taken.add(new)
            for nd in nodes + [l for l in loads if l.id == name]:
                edits.append((nd.lineno, nd.col_offset,
                              nd.end_lineno, nd.end_col_offset, new))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            handle_scope(node)

    for ln, col, end_ln, end_col, new in sorted(edits, reverse=True):
        if ln == end_ln:
            l = lines[ln - 1]
            lines[ln - 1] = l[:col] + new + l[end_col:]
    out = "".join(lines)
    compile(out, "<renamed>", "exec")  # страховка
    return out


_CLIKE_DECL_RE = re.compile(r"^\s*(?:const|let|var)\s+([a-z][A-Za-z0-9$]*)\s*=[^=]")


def rename_locals_clike(source: str, renamer) -> str:
    """Консервативное переименование в TS/JS: только простые const/let-локали.

    Имя пропускается, если оно: экспортируется, импортируется, встречается
    после точки, перед двоеточием, в object-shorthand/деструктуризации,
    в template-литерале, капслоком (константы часто дублируют ключи).
    """
    lines = source.splitlines()
    decls: dict[str, None] = {}
    for line in lines:
        m = _CLIKE_DECL_RE.match(line)
        if m:
            decls.setdefault(m.group(1))

    def is_safe(name: str) -> bool:
        for line in lines:
            for m in re.finditer(rf"\b{re.escape(name)}\b", line):
                i = m.start()
                before = line[:i].rstrip()
                after = line[m.end():]
                if before.endswith("."):
                    return False
                if after.lstrip().startswith(":"):
                    return False
                if re.search(r"\b(export|import)\b.*$", before):
                    return False
                if re.search(r"[{,]\s*$", before) and re.match(r"\s*[,}]", after):
                    return False  # shorthand / деструктуризация
                if "`" in line:
                    return False  # template-литерал — не рискуем
        return True

    taken = set(IDENT_RE.findall(source))
    rename: dict[str, str] = {}
    for name in decls:
        if name.upper() == name or len(name) < 3:
            continue
        if not is_safe(name):
            continue
        new = renamer(name, taken)
        if new != name:
            rename[name] = new
            taken.add(new)

    if not rename:
        return source
    out_lines = []
    for line in lines:
        code = mask_code(line)
        # сначала ВСЕ совпадения, потом применяем справа налево — позиции
        # из маски не протухают от замен других имён на этой строке
        matches = []
        for old, new in rename.items():
            for m in re.finditer(rf"\b{re.escape(old)}\b", code):
                matches.append((m.start(), m.end(), new))
        if not matches:
            out_lines.append(line)
            continue
        chars = list(line)
        for s, e, new in sorted(matches, reverse=True):
            chars[s:e] = list(new)
        out_lines.append("".join(chars))
    return "\n".join(out_lines) + ("\n" if source.endswith("\n") else "")


def rename_locals(source: str, backend, renamer) -> str:
    lang = getattr(backend, "LANG", "python")
    if lang == "python":
        return rename_locals_python(source, renamer)
    return rename_locals_clike(source, renamer)
