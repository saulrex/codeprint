"""public_names — repo-wide переименование публичных имён («режим волокиты»).

Публичное имя — контракт, поэтому процесс четырёхшаговый и по умолчанию
read-only (dry-run + отчёт):

1. collect_exports: именованные экспорты TS/JS-файлов репозитория.
2. safety scan: имя отклоняется, если встречается в строковом литерале,
   template-литерале или комментарии ХОТЯ БЫ в одном файле репо (потенциальная
   строковая ссылка: i18n, снапшоты, сериализация) — такие имена идут в отчёт.
3. build map: детерминированно (seed профиля) через словарь автора.
4. apply: замена по маске кода во всех файлах; obj.name (после точки) не трогаем —
   это может быть чужое свойство с совпадающим именем.

Python: не поддержан (нужен анализ импортов; локали — через naming_axis).
"""
from __future__ import annotations

import re
from pathlib import Path

from codeprint.naming_axis import IDENT_RE, _STRING_RE, mask_code

EXPORT_DECL_RE = re.compile(
    r"^\s*export\s+(?:async\s+)?(?:function\*?|class|const|let|var|type|"
    r"interface|enum)\s+([A-Za-z_$][\w$]*)", re.M)
EXPORT_LIST_RE = re.compile(r"^\s*export\s*\{([^}]*)\}", re.M)

SKIP_DIRS = {".git", "node_modules", "dist", "build", "__pycache__"}
EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}


def iter_source_files(repo: Path):
    for p in sorted(repo.rglob("*")):
        if p.suffix.lower() in EXTS and not (set(p.parts) & SKIP_DIRS):
            yield p


def collect_exports(repo: Path) -> dict[str, list[str]]:
    """имя -> [файлы, где объявлено]."""
    exports: dict[str, list[str]] = {}
    for f in iter_source_files(repo):
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in EXPORT_DECL_RE.finditer(src):
            exports.setdefault(m.group(1), []).append(str(f))
        for m in EXPORT_LIST_RE.finditer(src):
            for part in m.group(1).split(","):
                part = part.strip()
                if not part:
                    continue
                name = part.split(" as ")[-1].strip()  # export { a as b } -> b
                if IDENT_RE.fullmatch(name):
                    exports.setdefault(name, []).append(str(f))
    return exports


_STRING_WORD_RE = IDENT_RE


def safety_scan(repo: Path, names: set[str]) -> dict[str, list[str]]:
    """имя -> список причин отклонения (пустой = безопасно).

    Один проход: собираем слова из строковых/комментарных регионов
    (маскированных мест) и пересекаем с именами — без O(строки × имена).
    """
    reasons: dict[str, list[str]] = {n: [] for n in names}
    for n in names:
        if len(n) < 4:
            reasons[n].append("слишком короткое/общее")
    names_lc = {n for n in names if not reasons[n]}
    if not names_lc:
        return reasons
    hits: dict[str, str] = {}  # имя -> первый файл со строковой ссылкой
    for f in iter_source_files(repo):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            masked = mask_code(line)
            if masked == line:
                continue  # нет строк/комментариев
            for m in _STRING_RE.finditer(line):
                for w in _STRING_WORD_RE.findall(m.group(0)):
                    if w in names_lc and w not in hits:
                        hits[w] = f.name
            mm = re.search(r"//(.*)$", line)
            if mm:
                for w in _STRING_WORD_RE.findall(mm.group(1)):
                    if w in names_lc and w not in hits:
                        hits[w] = f.name
    for n, fname in hits.items():
        reasons[n].append(f"строковая ссылка: {fname}")
    return reasons


def build_public_map(names: set[str], renamer, taken: set[str]) -> dict[str, str]:
    out = {}
    for name in sorted(names):
        new = renamer(name, taken)
        if new != name:
            out[name] = new
            taken.add(new)
    return out


def apply_public_map(source: str, name_map: dict[str, str]) -> str:
    """Замена по маске кода; obj.name (после точки) не трогаем.

    Одна объединённая regex на всю карту — без O(строки × имена).
    """
    if not name_map:
        return source
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in
                          sorted(name_map, key=len, reverse=True)) + r")\b")
    out_lines = []
    for line in source.splitlines():
        code = mask_code(line)
        chars = list(line)
        shift = 0  # длины старого/нового имени разные — правим смещение
        for m in pattern.finditer(code):
            if m.start() > 0 and code[m.start() - 1] == ".":
                continue
            new = name_map[m.group(1)]
            s, e = m.start() + shift, m.end() + shift
            chars[s:e] = list(new)
            shift += len(new) - len(m.group(1))
        out_lines.append("".join(chars))
    return "\n".join(out_lines) + ("\n" if source.endswith("\n") else "")


def run(repo: Path, renamer, in_place: bool = False) -> dict:
    """Весь процесс. Возвращает отчёт (и пишет файлы при in_place=True)."""
    exports = collect_exports(repo)
    reasons = safety_scan(repo, set(exports))
    safe = {n for n, r in reasons.items() if not r}
    taken = set()
    for f in iter_source_files(repo):  # коллизии: все идентификаторы репо
        try:
            taken.update(IDENT_RE.findall(
                f.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass
    name_map = build_public_map(safe, renamer, taken)

    changed_files = 0
    for f in iter_source_files(repo):
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new = apply_public_map(src, name_map)
        if new != src:
            changed_files += 1
            if in_place:
                f.write_text(new, encoding="utf-8")

    return {
        "exports_found": len(exports),
        "safe": len(safe),
        "renamed": len(name_map),
        "rejected": {n: r for n, r in reasons.items() if r},
        "changed_files": changed_files,
        "in_place": in_place,
        "map": name_map,
    }
