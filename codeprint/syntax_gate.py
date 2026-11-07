"""syntax_gate — проверка синтаксиса результата рестайла.

Python — через compile(). TS/JS/JSX — через tree-sitter (has_error).
Если tree-sitter не установлен, TS-файлы пропускаются с предупреждением
(один раз на процесс), Python продолжает проверяться всегда.

Урок аудита: токен-инвариант НЕ ловит синтаксические мины
(`if (a,)` — валидные токены, битый синтаксис).
"""
from __future__ import annotations

_parsers: dict = {}
_warned = False


def _ts_parser(kind: str):
    """kind: typescript | tsx. Ленивая инициализация."""
    if kind not in _parsers:
        from tree_sitter import Language, Parser
        import tree_sitter_typescript as tst

        lang = Language(tst.language_tsx() if kind == "tsx"
                        else tst.language_typescript())
        _parsers[kind] = Parser(lang)
    return _parsers[kind]


def gate_allows(original: str, new: str, ext: str) -> bool:
    """Трёхпозиционный гейт:
    - результат парсится -> True;
    - результат бит, но и ОРИГИНАЛ не парсился -> гейт слеп, пропускаем
      (держава грамматики: styled.div<T>`...` и т.п.);
    - результат бит, оригинал парсился -> False, откат.
    """
    if syntax_ok(new, ext):
        return True
    return not syntax_ok(original, ext)


def syntax_ok(source: str, ext: str) -> bool:
    """True — код парсится. Неподдержанные расширения — True (не наш гейт)."""
    ext = ext.lower()
    if ext == ".py":
        try:
            compile(source, "<gate>", "exec")
            return True
        except SyntaxError:
            return False
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        global _warned
        try:
            kind = "tsx" if ext in (".tsx", ".jsx") else "typescript"
            tree = _ts_parser(kind).parse(source.encode("utf-8"))
            return not tree.root_node.has_error
        except ImportError:
            if not _warned:
                _warned = True
                print("!! tree-sitter не установлен — TS без syntax-gate "
                      "(pip install tree-sitter tree-sitter-typescript)")
            return True
    return True
