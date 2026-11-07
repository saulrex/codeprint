"""Профили расстановки пустых строк — эвристические заменители нейронки.

Каждый профиль реализует интерфейс Decider: на границе двух логических
строк он отвечает на единственный вопрос — «сколько пустых строк здесь
поставить: 0, 1 или 2». Это ровно тот контракт, который позже займёт
NeuralDecider (см. neural.py), поэтому замена эвристики на модель 0.4–0.8B
не потребует менять ничего, кроме фабрики профилей.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundaryContext:
    """Всё, что решающий модуль знает о границе между логическими строками."""

    prev_lines: list[str]  # до 8 предыдущих непустых физических строк
    next_line: str         # следующая непустая физическая строка
    had_blank: bool        # была ли здесь пустая строка в исходнике


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _kw(line: str) -> str:
    """Первое 'слово' строки — грубый класс конструкции."""
    return line.strip().split(" ", 1)[0].rstrip(":").split("(", 1)[0]


_DEFCLASS = {"def", "class", "async"}
_CLAUSERS = {"else", "elif", "except", "finally"}  # нельзя отрывать от тела блока
_NON_ASSIGN = {
    "return", "yield", "raise", "for", "while", "if", "elif", "else",
    "with", "try", "except", "finally", "import", "from", "def", "class",
    "async", "pass", "break", "continue", "del", "assert", "print",
    "global", "nonlocal", "lambda", "case", "match",
}


def _is_assign(stripped: str) -> bool:
    """Похоже ли, что строка — присваивание (а не сравнение/вызов/ключевое слово)."""
    if not stripped or stripped.endswith(":") or stripped.startswith("#"):
        return False
    if _kw(stripped) in _NON_ASSIGN:
        return False
    return "=" in stripped and "==" not in stripped


class Decider:
    """Интерфейс решающего модуля. Возвращает 0..2 пустых строк."""

    def blanks_between(self, ctx: BoundaryContext) -> int:
        raise NotImplementedError


class KeepDecider(Decider):
    """Сохраняет расстановку автора, лишь схлопывая длинные прогоны до одной строки."""

    def blanks_between(self, ctx: BoundaryContext) -> int:
        return 1 if ctx.had_blank else 0


class CompactDecider(Decider):
    """Стерильно-плотный стиль: пустые строки только между top-level конструкциями."""

    def blanks_between(self, ctx: BoundaryContext) -> int:
        if _indent(ctx.next_line) > 0:
            return 0
        if _kw(ctx.next_line) in _DEFCLASS or ctx.next_line.strip().startswith("@"):
            return 2
        return 1


class ParagraphDecider(Decider):
    """'Абзацный' стиль: внутри блоков логические фазы отбиваются пустой строкой.

    Правила (по убыванию приоритета):
    - декораторы: 2 на верхнем уровне, 1 внутри класса;
    - top-level def/class: 2, прочий top-level: 1;
    - служебные клаузы (else/except/...): никогда не отрывать от тела;
    - методы/вложенные def: 1;
    - комментарий-секция: 1 перед ним;
    - дедент (блок закончился): 1;
    - 'смена фазы': return/циклы/if/with/try после обычного кода: 1;
    - конец серии присваиваний (дальше вызов/конструкция): 1;
    - иначе: 0.
    """

    _PHASE_BREAKERS = {"return", "yield", "raise", "for", "while", "if", "with", "try"}

    def blanks_between(self, ctx: BoundaryContext) -> int:
        nxt = ctx.next_line
        ns = nxt.strip()
        ni = _indent(nxt)
        prev_s = ctx.prev_lines[-1].strip() if ctx.prev_lines else ""
        pi = _indent(ctx.prev_lines[-1]) if ctx.prev_lines else 0
        kw = _kw(nxt)

        if ns.startswith("@"):
            return 2 if ni == 0 else 1
        if ni == 0:
            return 2 if kw in _DEFCLASS else 1
        if kw in _CLAUSERS:
            return 0
        if kw in _DEFCLASS:
            return 1
        if ns.startswith("#"):
            return 1
        if ni < pi:
            return 1
        if kw in self._PHASE_BREAKERS and not prev_s.endswith(":"):
            return 1
        if ni == pi and _is_assign(prev_s) and not _is_assign(ns):
            return 1
        return 0


#: Реестр профилей для CLI. NeuralDecider подключится сюда же.
PROFILES: dict[str, type[Decider]] = {
    "keep": KeepDecider,
    "compact": CompactDecider,
    "paragraph": ParagraphDecider,
}
