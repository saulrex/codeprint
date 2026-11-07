"""composer — единый restyle по именованному профилю автора.

Профиль — JSON в profiles/<имя>.json:

{
  "axes": {
    "imports":    {"scheme": "flat_asis"},          // профильная ось
    "wrapping":   {"model": "models/wrapping-you"}, // нейронная ось
    "blanklines": {"model": "models/blanklines"}      // нейронная ось
  },
  "stylometer": "models/stylometer-you"            // чем меряем сдвиг
}

Порядок применения: imports (двигает строки) -> wrapping (меняет границы
физических строк) -> blanklines (финальная расстановка пустых строк) ->
naming (переименование локалей; seed профиля = ключ watermark'а:
та же seed => та же карта имён, воспроизводимо перед экспертом).
"""
from __future__ import annotations

import json
from pathlib import Path

from codeprint import clike, logiclines
from codeprint.imports_axis import apply_scheme
from codeprint.neural import LoadedNet, NeuralDecider
from codeprint.restyler import restyle
from codeprint.wrapping_axis import apply_wrapping

BACKEND_BY_EXT = {
    **{e: logiclines for e in (".py",)},
    **{e: clike for e in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")},
}


def load_profile(name: str) -> dict:
    path = Path("profiles") / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"профиль не найден: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class Composer:
    """Применяет все оси профиля к исходнику. Модели грузятся лениво, один раз."""

    def __init__(self, profile: dict, root: Path | str = "."):
        self.profile = profile
        self.root = Path(root)
        self._nets: dict[str, LoadedNet] = {}

    def _net(self, model_dir: str) -> LoadedNet:
        if model_dir not in self._nets:
            self._nets[model_dir] = LoadedNet(self.root / model_dir)
        return self._nets[model_dir]

    def apply(self, source: str, ext: str) -> str:
        backend = BACKEND_BY_EXT.get(ext.lower())
        if backend is None:
            return source
        lang = backend.LANG
        axes = self.profile.get("axes", {})

        original = source
        if "imports" in axes:
            source = apply_scheme(source, axes["imports"]["scheme"], lang)

        # naming ДО wrapping: длина имён влияет на решения о переносе строк
        if "naming" in axes:
            from codeprint.naming_axis import (build_vocab_from_model,
                                               make_renamer, rename_locals)
            cfg = axes["naming"]
            vocab = build_vocab_from_model(self.root / cfg["vocab_model"]
                                           / "meta.json")
            renamer = make_renamer(vocab, seed=cfg.get("seed", ""))
            source = rename_locals(source, backend, renamer)

        if "wrapping" in axes:
            net = self._net(axes["wrapping"]["model"])
            source = apply_wrapping(
                source, backend,
                lambda ex: net.predict(*net.encode(ex)))

        if "blanklines" in axes:
            net = self._net(axes["blanklines"]["model"])
            decider = NeuralDecider()
            decider._net = net
            source = restyle(source, decider, backend=backend)

        if "idioms" in axes:
            from codeprint.idioms_axis import apply_idioms
            source = apply_idioms(source, backend,
                                  axes["idioms"]["ternary_max_chars"])

        # синтаксический гейт: битый результат -> откат всего файла
        from codeprint.syntax_gate import gate_allows
        if not gate_allows(original, source, ext):
            print(f"!! syntax-gate откатил рестайл: {ext}")
            return original
        return source

    def score(self, source: str, backend) -> float:
        """P(автор) по stylometer'у профиля; nan если стилометр не задан."""
        stylo_path = self.profile.get("stylometer")
        if not stylo_path:
            return float("nan")
        if not hasattr(self, "_stylo"):
            from codeprint.stylometer import Stylometer
            self._stylo = Stylometer(self.root / stylo_path)
        return self._stylo.score_source(source, backend)
