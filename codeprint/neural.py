"""Шов под нейронку: sequence-labeling вместо генерации.

Модель НЕ переписывает код — она отвечает на узкие вопросы («пустая строка
здесь?», «переносить этот стейтмент?»), а применяют решения
детерминированные движки осей.

LoadedNet — единая загрузка чекпоинтов trainer'а: ONNX (model.onnx +
vocab.json + meta.json, torch не нужен) или torch (model.pt).
NeuralDecider — Decider-обёртка LoadedNet для оси blanklines.
"""
from __future__ import annotations

import json
from pathlib import Path

from .profiles import BoundaryContext, Decider, ParagraphDecider


class LoadedNet:
    """Загруженная бинарная/мультиклассовая решалка. predict(ids, nums) -> int."""

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        onnx_path = d / "model.onnx"
        ckpt_path = d / "model.pt"
        meta_path = d / "meta.json"

        self._n_num = 8
        if meta_path.exists():
            self._n_num = json.loads(
                meta_path.read_text(encoding="utf-8")).get("n_num", 8)

        self._session = None
        self._model = None
        self._torch = None

        if onnx_path.exists():
            try:
                import numpy as np
                import onnxruntime as ort

                self._np = np
                self._session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"])
                self._vocab = json.loads(
                    (d / "vocab.json").read_text(encoding="utf-8"))
                return
            except ImportError:
                pass

        import numpy as np
        import torch

        from codeprint.trainer import BlanklineNet

        self._np = np
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self._n_num = cfg.get("n_num", 8)
        model = BlanklineNet(cfg["vocab_size"], dim=cfg["dim"],
                             n_classes=cfg.get("n_classes", 2),
                             n_num=self._n_num)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model
        self._torch = torch
        self._vocab = ckpt["vocab"]

    def encode(self, ex: dict):
        from codeprint.features import encode_example

        ids, nums, _ = encode_example(ex, self._vocab)
        return ids, nums[: self._n_num]

    def predict(self, ids: list[int], nums: list[float]) -> int:
        np = self._np
        if self._session is not None:
            logits = self._session.run(
                ["logits"],
                {"ids": np.array([ids], dtype=np.int64),
                 "nums": np.array([nums], dtype=np.float32)},
            )[0]
            return int(logits.argmax(-1)[0])
        t = self._torch
        with t.no_grad():
            logits = self._model(
                t.tensor([ids], dtype=t.long),
                t.tensor([nums], dtype=t.float32),
            )
        return int(logits.argmax(-1).item())


class NeuralDecider(Decider):
    """Drop-in замена эвристических профилей: та же сигнатура blanks_between."""

    def __init__(self, checkpoint: str | None = None, fallback: Decider | None = None):
        self.checkpoint = checkpoint
        self.fallback = fallback or ParagraphDecider()
        self._net: LoadedNet | None = None
        if checkpoint is not None:
            ckpt = Path(checkpoint)
            self._net = LoadedNet(ckpt.parent if ckpt.name == "model.pt" else ckpt)

    def blanks_between(self, ctx: BoundaryContext) -> int:
        if self._net is None:
            return self.fallback.blanks_between(ctx)
        from codeprint.features import ex_from_context

        ids, nums = self._net.encode(ex_from_context(ctx))
        return self._net.predict(ids, nums)  # 0 или 1; двойки модель не ставит
