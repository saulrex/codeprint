"""stylometer v3 — файл-уровневый классификатор «чей это код».

Отличия от v2 (codeprint/stylometer.py):
- признаки: вектор из 12 мульти-осевых статистик файла (filefeat_axis),
  а не сырые контексты границ — нет утечек, интерпретируемо;
- скоринг файла — одно число, а не размытое среднее по границам.

Обучение: train_stylometer3.py --pos filefeat.jsonl --neg filefeat.jsonl.
Инференс: Stylometer3(dir).score_source(source, backend) -> 0..1.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from pathlib import Path

import numpy as np

from codeprint.filefeat_axis import FEATURE_KEYS, extract_filefeat
from codeprint.harvest import BACKEND_BY_EXT


class FileNet:  # тонкая обёртка, чтобы инференс жил без torch-классов
    pass


class Stylometer3:
    """ONNX-first инференс файл-уровневого стилометра. torch не нужен."""

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self._keys = meta["feature_keys"]
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            str(d / "model.onnx"), providers=["CPUExecutionProvider"])

    def _vector(self, source: str, backend) -> np.ndarray | None:
        exs = extract_filefeat(source, backend)
        if not exs:
            return None
        f = exs[0]["features"]
        return np.array([[f[k] for k in self._keys]], dtype=np.float32)

    def score_source(self, source: str, backend) -> float:
        v = self._vector(source, backend)
        if v is None:
            return float("nan")
        logits = self._session.run(["logits"], {"feats": v})[0]
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[0, 1])


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch import nn

    ap = argparse.ArgumentParser(description="обучение stylometer v3 (файл-уровень)")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    t0 = time.time()

    def load(path: str, label: int):
        rows = []
        with open(path, encoding="utf-8") as f:
            for l in f:
                ex = json.loads(l)
                rows.append(([ex["features"][k] for k in FEATURE_KEYS], label,
                             json.dumps(ex["features"], sort_keys=True)))
        return rows

    data = load(args.pos, 1) + load(args.neg, 0)
    random.Random(42).shuffle(data)
    X = torch.tensor([r[0] for r in data], dtype=torch.float32)
    Y = torch.tensor([r[1] for r in data], dtype=torch.long)
    keys = [zlib.crc32(r[2].encode()) % 5 for r in data]
    val = torch.tensor([k == 0 for k in keys])
    Xt, Yt = X[~val].to(device), Y[~val].to(device)
    Xv, Yv = X[val].to(device), Y[val].to(device)
    print(f"train {len(Xt)}, val {len(Xv)}")

    model = nn.Sequential(
        nn.Linear(len(FEATURE_KEYS), 64), nn.GELU(),
        nn.Linear(64, 64), nn.GELU(),
        nn.Linear(64, 2),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    counts = np.bincount(Yt.cpu().numpy(), minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                     dtype=torch.float32).to(device)
    lossf = nn.CrossEntropyLoss(weight=w)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 2048):
            idx = perm[i:i + 2048]
            loss = lossf(model(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(-1) == Yv).float().mean().item()
            print(f"  ep{ep}: loss={loss.item():.4f} val_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        P = model(Xv).argmax(-1)
    acc = (P == Yv).float().mean().item()
    tp = ((P == 1) & (Yv == 1)).float().sum()
    metrics = {
        "val_accuracy": round(acc, 4),
        "val_precision_author": round(float(tp / (P == 1).sum().clamp(min=1)), 4),
        "val_recall_author": round(float(tp / (Yv == 1).sum().clamp(min=1)), 4),
        "train_examples": len(Xt), "val_examples": len(Xv),
        "feature_keys": FEATURE_KEYS,
        "device": device, "train_seconds": round(time.time() - t0),
    }
    print(json.dumps(metrics, ensure_ascii=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    dummy = torch.zeros(1, len(FEATURE_KEYS))
    torch.onnx.export(model.cpu(), dummy, str(out / "model.onnx"),
                      input_names=["feats"], output_names=["logits"],
                      dynamic_axes={"feats": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
