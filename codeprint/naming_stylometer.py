"""naming-стилометр: авторство по словарю идентификаторов.

Prettier не трогает имена — поэтому если атрибуция в отформатированном
репозитории вообще возможна по коду, то здесь. Классический подход
стилометрии: частоты словаря автора.

Признаки файла: топ-N дискриминативных слов (нормированные частоты) +
скаляры (длина имён, доля однобуквенных, camel/snake).

Обучение: train_naming.py --pos naming.jsonl --neg naming.jsonl --out DIR
Инференс: NamingStylometer(dir).score_source(source, backend) -> 0..1
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from codeprint.naming_axis import extract_naming_examples

TOP_WORDS = 1000
MIN_WORD_FILES = 30  # слово должно встретиться хотя бы в стольких файлах
SCALARS = ["mean_len", "single_letter_rate", "camel_rate", "snake_rate"]


def _load(path: str, label: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        r["author"] = label
    return rows


def select_words(pos: list[dict], neg: list[dict]) -> list[str]:
    """Топ слов по разнице нормированных частот между авторами."""
    def freqs(rows):
        cnt: Counter = Counter()
        files_with: Counter = Counter()
        total = 0
        for r in rows:
            for w, c in r["words"].items():
                cnt[w] += c
                files_with[w] += 1
                total += c
        return cnt, files_with, max(total, 1)

    cp, fp, tp = freqs(pos)
    cn, fn, tn = freqs(neg)
    words = set(cp) | set(cn)
    scored = []
    for w in words:
        if fp[w] < MIN_WORD_FILES and fn[w] < MIN_WORD_FILES:
            continue
        diff = abs(cp[w] / tp - cn[w] / tn)
        scored.append((diff, w))
    scored.sort(reverse=True)
    return [w for _, w in sorted(scored, key=lambda x: -x[0])[:TOP_WORDS]]


def vectorize(row: dict, vocab: list[str]) -> list[float]:
    total = max(sum(row["words"].values()), 1)
    vec = [row["words"].get(w, 0) / total for w in vocab]
    vec.append(row["mean_len"] / 20.0)
    vec.append(row["single_letter_rate"])
    vec.append(row["camel_rate"])
    vec.append(row["snake_rate"])
    vec.append(min(row["n_idents"], 500) / 500.0)
    return vec


class NamingStylometer:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self._vocab = meta["vocab"]
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            str(d / "model.onnx"), providers=["CPUExecutionProvider"])

    def score_source(self, source: str, backend) -> float:
        exs = extract_naming_examples(source, backend)
        if not exs:
            return float("nan")
        v = np.array([vectorize(exs[0], self._vocab)], dtype=np.float32)
        logits = self._session.run(["logits"], {"feats": v})[0]
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[0, 1])


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch import nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    pos = _load(args.pos, 1)
    neg = _load(args.neg, 0)
    print(f"файлов: автор {len(pos)}, чужие {len(neg)}")

    vocab = select_words(pos, neg)
    print(f"словарь: {len(vocab)} дискриминативных слов за {time.time()-t0:.0f}с")

    data = pos + neg
    random.Random(42).shuffle(data)
    X = np.array([vectorize(r, vocab) for r in data], dtype=np.float32)
    Y = np.array([r["author"] for r in data], dtype=np.int64)
    keys = np.array([zlib.crc32(json.dumps(r["words"], sort_keys=True).encode()) % 5
                     for r in data])
    val = keys == 0
    print(f"train {int((~val).sum())}, val {int(val.sum())}, признаков {X.shape[1]}")

    Xt = torch.from_numpy(X[~val]).to(device)
    Yt = torch.from_numpy(Y[~val]).to(device)
    Xv = torch.from_numpy(X[val]).to(device)
    Yv = torch.from_numpy(Y[val]).to(device)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 256), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, 64), nn.GELU(),
        nn.Linear(64, 2),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(Y[~val], minlength=2).astype(np.float32)
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
        "vocab": vocab,
        "train_examples": int(len(Xt)), "val_examples": int(len(Xv)),
        "device": device, "train_seconds": round(time.time() - t0),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "vocab"}))

    # любимые слова каждой стороны — для глаз
    def top_words(rows, n=15):
        cnt: Counter = Counter()
        for r in rows:
            cnt.update(r["words"])
        total = max(sum(cnt.values()), 1)
        return [(w, round(c / total * 1000, 1)) for w, c in cnt.most_common(n)]

    print("  топ-слова автора: ", top_words(pos))
    print("  топ-слова чужих:  ", top_words(neg))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    dummy = torch.zeros(1, X.shape[1])
    torch.onnx.export(model.cpu(), dummy, str(out / "model.onnx"),
                      input_names=["feats"], output_names=["logits"],
                      dynamic_axes={"feats": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
