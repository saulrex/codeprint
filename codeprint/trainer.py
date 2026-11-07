"""trainer — обучение мелкой нейронки-решалки на датасете harvest.

Модель ~4M параметров: Embedding -> 2-слойный TransformerEncoder ->
masked mean-pool -> + структурные признаки -> MLP -> 2 класса
(0 = без пустой строки, 1 = ставить пустую).

Запуск (на GB10 в контейнере sglang, где есть torch+CUDA):
  python3 train.py --data datasets/front/2026-07-03/blanklines.jsonl \
                   --out models/blanklines
Результат: <out>/model.pt (state_dict + словарь + конфиг) и meta.json (метрики).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from codeprint.features import MAX_LEN, N_NUM, encode_example, tokenize

VOCAB_SIZE = 30000


class BlanklineNet(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, heads: int = 4,
                 layers: int = 2, n_classes: int = 2, n_num: int = 8):
        super().__init__()
        self.n_num = n_num
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(
            nn.Linear(dim + n_num, 128), nn.GELU(), nn.Linear(128, n_classes))

    def forward(self, ids: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        mask = ids != 0
        x = self.emb(ids) + self.pos.weight.unsqueeze(0)
        x = self.enc(x, src_key_padding_mask=~mask)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([x, nums], dim=1))


def export_onnx(model: BlanklineNet, out: Path) -> None:
    """Экспорт в ONNX: инференс без torch (достаточно маленького onnxruntime)."""
    try:
        model_cpu = model.cpu().eval()
        dummy_ids = torch.zeros(1, MAX_LEN, dtype=torch.long)
        dummy_nums = torch.zeros(1, getattr(model_cpu, "n_num", N_NUM))
        torch.onnx.export(
            model_cpu, (dummy_ids, dummy_nums), str(out / "model.onnx"),
            input_names=["ids", "nums"], output_names=["logits"],
            dynamic_axes={"ids": {0: "batch"}, "nums": {0: "batch"},
                          "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False)  # legacy-экспортёр: dynamo-путь ломает динамический батч
        print(f"ONNX-экспорт: {out / 'model.onnx'}")
    except Exception as e:
        print(f"ONNX-экспорт пропущен: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="blanklines.jsonl от harvest")
    ap.add_argument("--out", default="models/blanklines")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-examples", type=int, default=None,
                    help="подвыборка для CPU/быстрой проверки")
    ap.add_argument("--binary", action="store_true",
                    help="слить метки в 0/1 (для blanklines: 2 = «2+», их слишком мало)")
    ap.add_argument("--export-only", action="store_true",
                    help="не учить, только ONNX-экспорт существующего model.pt")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.export_only:
        ckpt = torch.load(Path(args.out) / "model.pt", map_location="cpu",
                          weights_only=False)
        cfg = ckpt["config"]
        model = BlanklineNet(cfg["vocab_size"], dim=cfg["dim"],
                             n_classes=cfg.get("n_classes", 2),
                             n_num=cfg.get("n_num", 8))
        model.load_state_dict(ckpt["state_dict"])
        (Path(args.out) / "vocab.json").write_text(
            json.dumps(ckpt["vocab"], ensure_ascii=False), encoding="utf-8")
        export_onnx(model, Path(args.out))
        return 0

    if not args.data:
        ap.error("без --export-only нужен --data")

    t0 = time.time()
    print(f"[1/5] читаю {args.data}")
    with open(args.data, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f]
    if args.max_examples and len(examples) > args.max_examples:
        import random
        examples = random.Random(42).sample(examples, args.max_examples)
        print(f"  подвыборка: {args.max_examples}")
    if not examples:
        print("датасет пуст — нечего учить. Проверь --repo/--author/--before у harvest.",
              file=sys.stderr)
        return 1
    print(f"  {len(examples)} примеров за {time.time() - t0:.0f}с")

    print("[2/5] строю словарь")
    cnt: Counter = Counter()
    for ex in examples:
        for l in ex["prev_tail"]:
            cnt.update(tokenize(l))
        cnt.update(tokenize(ex["next_line"]))
    vocab = {t: i + 2 for i, (t, _) in enumerate(cnt.most_common(VOCAB_SIZE))}
    print(f"  словарь {len(vocab)} (+PAD/UNK); уникальных токенов {len(cnt)}")

    print("[3/5] кодирую примеры")
    n = len(examples)
    X = np.zeros((n, MAX_LEN), dtype=np.int32)
    F = np.zeros((n, N_NUM), dtype=np.float32)
    Y = np.zeros(n, dtype=np.int64)
    for i, ex in enumerate(examples):
        ids, nums, y = encode_example(ex, vocab)
        X[i] = ids
        F[i] = nums
        Y[i] = y
    if args.binary:
        Y = np.minimum(Y, 1)
    n_classes = int(Y.max()) + 1
    print(f"  классов: {n_classes}, распределение: "
          f"{np.bincount(Y).tolist()}")
    # сплит по crc32 контента: одинаковые примеры не попадают в обе части
    keys = np.array([
        zlib.crc32((ex["next_line"] + "|".join(ex["prev_tail"])).encode()) % 20
        for ex in examples
    ])
    val_mask = keys == 0
    print(f"  train {int((~val_mask).sum())}, val {int(val_mask.sum())}, "
          f"за {time.time() - t0:.0f}с")

    print("[4/5] учу")
    model = BlanklineNet(len(vocab) + 2, dim=args.dim, n_classes=n_classes,
                         n_num=N_NUM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  параметров: {n_params / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # взвешиваем классы: на перекошенных датасетах модель иначе
    # вырождается в «всегда мажоритарный класс»
    counts = np.bincount(Y[~val_mask], minlength=n_classes).astype(np.float32)
    weights = torch.tensor(
        counts.sum() / (n_classes * np.maximum(counts, 1))).to(device)
    print(f"  веса классов: {[round(w, 3) for w in weights.tolist()]}")
    lossf = nn.CrossEntropyLoss(weight=weights)

    Xt = torch.from_numpy(X[~val_mask])
    Ft = torch.from_numpy(F[~val_mask])
    Yt = torch.from_numpy(Y[~val_mask])
    Xv = torch.from_numpy(X[val_mask]).long().to(device)
    Fv = torch.from_numpy(F[val_mask]).to(device)
    Yv = torch.from_numpy(Y[val_mask]).to(device)

    nt = Xt.shape[0]
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(nt)
        t_ep = time.time()
        for step, i in enumerate(range(0, nt, args.bs)):
            idx = perm[i:i + args.bs]
            logits = model(Xt[idx].long().to(device), Ft[idx].to(device))
            loss = lossf(logits, Yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 200 == 0:
                acc = (logits.argmax(-1) == Yt[idx].to(device)).float().mean()
                print(f"  ep{ep} шаг{step} loss={loss.item():.4f} acc={acc.item():.3f}")
        print(f"  эпоха {ep}: {time.time() - t_ep:.0f}с")

    print("[5/5] оценка на val")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, Xv.shape[0], 4096):
            preds.append(model(Xv[i:i + 4096], Fv[i:i + 4096]).argmax(-1))
    P = torch.cat(preds)
    acc = (P == Yv).float().mean().item()
    majority = max(float((Yv == c).float().mean()) for c in range(n_classes))
    recall = {}
    for c in range(n_classes):
        m = Yv == c
        recall[c] = float(((P == c) & m).float().sum() / m.sum().clamp(min=1))
    metrics = {
        "val_accuracy": round(acc, 4),
        "val_majority_baseline": round(majority, 4),
        "val_recall": {str(k): round(v, 4) for k, v in recall.items()},
        "train_examples": int(nt),
        "val_examples": int(Xv.shape[0]),
        "params_m": round(n_params / 1e6, 2),
        "n_classes": n_classes,
        "n_num": N_NUM,
        "device": device,
        "train_seconds": round(time.time() - t0),
    }
    print("  " + json.dumps(metrics, ensure_ascii=False))

    # примеры предсказаний для глаз
    val_examples = [ex for ex, v in zip(examples, val_mask) if v]
    val_ys = Y[val_mask]
    for k, ex in enumerate(val_examples[:5]):
        ids, nums, _ = encode_example(ex, vocab)
        with torch.no_grad():
            p = int(model(torch.tensor([ids]).long().to(device),
                          torch.tensor([nums]).to(device)).argmax(-1))
        truth = int(val_ys[k])
        mark = "OK " if p == truth else "ПРОМАХ"
        print(f"  {mark} правда={truth} модель={p} | "
              f"{ex['next_line'].strip()[:60]!r}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {"vocab_size": len(vocab) + 2, "dim": args.dim,
                   "n_classes": n_classes, "n_num": N_NUM},
        "metrics": metrics,
    }, out / "model.pt")
    (out / "meta.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"сохранено: {out / 'model.pt'}")
    export_onnx(model, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
