"""stylometer — классификатор «чей это код» (модуль 10, см. docs/ARCHITECTURE.md).

Бинарная модель на тех же границах логических строк, что и blanklines-решалка:
«эта расстановка пустых строк — целевого автора или чужая?»

Двойная роль:
- ТАРГЕТ: после рестайла P(автор) должна расти;
- ЩИТ:   атрибуция исходного автора должна ломаться.

Обучение: train_stylometer.py --pos <jsonl автора> --neg <jsonl чужих>.
Инференс: Stylometer(model_dir).score_source(source, backend) -> 0..1,
torch не нужен, если рядом есть model.onnx (достаточно onnxruntime).

Ограничение v1: валидация на уровне границ (boundary-level), сплит по хэшу
контекста; соседние границы одного файла могут попасть в разные части.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from codeprint.features import MAX_LEN, N_NUM, encode_example, tokenize
from codeprint.harvest import extract_blankline_examples

VOCAB_SIZE = 30000


def _load_jsonl(path: str, label: int, cap: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        exs = [json.loads(l) for l in f]
    if len(exs) > cap:
        exs = random.Random(42).sample(exs, cap)
    for ex in exs:
        ex["author"] = label
    return exs


class Stylometer:
    """Инференс: ONNX если есть, иначе torch, иначе ImportError.

    v2: десятый признак — наблюдаемое решение («пустая строка тут есть»).
    Без него стилометр видел только контекст и был слеп к рестайлу.
    """

    N_FEATURES = N_NUM + 1

    def __init__(self, model_dir: str):
        d = Path(model_dir)
        self._vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self._np = np
        meta_path = d / "meta.json"
        self._n_num = (json.loads(meta_path.read_text(encoding="utf-8"))
                       .get("n_num", 8)) if meta_path.exists() else 8
        self._session = None
        self._model = None
        if (d / "model.onnx").exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(d / "model.onnx"), providers=["CPUExecutionProvider"])
                return
            except ImportError:
                pass
        import torch as _torch

        from codeprint.trainer import BlanklineNet  # torch-зависимый, лениво

        ckpt = _torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        model = BlanklineNet(ckpt["config"]["vocab_size"], dim=ckpt["config"]["dim"],
                             n_num=ckpt["config"].get("n_num", 8))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model, self._torch = model, _torch

    def score_source(self, source: str, backend) -> float:
        """P(«написал целевой автор») для файла: среднее по границам."""
        exs = [e for e in extract_blankline_examples(source, backend) if e["prev_tail"]]
        if not exs:
            return float("nan")
        ids = np.zeros((len(exs), MAX_LEN), dtype=np.int64)
        nums = np.zeros((len(exs), self.N_FEATURES), dtype=np.float32)
        for i, ex in enumerate(exs):
            ids[i], nums[i, :N_NUM], _ = encode_example(ex, self._vocab)
            nums[i, N_NUM] = float(min(ex["label"], 1))  # наблюдаемое решение
        nums = nums[:, :self._n_num]  # старые чекпоинты могли иметь меньше признаков
        if self._session is not None:
            # NB: экспортированный граф TransformerEncoder'а корёжит batch>1,
            # поэтому гоняем по одному примеру — для файла это миллисекунды
            probs = []
            for i in range(len(exs)):
                logits = self._session.run(
                    ["logits"],
                    {"ids": ids[i:i + 1], "nums": nums[i:i + 1]})[0]
                e = np.exp(logits - logits.max())
                probs.append(float((e / e.sum())[0, 1]))
            return float(np.mean(probs))
        with self._torch.no_grad():
            logits = self._model(self._torch.from_numpy(ids),
                                 self._torch.from_numpy(nums)).numpy()
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        p_author = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
        return float(p_author.mean())


def main(argv: list[str] | None = None) -> int:
    import torch  # обучение — только там, где torch есть
    from torch import nn

    from codeprint.trainer import BlanklineNet, export_onnx

    ap = argparse.ArgumentParser(description="обучение stylometer'а")
    ap.add_argument("--pos", required=True, help="jsonl целевого автора (label 1)")
    ap.add_argument("--neg", required=True, help="jsonl чужих (label 0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--max-per-class", type=int, default=300_000)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    t0 = time.time()

    print("[1/4] читаю и балансирую")
    examples = _load_jsonl(args.pos, 1, args.max_per_class) + \
               _load_jsonl(args.neg, 0, args.max_per_class)
    random.Random(42).shuffle(examples)
    n_pos = sum(e["author"] for e in examples)
    print(f"  всего {len(examples)} (автор: {n_pos}, чужие: {len(examples) - n_pos})")

    print("[2/4] словарь и кодирование")
    cnt: Counter = Counter()
    for ex in examples:
        for l in ex["prev_tail"]:
            cnt.update(tokenize(l))
        cnt.update(tokenize(ex["next_line"]))
    vocab = {t: i + 2 for i, (t, _) in enumerate(cnt.most_common(VOCAB_SIZE))}

    n = len(examples)
    X = np.zeros((n, MAX_LEN), dtype=np.int32)
    F = np.zeros((n, Stylometer.N_FEATURES), dtype=np.float32)
    Y = np.zeros(n, dtype=np.int64)
    keys = np.zeros(n, dtype=np.int64)
    for i, ex in enumerate(examples):
        ids, nums, _ = encode_example(ex, vocab)
        X[i] = ids
        # 10-й признак — наблюдаемое решение автора (пустая строка или нет):
        # именно оно делает стилометр чувствительным к рестайлу
        F[i] = nums + [float(min(ex["label"], 1))]
        Y[i] = ex["author"]
        keys[i] = zlib.crc32(
            (ex["next_line"] + "|".join(ex["prev_tail"])).encode()) % 20
    val_mask = keys == 0
    print(f"  train {int((~val_mask).sum())}, val {int(val_mask.sum())}, "
          f"за {time.time() - t0:.0f}с")

    print("[3/4] учу")
    model = BlanklineNet(len(vocab) + 2, n_num=Stylometer.N_FEATURES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  параметров: {n_params / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    counts = np.bincount(Y[~val_mask], minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1))).to(device)
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

    print("[4/4] оценка")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, Xv.shape[0], 4096):
            preds.append(model(Xv[i:i + 4096], Fv[i:i + 4096]).argmax(-1))
    P = torch.cat(preds)
    acc = (P == Yv).float().mean().item()
    tp = ((P == 1) & (Yv == 1)).float().sum()
    precision = float(tp / (P == 1).sum().clamp(min=1))
    recall = float(tp / (Yv == 1).sum().clamp(min=1))
    metrics = {
        "val_accuracy": round(acc, 4),
        "val_precision_author": round(precision, 4),
        "val_recall_author": round(recall, 4),
        "train_examples": int(nt),
        "val_examples": int(Xv.shape[0]),
        "params_m": round(n_params / 1e6, 2),
        "n_num": Stylometer.N_FEATURES,
        "device": device,
        "train_seconds": round(time.time() - t0),
    }
    print("  " + json.dumps(metrics, ensure_ascii=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"vocab_size": len(vocab) + 2, "dim": 128,
                   "n_num": Stylometer.N_FEATURES},
        "metrics": metrics,
    }, out / "model.pt")
    (out / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    export_onnx(model, out)
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
