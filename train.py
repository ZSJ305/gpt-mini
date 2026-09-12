"""第 3 关：训练。自回归语言建模——最小化预测下一个 token 的损失。

用法:
    python3 train.py --steps 3000
训练完自动生成几个样例对话，并把权重存到 model.pkl。
"""
import argparse
import os
import time

import numpy as np

from model import Adam, MiniGPT, generate, save
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = [
    "问：你好",
    "问：你是谁",
    "问：今天天气怎么样",
    "问：什么是大模型",
    "问：我心情不好",
    "问：二十加十五等于几",
]


def build_data(tok, block_size, val_ratio=0.05):
    """把语料切成定长序列。每条对话以换行结尾，天然构成一个样本边界。"""
    text = open(os.path.join(HERE, "data/corpus.txt"), encoding="utf-8").read()
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids)
    split = int(n * (1 - val_ratio))

    def make(arr):
        m = len(arr) // block_size
        arr = arr[: m * block_size].reshape(m, block_size)
        return arr[:, :-1], arr[:, 1:]      # 输入 / 目标（错开一位）

    return make(ids[:split]), make(ids[split:])


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=os.path.join(HERE, "model.pkl"))
    args = ap.parse_args()

    np.random.seed(args.seed)

    text = open(os.path.join(HERE, "data/corpus.txt"), encoding="utf-8").read()
    tok = Tokenizer.train(text)
    tok.save(os.path.join(HERE, "data/tokenizer.json"))
    print(f"vocab={tok.vocab_size}  语料={len(text)/1024:.0f} KB")

    (xtr, ytr), (xva, yva) = build_data(tok, args.block)
    print(f"训练序列={len(xtr)}  验证序列={len(xva)}")

    model = MiniGPT(tok.vocab_size, block_size=args.block, n_layer=args.layers,
                    n_head=args.heads, dim=args.dim)
    opt = Adam(model.all_params(), lr=args.lr)

    best_val = float("inf")
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            idxs = np.random.randint(0, len(xva), size=min(args.batch, len(xva)))
            _, vloss = model.forward(xva[idxs], yva[idxs])[:2]
            print(f"[{step:5d}] train={(train_loss if step else float('nan')):.3f} "
                  f"val={vloss:.3f} lr={opt.lr:.2e} "
                  f"{(time.time()-t0)/max(1e-9, time.time()-t0):.0f}s")
            if step and vloss < best_val:
                best_val = vloss
                save(model, tok, args.out)

        if step == args.steps:
            break

        opt.lr = lr_at(step, args.steps, args.lr, args.warmup)
        idxs = np.random.randint(0, len(xtr), size=args.batch)
        _, train_loss, dlogits = model.forward(xtr[idxs], ytr[idxs])
        model.backward(dlogits)
        opt.step()

        if step % 50 == 0:
            print(f"  step {step:5d}  loss={train_loss:.4f}  "
                  f"{(time.time()-t0):.0f}s")

    save(model, tok, args.out)

    print("\n===== 生成样例（训练后）=====")
    for p in PROMPTS:
        print(generate(model, tok, p, max_new=60, temperature=0.7, top_k=10))
        print("-" * 30)


if __name__ == "__main__":
    main()
