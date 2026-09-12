"""建多阶概率树（backoff）。

用法:
    python3 build_tree.py --orders 5,4,3 --min-count 2 --answers-only
生成 {"orders": [...], "trees": {5: {...}, 4: {...}, 3: {...}}}

每阶是一张独立的 n-gram 转移表：
    key   = 前 (order-1) 个字
    value = [(下一个字, 概率), ...] 按概率降序

生成时从最高阶开始查，查不到就降一阶 —— 长上下文保连贯，
短上下文保覆盖。
"""
import argparse
import os
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_one(text, chars, idx, order, min_count=2, span_reuse=None):
    """为单个阶数建树。chars/idx 由外部复用，避免重复计算。"""
    n = len(chars)
    span = max(1, order - 1)

    table = {ord(c): i for i, c in enumerate(chars)}
    ids = np.array([ord(ch) for ch in text.translate(table)], dtype=np.int64)
    N = len(ids) - span

    ctx = np.zeros(N, dtype=np.int64)
    for k in range(span):
        ctx = ctx * n + ids[k:k + N]
    nxt = ids[span:]

    pair = np.stack([ctx, nxt], axis=1)
    uniq, counts = np.unique(pair, axis=0, return_counts=True)
    keep = counts >= min_count
    uniq, counts = uniq[keep], counts[keep]

    from collections import defaultdict
    tmp = defaultdict(list)
    for (ck, nx), c in zip(uniq.tolist(), counts.tolist()):
        tmp[ck].append((nx, c))

    tree = {}
    for ck, items in tmp.items():
        total = sum(c for _, c in items)
        items.sort(key=lambda x: -x[1])
        s, ctx_chars = ck, []
        for _ in range(span):
            ctx_chars.append(chars[s % n])
            s //= n
        tree["".join(reversed(ctx_chars))] = [(chars[i], c / total) for i, c in items]

    return tree


def build_all(text, orders, min_count=2):
    chars = sorted(set(text))
    idx = {c: i for i, c in enumerate(chars)}
    print(f"字符数 {len(text):,}  不同字 {len(chars)}  阶数 {orders}")
    trees = {}
    for o in orders:
        t0 = time.time()
        trees[o] = build_one(text, chars, idx, o, min_count)
        print(f"  order={o}: 节点 {len(trees[o]):,}  边 "
              f"{sum(len(v) for v in trees[o].values()):,}  {time.time()-t0:.1f}s")
    return {"orders": list(orders), "trees": trees}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "data/corpus_big.txt"))
    ap.add_argument("--orders", default="5,4,3")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--answers-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "data/tree.pkl"))
    args = ap.parse_args()

    orders = [int(x) for x in args.orders.split(",") if x.strip()]
    t0 = time.time()
    text = open(args.corpus, encoding="utf-8").read()
    if args.answers_only:
        text = "\n".join(ln[2:] for ln in text.split("\n") if ln.startswith("答："))
        print("只取答案文本", end="  ")
    print(f"语料 {len(text):,} 字符，读入 {time.time()-t0:.1f}s")

    blob = build_all(text, orders, args.min_count)
    with open(args.out, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    print(f"已保存 {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
