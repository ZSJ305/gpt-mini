"""在两个阶段建概率树。

用法:
    python3 build_tree.py --order 3 --min-count 3 --out tree.pkl
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build(text, order=3, min_count=3):
    """在「字」层面统计 n-gram 转移，返回概率树。

    key   = 前 (order-1) 个字（str）
    value = [(下一个字, 概率), ...]，按概率降序

    用 translate 把每个字映射成单字节码位，避免 Python 逐字循环。
    """
    chars = sorted(set(text))
    idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    span = max(1, order - 1)
    print(f"字符数 {len(text):,}  不同字 {n}  上下文长度 {span}")

    # 用 translate 得到紧凑的码位序列（每个字 -> 一个 0..n-1 的码位）
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
    print(f"n-gram 组合 {len(uniq):,} 个（>= {min_count} 次）")

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

    print(f"树节点 {len(tree):,}  边 {sum(len(v) for v in tree.values()):,}")
    return tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "data/corpus_big.txt"))
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "data/tree.pkl"))
    ap.add_argument("--answers-only", action="store_true",
                    help="只对「答：」后面的文本建树")
    args = ap.parse_args()

    t0 = time.time()
    text = open(args.corpus, encoding="utf-8").read()
    if args.answers_only:
        text = "\n".join(ln[2:] for ln in text.split("\n") if ln.startswith("答："))
        print("只取答案文本", end="  ")
    print(f"语料 {len(text):,} 字符，读入 {time.time()-t0:.1f}s")

    tree = build(text, args.order, args.min_count)
    with open(args.out, "wb") as f:
        pickle.dump({"order": args.order, "tree": tree}, f, protocol=4)
    print(f"已保存 {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
