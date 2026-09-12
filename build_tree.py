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
    """在 UTF-8 字节层面统计 n-gram 转移，返回概率树。

    key   = 前 (order-1) 个字节（bytes）
    value = [(下一个字节, 概率), ...]，按概率降序
    """
    data = text.encode("utf-8")
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    span = max(1, order - 1)
    N = len(arr) - span
    print(f"字节数 {len(arr):,}  上下文长度 {span}  样本数 {N:,}")

    # 上下文 key：把前 span 个字节编成整数
    ctx = np.zeros(N, dtype=np.int64)
    for k in range(span):
        ctx = ctx * 256 + arr[k:k + N]
    nxt = arr[span:]

    # 稀疏计数（不展开成稠密空间）
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
        s, b = ck, bytearray()
        for _ in range(span):
            b.append(s % 256)
            s //= 256
        b.reverse()
        tree[bytes(b)] = [(i, c / total) for i, c in items]

    print(f"树节点 {len(tree):,}  边 {sum(len(v) for v in tree.values()):,}")
    return tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "data/corpus_big.txt"))
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "data/tree.pkl"))
    args = ap.parse_args()

    t0 = time.time()
    text = open(args.corpus, encoding="utf-8").read()
    print(f"语料 {len(text):,} 字符，读入 {time.time()-t0:.1f}s")

    tree = build(text, args.order, args.min_count)
    with open(args.out, "wb") as f:
        pickle.dump({"order": args.order, "tree": tree}, f, protocol=4)
    print(f"已保存 {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
