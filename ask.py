#!/usr/bin/env python3
"""ask —— 从概率树生成回答。

    ask 你好
    ask -v 什么是大模型
    ask --topk 1 今天天气怎么样
    ask              # 交互模式
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

_TREE = None


def get_tree():
    global _TREE
    if _TREE is None:
        import tree as T
        t0 = time.time()
        _TREE = (T.load_tree(), T)
        print(f"(树载入 {time.time()-t0:.2f}s)", file=sys.stderr)
    return _TREE


def main():
    ap = argparse.ArgumentParser(prog="ask", add_help=True)
    ap.add_argument("question", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.9)
    args = ap.parse_args()

    tr, T = get_tree()

    if args.question:
        q = " ".join(args.question)
        if args.verbose:
            start = T.pick_start(tr, q, args.order)
            print(f"起点: {start!r}", file=sys.stderr)
        print(T.answer(tr, q, random.Random(), topk=args.topk,
                       order=args.order, temperature=args.temp))
        return

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/q", "/exit"):
            break
        print(T.answer(tr, q, random.Random(), topk=args.topk,
                       order=args.order, temperature=args.temp))


if __name__ == "__main__":
    main()
