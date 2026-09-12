"""最简版：随机采样语料，不做任何统计。

思路就像你说的 —— 随机数生成一样的：
    1. 输入切成词
    2. 在语料里随机找一个包含这些词的位置
    3. 从那个位置往后读一段，就是回答

没有概率表，没有树，没有 n-gram。只用一个随机数。
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data/corpus_big.txt")

_lines = None
_index = None


def load():
    global _lines, _index
    if _lines is not None:
        return
    text = open(CORPUS, encoding="utf-8").read()
    _lines = text.split("\n")
    print(f"载入 {len(_lines)} 行")


def answer(query, rng=None):
    """随机挑一条包含输入词的语料，把它的答句给你。"""
    load()
    rng = rng or random
    q = query.strip()

    # 找出所有包含输入内容的行号
    hits = [i for i, ln in enumerate(_lines) if q in ln]
    if not hits:
        # 退一步：按字拆开，找同时包含最多字的行
        qs = set(q)
        best, best_n = [], 0
        for i, ln in enumerate(_lines):
            n = len(qs & set(ln))
            if n > best_n:
                best_n, best = n, [i]
            elif n == best_n and n > 0:
                best.append(i)
        hits = best

    if not hits:
        return "这个我还不太会说，教教我吧。"

    # 随机挑一个
    if os.environ.get("MINIMODEL_VERBOSE"):
        print(f"命中 {len(hits)} 条，随机取第 {rng.randrange(len(hits))} 条")
    i = rng.choice(hits)
    line = _lines[i]

    # 命中的是问句，就取下一行答句
    if line.startswith("问："):
        nxt = _lines[i + 1] if i + 1 < len(_lines) else ""
        if nxt.startswith("答："):
            return nxt[2:]
        return line[2:]
    return line[2:] if line.startswith("答：") else line


if __name__ == "__main__":
    import sys
    load()
    for q in sys.argv[1:] or ["你好", "今天天气", "什么是大模型"]:
        print(f"{q} → {answer(q)}")
