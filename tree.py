"""概率树生成器（多阶 backoff）。

从高阶 n-gram 往下走：查得到就用长上下文（连贯），
查不到就自动降一阶（保证能接下去）。
"""
import os
import pickle
import random

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
CACHE = os.path.join(HERE, "data/tree.pkl")


def load_tree(path=CACHE):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "trees" in blob:
        return blob
    # 兼容旧的单阶格式
    if isinstance(blob, dict) and "tree" in blob:
        o = blob.get("order", 3)
        return {"orders": [o], "trees": {o: blob["tree"]}}
    return {"orders": [3], "trees": {3: blob}}


def branches_at(model, ctx, topk):
    """按阶数从高到低找 ctx 的分支。"""
    for o in model["orders"]:
        span = max(1, o - 1)
        key = ctx[-span:]
        if len(key) < span:
            continue
        br = model["trees"][o].get(key)
        if br:
            return br[:topk], o
    return None, None


def walk(model, start, max_len=40, rng=None, temperature=1.0, topk=3, min_out=3):
    rng = rng or random.Random()
    max_span = max(1, max(model["orders"]) - 1)
    ctx = start[-max_span:]
    out = []
    for _ in range(max_len):
        br, o = branches_at(model, ctx, topk)
        if not br:
            break
        cand = [c for c, _ in br]
        probs = np.array([p for _, p in br], dtype=np.float64)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        probs /= probs.sum()
        ch = cand[int(np.random.default_rng(rng.randrange(1 << 30)).choice(len(cand), p=probs))]
        out.append(ch)
        ctx = (ctx + ch)[-max_span:]
        if ch == "\n":
            break
        if ch in "。！？" and len(out) >= min_out:
            break
    return "".join(out)


def pick_start(model, query):
    """在最高阶树里找与输入尾部重叠最长的节点。"""
    q = query.strip()
    if not q:
        return None
    high = model["trees"][model["orders"][0]]
    max_span = max(1, max(model["orders"]) - 1)

    best, score = None, 0
    for k in high:
        for L in range(min(max_span, len(q)), 0, -1):
            if q[-L:] in k:
                if L > score:
                    score, best = L, k
                break
    if best:
        return best

    # 退回低阶树找
    for o in model["orders"][1:]:
        span = max(1, o - 1)
        for k in model["trees"][o]:
            for L in range(min(span, len(q)), 0, -1):
                if q[-L:] in k:
                    if L > score:
                        score, best = L, k
                    break
        if best:
            return best
    return best


def extract_answer(text):
    for marker in ("\n答：", "答："):
        if marker in text:
            return text.split(marker)[-1].strip()
    return text.strip()


def answer(model, query, rng=None, max_len=60, temperature=0.9, topk=3, order=None, tries=8):
    rng = rng or random.Random()
    seed = pick_start(model, query)
    if seed is None:
        return "这个我还不太会说，教教我吧。"

    best = ""
    for _ in range(tries):
        seg = walk(model, seed, max_len=max_len, rng=rng,
                   temperature=temperature, topk=topk)
        s = seed + seg
        ans = extract_answer(s)
        if 4 <= len(ans) <= 45 and ans.endswith(("。", "！", "？")):
            return ans
        if len(ans) > len(best):
            best = ans

    if best and not best.endswith(("。", "！", "？")):
        best += "。"
    return best or "这个我还不太会说，教教我吧。"
