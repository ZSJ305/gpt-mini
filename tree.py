"""概率树生成器：不用检索原句，改为沿着共现树往下走。

结构：
    根节点
      └─ 字A (概率)          ← 一层
           ├─ 字B (概率)     ← 二层
           │    ├─ 字C ...   ← 三层
           └─ 字D
每层按转移概率分叉，生成时从起点随机走一条路径，拼成句子。

优点：不会「答非所问」地照抄原句，输出天然多样、可无限延伸。
"""
import os
import pickle
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data/tree.pkl")


# ----------------------------------------------------------------------
# 建树：字级转移概率
# ----------------------------------------------------------------------
def build_tree(text, min_count=2, order=2):
    """在 UTF-8 字节层面建树。

    不用把文本编码成字符 ID（那是最大的性能瓶颈），
    直接对 bytes 统计转移 —— 中文一个字 3 字节，字节级一样能建模。
    """
    data = text.encode("utf-8")
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    span = max(1, order - 1)

    # 上下文 key：前 span 个字节
    ctx_key = np.zeros(len(arr) - span, dtype=np.int64)
    for k in range(span):
        ctx_key = ctx_key * 256 + arr[k:len(arr) - span + k]
    nxt = arr[span:]

    # 稀疏计数
    pair = np.stack([ctx_key, nxt], axis=1)
    uniq, counts = np.unique(pair, axis=0, return_counts=True)
    keep = counts >= min_count
    uniq, counts = uniq[keep], counts[keep]
    print(f"字节 n-gram 组合 {len(uniq)} 个（出现 >= {min_count} 次）")

    from collections import defaultdict
    tmp = defaultdict(list)
    for (ck, nx), c in zip(uniq.tolist(), counts.tolist()):
        tmp[ck].append((nx, c))

    # 上下文还原成字节串，分支按概率降序
    tree = {}
    for ck, items in tmp.items():
        total = sum(c for _, c in items)
        items.sort(key=lambda x: -x[1])
        s = ck
        ctx = bytearray()
        for _ in range(span):
            ctx.append(s % 256)
            s //= 256
        ctx.reverse()
        tree[bytes(ctx)] = [(i, c / total) for i, c in items]

    print(f"树节点 {len(tree)}，边 {sum(len(v) for v in tree.values())}（{order}-gram 字节级）")
    return tree


def save_tree(tree, path=CACHE):
    with open(path, "wb") as f:
        pickle.dump(tree, f, protocol=4)
    print(f"已保存 {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def load_tree(path=CACHE, corpus=None):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("首次构建概率树……")
    text = open(corpus or os.path.join(HERE, "data/corpus_big.txt"), encoding="utf-8").read()
    tree = build_tree(text)
    save_tree(tree, path)
    return tree


# ----------------------------------------------------------------------
# 走树生成
# ----------------------------------------------------------------------
def walk(tree, start, max_len=40, rng=None, temperature=1.0, topk=8):
    """从 start 开始沿概率树往下走，走出一条路径。

    tree 的 key 是上下文（order-1 个字），所以每走一步都要
    把新字拼到上下文末尾、丢掉最前面那个字。
    """
    rng = rng or random.Random()
    if not start:
        return ""
def walk(tree, start, max_len=40, rng=None, temperature=1.0, topk=4, order=2):
    """从 start 出发沿树往下走。

    每一步只在概率最高的前 topk 个分支里随机选一个 —— 既保证通顺，
    又有变化（同样的输入能给出不同回答）。
    """
    rng = rng or random.Random()
    if isinstance(start, str):
        start = start.encode("utf-8")
    span = max(1, order - 1)
    ctx = start[-span:]
    out = bytearray()
    for _ in range(max_len):
        branches = tree.get(ctx)
        if not branches:
            break
        branches = branches[:topk]
        cand = [c for c, _ in branches]
        probs = np.array([p for _, p in branches], dtype=np.float64)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        probs /= probs.sum()
        pick = int(np.random.default_rng(rng.randrange(1 << 30)).choice(len(cand), p=probs))
        b = cand[pick]
        out.append(b)
        ctx = (ctx + bytes([b]))[-span:]
        # 句末就停
        if out.endswith(b"\xe3\x80\x82") or out.endswith(b"\xef\xbc\x81") or out.endswith(b"\xef\xbc\x9f") or out.endswith(b"\n"):
            break
    return out.decode("utf-8", errors="ignore")


def pick_start(tree, query, order=2):
    """从问题里挑起点：优先用末尾的 span 个字节。"""
    span = max(1, order - 1)
    qb = query.encode("utf-8")
    for L in range(min(span, len(qb)), 0, -1):
        cand = qb[-L:]
        if cand in tree:
            return cand
    return None


def answer(tree, query, rng=None, max_len=60, temperature=0.9, topk=4, order=2, tries=6):
    """生成回答：从问题末尾接话，走树拼句子。"""
    rng = rng or random.Random()
    seed = pick_start(tree, query, order)
    if seed is None:
        return "这个我还不太会说，教教我吧。"

    best = ""
    for _ in range(tries):
        seg = walk(tree, seed, max_len=max_len, rng=rng,
                   temperature=temperature, topk=topk, order=order)
        if not seg:
            continue
        s = seed.decode("utf-8", errors="ignore") + seg
        if s.endswith(("。", "！", "？")) and 6 <= len(s) <= 45:
            return s
        if len(s) > len(best):
            best = s

    if best and not best.endswith(("。", "！", "？")):
        best += "。"
    return best or "这个我还不太会说，教教我吧。"
