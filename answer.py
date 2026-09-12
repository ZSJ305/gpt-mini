"""统计式回答生成器：不训练，直接查语料库里的相邻概率。

流程（对应用户的描述）：
    1. input 拆成 token
    2. 去语料库里找这些 token 出现的位置
    3. 把它「周围」的 token 框进来，作为候选上下文
    4. 按相邻概率一步步生成回答

性能关键：在 UTF-8 字节层面做统计。
    - 中文一个字 = 3 字节，字节级转移表只有 256x256
    - 用 bincount 一次性算完，0.3 秒搞定千万字符
    - 生成时按字节流走，最后 decode 回文字
"""
import os
import pickle
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data/stats.pkl")


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------
def build(text):
    data = text.encode("utf-8")
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)

    # 256x256 的字节转移计数，bincount 一次算完
    a, b = arr[:-1], arr[1:]
    trans = np.bincount(a * 256 + b, minlength=65536).reshape(256, 256).astype(np.float32)
    freq = np.bincount(arr, minlength=256).astype(np.float32)

    # 问答对 + 倒排索引
    qa = []
    lines = text.split("\n")
    for i in range(0, len(lines) - 1, 2):
        q, ans = lines[i], lines[i + 1]
        if q.startswith("问：") and ans.startswith("答："):
            qa.append((q[2:], ans[2:]))

    inv = {}
    for qi, (qq, _) in enumerate(qa):
        for ch in set(qq):
            inv.setdefault(ch, []).append(qi)

    return {"trans": trans, "freq": freq, "qa": qa, "inv": inv,
            "n": len(qa), "bytes": len(data)}


def load_stats(path=os.path.join(HERE, "data/corpus_big.txt")):
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    print("首次运行，正在统计语料……")
    st = build(open(path, encoding="utf-8").read())
    with open(CACHE, "wb") as f:
        pickle.dump(st, f, protocol=4)
    print(f"统计完成：{st['n']} 条问答，{st['bytes']/1024/1024:.1f} MB")
    return st


# ----------------------------------------------------------------------
# 1. token 化
# ----------------------------------------------------------------------
def tokenize(s):
    """字级 token：中文一个字一个 token（最贴近"词元"的直觉）。"""
    return list(s)


# ----------------------------------------------------------------------
# 2. 检索：找出命中最多的条目，把周围上下文"框"进来
# ----------------------------------------------------------------------
def retrieve(st, query, topk=8):
    q = query.strip()
    inv, qa = st["inv"], st["qa"]

    hit_count = {}
    for ch in set(q):
        for qi in inv.get(ch, ()):
            hit_count[qi] = hit_count.get(qi, 0) + 1
    if not hit_count:
        return []

    max_hit = max(hit_count.values())
    cand = [qi for qi, c in hit_count.items() if c >= max(1, max_hit - 1)][:4000]

    scored = []
    denom = max(1, len(set(q)))
    for qi in cand:
        qq, aa = qa[qi]
        score = hit_count[qi] / denom
        # 连续子串命中是强信号，越长权重越高
        for L in range(min(len(q), 6), 1, -1):
            ok = False
            for i in range(len(q) - L + 1):
                if q[i:i + L] in qq:
                    score += 0.6 * L
                    ok = True
                    break
            if ok:
                break
        # 长度差惩罚：问句比输入长太多，多半是蒙中的
        score -= 0.02 * max(0, len(qq) - len(q))
        # 覆盖率：输入里有几个字在问句中出现过
        cov = sum(1 for ch in set(q) if ch in qq) / denom
        score *= 0.4 + 0.6 * cov
        # 尾部错位惩罚：「你好棒」vs「你好」——多出来的字在问句里找不到，扣分
        if len(q) > len(qq) and qq and qq not in q:
            score -= 0.5 * (len(q) - len(qq))
        scored.append((score, qi, qq, aa))

    scored.sort(key=lambda x: -x[0])
    # 去重：同样的问题文本只留最高分那条
    seen = set()
    out = []
    for h in scored:
        if h[2] in seen:
            continue
        seen.add(h[2])
        out.append(h)
        if len(out) >= topk:
            break
    return out


# ----------------------------------------------------------------------
# 3. 按相邻概率生成（字节级 → 解码回文字）
# ----------------------------------------------------------------------
def next_byte(st, cur, rng, temp=1.0):
    row = st["trans"][cur].astype(np.float64)
    if temp != 1.0:
        row = np.power(row, 1.0 / temp)
    s = row.sum()
    if s <= 0:
        return None
    p = row / s
    return int(rng.choice(256, p=p))


def generate(st, seed, max_len=60, rng=None, temp=1.0):
    """从 seed 的最后一个字节出发，按共现概率把"周围"的字节滚出来。"""
    rng = rng or random.Random()
    nb = np.random.default_rng(rng.randint(0, 1 << 30))
    buf = bytearray(seed.encode("utf-8"))
    cur = buf[-1] if buf else None
    if cur is None:
        return ""
    out = bytearray()
    for _ in range(max_len):
        row = st["trans"][cur].astype(np.float64)
        if temp != 1.0:
            row = np.power(row, 1.0 / temp)
        s = row.sum()
        if s <= 0:
            break
        p = row / s
        nx = int(nb.choice(256, p=p))
        out.append(nx)
        cur = nx
        # 遇到句末就停
        try:
            tail = out.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if tail.endswith(("。", "！", "？", "\n")):
            break
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return out.decode("utf-8", errors="ignore")


# ----------------------------------------------------------------------
# 4. 回答
# ----------------------------------------------------------------------
def answer(st, query, verbose=False):
    hits = retrieve(st, query, topk=8)
    if not hits or hits[0][0] < 0.5:
        return "这个我还不太会说，教教我吧。"

    if verbose:
        print("--- 框住的语料上下文 ---")
        for s, qi, qq, aa in hits[:5]:
            print(f"  [{s:.2f}] 问：{qq}  →  答：{aa}")

    if hits[0][0] >= 1.2 or len(query) >= 6:
        return hits[0][3]

    pool = [h[3] for h in hits if h[0] >= hits[0][0] * 0.75]
    return random.choice(pool or [hits[0][3]])


# ----------------------------------------------------------------------
def main():
    import time
    t0 = time.time()
    st = load_stats()
    print(f"载入耗时 {time.time()-t0:.1f}s")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"> {q}")
        print(answer(st, q, verbose=True))
        return

    print("输入问题，Ctrl-C 退出。")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print(answer(st, q, verbose=True))


if __name__ == "__main__":
    main()
