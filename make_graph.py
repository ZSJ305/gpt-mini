"""把语料库的「相邻概率」排成一张图（快速版，无迭代物理模拟）。

思路换掉慢的力导向：
    1. 字按出现频次排序
    2. 沿阿基米德螺线铺开 —— 频率相近的字自然落在相邻位置
    3. 只画最强的若干条共现边，线宽 = 相邻概率
    4. 输出 SVG（矢量，浏览器直接看）

全程 O(n log n)，无 n² 计算，秒级完成。
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_graph_data(text, min_freq=30, top_edges=2000):
    vocab = {}
    for ch in text:
        vocab[ch] = vocab.get(ch, 0) + 1

    kept = [(c, f) for c, f in vocab.items() if f >= min_freq]
    kept.sort(key=lambda x: -x[1])
    idx = {c: i for i, (c, _) in enumerate(kept)}
    n = len(idx)
    print(f"节点数 {n} (频次 >= {min_freq})，共 {len(vocab)} 种字")

    freq = np.array([f for _, f in kept], dtype=np.float64)

    trans = {}
    prev = None
    for ch in text:
        i = idx.get(ch)
        if i is None:
            prev = None
            continue
        if prev is not None and prev != i:
            k = prev * n + i
            trans[k] = trans.get(k, 0) + 1
        prev = i

    items = sorted(trans.items(), key=lambda x: -x[1])[:top_edges]
    ea = np.array([k // n for k, _ in items], dtype=np.int32)
    eb = np.array([k % n for k, _ in items], dtype=np.int32)
    w = np.array([v for _, v in items], dtype=np.float64)
    w = w / w.max()

    inv = [c for c, _ in kept]
    return inv, freq, ea, eb, w


def spiral_layout(n, size=2200, pad=100):
    """阿基米德螺线：按索引顺序从中心向外铺。索引即频率排名。"""
    cx = cy = size / 2
    rmax = size / 2 - pad
    out = np.zeros((n, 2), dtype=np.float64)
    golden = math.pi * (3 - math.sqrt(5))     # 黄金角，避免同一直线上堆积
    for i in range(n):
        t = (i + 0.5) / n                      # 0 -> 中心, 1 -> 边缘
        r = rmax * math.sqrt(t)                # sqrt 让面积均匀分布
        a = i * golden
        out[i, 0] = cx + r * math.cos(a)
        out[i, 1] = cy + r * math.sin(a)
    return out


def render_svg(inv, freq, pos, ea, eb, w, out_path, size=2200):
    fmax = freq.max()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        '<rect width="100%" height="100%" fill="#0c0e16"/>',
        '<g stroke-linecap="round">',
    ]
    for i in np.argsort(-w):
        a, b = int(ea[i]), int(eb[i])
        alpha = 0.05 + 0.5 * float(w[i]) ** 0.7
        lw = 0.3 + 3.0 * float(w[i]) ** 0.6
        parts.append(
            f'<line x1="{pos[a,0]:.1f}" y1="{pos[a,1]:.1f}" '
            f'x2="{pos[b,0]:.1f}" y2="{pos[b,1]:.1f}" '
            f'stroke="#5aa0ff" stroke-opacity="{alpha:.3f}" stroke-width="{lw:.2f}"/>')
    parts.append('</g>')

    for i in range(len(inv)):
        f = freq[i]
        r = 2.0 + 24 * (f / fmax) ** 0.45
        x, y = pos[i]
        op = 0.3 + 0.65 * (f / fmax) ** 0.5
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#ff963c" '
            f'fill-opacity="{op:.2f}" stroke="#ffdcb4" stroke-opacity="0.65" stroke-width="0.5"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{max(6, r*1.1):.1f}" fill="#14100c" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-family="PingFang SC, Noto Sans CJK SC, sans-serif">{inv[i]}</text>')

    parts.append('</svg>')
    open(out_path, "w", encoding="utf-8").write("\n".join(parts))
    print("已保存", out_path)


def main():
    import time
    t0 = time.time()
    text = open(os.path.join(HERE, "data/corpus_big.txt"), encoding="utf-8").read()
    inv, freq, ea, eb, w = build_graph_data(text)
    print(f"统计耗时 {time.time()-t0:.1f}s，边数 {len(ea)}")

    pos = spiral_layout(len(inv))
    render_svg(inv, freq, pos, ea, eb, w, os.path.join(HERE, "data/graph.svg"))
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
