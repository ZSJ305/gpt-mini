"""第 3 关：真正的 LLM —— 用 numpy 从零手写、可训练的小型 GPT。

不含任何深度学习框架：前向、反向传播、Adam 优化器全部手写。
结构对齐那张简化图：
    输入层:  文本 -> Tokenizer -> input ids
    表示层:  Token Embedding + Position Embedding
    主干层:  Transformer Block x N
             (LN -> Multi-Head Attention -> 残差 -> LN -> Feed Forward -> 残差)
    输出层:  Final Norm -> Linear Output
"""
import math
import os
import pickle

import numpy as np


# ----------------------------------------------------------------------
# 基础模块
# ----------------------------------------------------------------------
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.g = np.ones(dim, dtype=np.float32)
        self.b = np.zeros(dim, dtype=np.float32)
        self.eps = eps

    def __call__(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        xh = (x - mu) / np.sqrt(var + self.eps)
        self.cache = (xh, var)
        return self.g * xh + self.b

    def backward(self, dy):
        xh, var = self.cache
        n = xh.shape[-1]
        std = np.sqrt(var + self.eps)
        self.dg = (dy * xh).sum(axis=tuple(range(dy.ndim - 1)))
        self.db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxh = dy * self.g
        dx = (1.0 / (n * std)) * (
            n * dxh
            - dxh.sum(-1, keepdims=True)
            - xh * (dxh * xh).sum(-1, keepdims=True)
        )
        return dx

    def params(self, prefix=""):
        return [(prefix + "g", self.g, None), (prefix + "b", self.b, None)]


class Embedding:
    """查表：ID -> 稠密向量。反向就是把梯度散射回对应行。"""

    def __init__(self, vocab, dim, scale=0.02):
        self.W = (np.random.randn(vocab, dim) * scale).astype(np.float32)
        self.idx = None

    def __call__(self, idx):
        self.idx = np.asarray(idx)
        return self.W[self.idx]

    def backward(self, dout):
        # 统一成 (N, D)，按 token 把梯度累加回词表行
        D = dout.shape[-1]
        flat_d = dout.reshape(-1, D)
        flat_i = np.broadcast_to(
            self.idx.reshape(-1), (flat_d.shape[0] // self.idx.size, self.idx.size)
        ).reshape(-1)
        # 行一热乘：onehot(T) 转置左乘梯度，纯矩阵运算，等价于 add.at
        V = self.W.shape[0]
        onehot = np.zeros((flat_i.shape[0], V), dtype=np.float32)
        onehot[np.arange(flat_i.shape[0]), flat_i] = 1.0
        self.dW = onehot.T @ flat_d
        return None

    def params(self, prefix=""):
        return [(prefix + "W", self.W, "dW")]


class Linear:
    def __init__(self, nin, nout, bias=True, scale=None):
        scale = scale if scale is not None else 1.0 / math.sqrt(nin)
        self.W = (np.random.randn(nin, nout) * scale).astype(np.float32)
        self.b = np.zeros(nout, dtype=np.float32) if bias else None
        self.x = None

    def __call__(self, x):
        self.x = x
        return x @ self.W + (self.b if self.b is not None else 0.0)

    def backward(self, dy):
        xf = self.x.reshape(-1, self.x.shape[-1])
        dyf = dy.reshape(-1, dy.shape[-1])
        self.dW = xf.T @ dyf
        if self.b is not None:
            self.db = dyf.sum(0)
        return dy @ self.W.T

    def params(self, prefix=""):
        p = [(prefix + "W", self.W, "dW")]
        if self.b is not None:
            p.append((prefix + "b", self.b, "db"))
        return p


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class CausalSelfAttention:
    """多头因果自注意力：query / key / value 三组向量互相打分。"""

    def __init__(self, dim, n_head):
        assert dim % n_head == 0
        self.n_head = n_head
        self.dh = dim // n_head
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)

    def __call__(self, x):
        B, T, C = x.shape
        h, dh = self.n_head, self.dh
        qkv = self.qkv(x)
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, h, dh).transpose(0, 2, 1, 3)   # (B,h,T,dh)
        k = k.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, h, dh).transpose(0, 2, 1, 3)

        att = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        att = np.where(mask, -1e9, att)
        att = softmax(att)

        self.cache = (B, T, C, h, dh, q, k, v, att)
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)

    def backward(self, dy):
        B, T, C, h, dh, q, k, v, att = self.cache
        dy = self.proj.backward(dy)              # 回到注意力输出空间
        dy = dy.reshape(B, T, h, dh).transpose(0, 2, 1, 3)

        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        ds = att * (datt - (datt * att).sum(-1, keepdims=True))
        ds = ds / math.sqrt(dh)

        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q

        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(B, T, C)

        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
        return self.qkv.backward(dqkv)

    def params(self, prefix=""):
        return self.qkv.params(prefix + "qkv.") + self.proj.params(prefix + "proj.")


class MLP:
    def __init__(self, dim, mult=4):
        self.fc1 = Linear(dim, mult * dim)
        self.fc2 = Linear(mult * dim, dim)
        self.pre = None

    def __call__(self, x):
        self.pre = self.fc1(x)
        return self.fc2(np.maximum(0, self.pre))   # GELU 的廉价替身：ReLU

    def backward(self, dy):
        dh = self.fc2.backward(dy)
        dh = dh * (self.pre > 0)
        return self.fc1.backward(dh)

    def params(self, prefix=""):
        return self.fc1.params(prefix + "fc1.") + self.fc2.params(prefix + "fc2.")


class Block:
    def __init__(self, dim, n_head, mult=4):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, mult)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def backward(self, dy):
        dattn = self.attn.backward(dy)
        dy = dy + self.ln1.backward(dattn)
        dmlp = self.mlp.backward(dy)
        return dy + self.ln2.backward(dmlp)

    def params(self, prefix=""):
        return (self.ln1.params(prefix + "ln1.")
                + self.attn.params(prefix + "attn.")
                + self.ln2.params(prefix + "ln2.")
                + self.mlp.params(prefix + "mlp."))


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class MiniGPT:
    def __init__(self, vocab, block_size=128, n_layer=4, n_head=4, dim=128, mult=4):
        self.vocab = vocab
        self.block_size = block_size
        self.tok_emb = Embedding(vocab, dim)
        self.pos_emb = Embedding(block_size, dim)
        self.blocks = [Block(dim, n_head, mult) for _ in range(n_layer)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab, bias=False, scale=0.02)
        self.cfg = dict(vocab=vocab, block_size=block_size, n_layer=n_layer,
                        n_head=n_head, dim=dim, mult=mult)

    def n_params(self):
        return sum(p.size for _, p, _ in self.all_params())

    def all_params(self):
        out = self.tok_emb.params("tok_emb.") + self.pos_emb.params("pos_emb.")
        for i, b in enumerate(self.blocks):
            out += b.params(f"b{i}.")
        return out + self.ln_f.params("ln_f.") + self.head.params("head.")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "序列超过 block_size"
        x = self.tok_emb(idx) + self.pos_emb(np.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))

        if targets is None:
            return logits, None, None

        logits_f = logits.reshape(-1, self.vocab)
        tgt = targets.reshape(-1)
        probs = softmax(logits_f)
        n = tgt.shape[0]
        loss = float(-np.log(probs[np.arange(n), tgt] + 1e-9).mean())

        dlogits = probs.copy()
        dlogits[np.arange(n), tgt] -= 1.0
        dlogits /= n
        return logits, loss, dlogits.reshape(logits.shape)

    def backward(self, dlogits):
        dx = self.head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for b in reversed(self.blocks):
            dx = b.backward(dx)
        self.pos_emb.backward(dx)
        self.tok_emb.backward(dx)


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8, wd=0.01, max_norm=1.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.max_norm = wd, max_norm
        self.m = [np.zeros_like(it[1]) for it in params]
        self.v = [np.zeros_like(it[1]) for it in params]
        self.t = 0

    def step(self):
        self.t += 1
        total = 0.0
        grads = []
        for i, (name, p, gname) in enumerate(self.p):
            g = getattr(p, gname) if gname and hasattr(p, gname) else None
            grads.append((i, p, g))
            if g is not None:
                total += float((g ** 2).sum())
        scale = min(1.0, self.max_norm / (math.sqrt(total) + 1e-9))

        for i, p, g in grads:
            if g is None:
                continue
            g = g * scale + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------
# 推理与存取
# ----------------------------------------------------------------------
def named_arrays(model):
    out = {"tok_emb": model.tok_emb.W, "pos_emb": model.pos_emb.W}
    for i, b in enumerate(model.blocks):
        out[f"b{i}.ln1.g"] = b.ln1.g
        out[f"b{i}.ln1.b"] = b.ln1.b
        out[f"b{i}.attn.qkv.W"] = b.attn.qkv.W
        out[f"b{i}.attn.qkv.b"] = b.attn.qkv.b
        out[f"b{i}.attn.proj.W"] = b.attn.proj.W
        out[f"b{i}.attn.proj.b"] = b.attn.proj.b
        out[f"b{i}.ln2.g"] = b.ln2.g
        out[f"b{i}.ln2.b"] = b.ln2.b
        out[f"b{i}.mlp.fc1.W"] = b.mlp.fc1.W
        out[f"b{i}.mlp.fc1.b"] = b.mlp.fc1.b
        out[f"b{i}.mlp.fc2.W"] = b.mlp.fc2.W
        out[f"b{i}.mlp.fc2.b"] = b.mlp.fc2.b
    out["ln_f.g"] = model.ln_f.g
    out["ln_f.b"] = model.ln_f.b
    out["head.W"] = model.head.W
    return out


def save(model, tok, path):
    blob = {"cfg": model.cfg, "stoi": tok.stoi, "arrays": named_arrays(model)}
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    print(f"已保存 {path} ({os.path.getsize(path)/1e6:.2f} MB)")


def load(path):
    from tokenizer import Tokenizer
    with open(path, "rb") as f:
        blob = pickle.load(f)
    model = MiniGPT(**blob["cfg"])
    named = named_arrays(model)
    for k, arr in blob["arrays"].items():
        named[k][...] = arr
    return model, Tokenizer(blob["stoi"])


def generate(model, tok, prompt, max_new=80, temperature=0.8, top_k=20, seed=None):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt) or [tok.stoi.get("问", 0)]
    ctx = ids[-model.block_size:]
    out = list(ids)
    for _ in range(max_new):
        logits, _, _ = model.forward(np.array([ctx], dtype=np.int64))
        logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k:
            kth = np.sort(logits)[-min(top_k, len(logits))]
            logits = np.where(logits < kth, -1e9, logits)
        probs = softmax(logits)
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
        ctx = (ctx + [nxt])[-model.block_size:]
        if len(out) > 6 and tok.decode(out[-2:]) == "\n答":
            break
    return tok.decode(out)
