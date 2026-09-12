"""第 2 关：语言表示。把人类文本翻译成模型能吃的数字。

最朴素也最透明的方案——字级 tokenizer。
每个不同的字符就是一个 token，查表得到 ID。
"""
import json
import os


class Tokenizer:
    def __init__(self, stoi=None):
        self.stoi = stoi or {}
        self.itos = {i: s for s, i in self.stoi.items()}

    @classmethod
    def train(cls, text):
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return cls(stoi)

    def encode(self, text):
        unk = self.stoi.get("<unk>", 0)
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["stoi"])


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "data/corpus.txt"), encoding="utf-8").read()

    tok = Tokenizer.train(text)
    print("vocab_size:", tok.vocab_size)
    print("字典前 20 个:", "".join(sorted(tok.stoi))[:20])

    s = "问：你好\n答：你好呀，我是小豆。\n"
    ids = tok.encode(s)
    print("原文:", repr(s))
    print("ids :", ids)
    print("还原:", repr(tok.decode(ids)))
    print("往返一致:", tok.decode(ids) == s)

    tok.save(os.path.join(here, "data/tokenizer.json"))
    print("已保存 data/tokenizer.json")
