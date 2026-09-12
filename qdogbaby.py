"""QdogBaby —— 第 1 关：规则替换的伪 LLM。

它不会理解语义，只是在做字符替换。存在的意义是让人先看清
「能跑」和「真的懂」之间的差距。
"""


class QdogBaby:
    """纯规则版：把疑问句改成感叹句，把否定改成肯定，仅此而已。"""

    RULES = [
        ("吗？", "！"),
        ("呢？", "！"),
        ("吧？", "吧。"),
        ("怎么", "就这样"),
        ("为什么", "因为就是这样"),
    ]

    def chat(self, text: str) -> str:
        for old, new in self.RULES:
            if old in text:
                return text.replace(old, new)
        return text + "。"


if __name__ == "__main__":
    model = QdogBaby()
    for q in ["会说话吗？", "是人工智能吗？", "今天天气怎么样？", "你好"]:
        print(">", q)
        print(model.chat(q))
        print()
