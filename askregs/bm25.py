"""BM25 keyword search, written from scratch so every step is visible.

BM25 scores a chunk higher when it contains the query's words, especially rare words
("refuge", "evacuation") rather than common ones ("should", "building"), and it stops
rewarding a word after it has appeared a few times. It is the classic baseline that
embedding (meaning-based) search is measured against.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or that the
their there these this to was were which with should may can will not any all other than""".split())


def stem(word: str) -> str:
    """Very light stemming so 'stairs'/'stair' and 'doors'/'door' match."""
    for suffix in ("ings", "ing", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b  # k1: how fast repeats stop counting; b: how much long chunks are penalised
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        n = len(self.docs)
        doc_freq = Counter(term for d in self.docs for term in d)
        # Inverse document frequency: rare words weigh more.
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        result = []
        for counts, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result
