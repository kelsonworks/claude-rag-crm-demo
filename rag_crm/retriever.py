"""Dependency-free TF-IDF retrieval over section chunks.

Deliberately simple: pure-Python term frequency / inverse document frequency
with cosine similarity. No numpy, no sklearn, no embedding service — it runs
anywhere Python runs, and for a corpus of a few hundred sections it is more
than accurate enough. Swapping this module for a vector database is a
contained change (see README → Scaling notes).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunker import Chunk

# Minimal English stopword list — enough to stop function words from
# dominating the similarity scores.
_STOPWORDS = frozenset(
    """a about above after again all an and any are as at be before both but
    by can come do does down during each few for from get had has have here
    how i if in into is it its just me more most my of on only or other our
    out over so some such than that the their them then there these they this
    to too und under until up us very was we what when where which while who
    will with would you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(token: str) -> str:
    """Light suffix stripping so 'cooling' matches 'cool', 'fees' matches 'fee'.

    Applied identically at index time and query time, so it only ever adds
    cross-form matches — equal words always still match.
    """
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class TfidfRetriever:
    """Index a list of chunks once, then answer top-k similarity queries."""

    def __init__(self, chunks: list[Chunk], heading_weight: int = 3):
        """heading_weight: section headings are strong topical signals, so
        their tokens are counted this many times in each chunk's vector."""
        self.chunks = chunks
        token_lists = [
            tokenize(c.section) * heading_weight + tokenize(c.text) for c in chunks
        ]

        df: Counter[str] = Counter()
        for tokens in token_lists:
            df.update(set(tokens))

        n = len(chunks)
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}
        self._default_idf = math.log(n + 1) + 1.0  # unseen term
        self._vectors = [self._vectorize(tokens) for tokens in token_lists]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        vec = {
            term: (count / len(tokens)) * self._idf.get(term, self._default_idf)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm:
            vec = {term: w / norm for term, w in vec.items()}
        return vec

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return up to top_k (chunk, cosine_similarity) pairs, best first."""
        query_vec = self._vectorize(tokenize(query))
        scored = []
        for chunk, vec in zip(self.chunks, self._vectors):
            score = sum(w * vec.get(term, 0.0) for term, w in query_vec.items())
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
