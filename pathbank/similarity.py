"""Query-conditioned BM25 similarity backbone for edge weighting.

Returns scores in [0, 1] so the edge_weight formula
    w = β·s(r) + (1-β)·(s(u)+s(v))/2
operates on a consistent range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_BM25_TOKEN_RE = re.compile(r"[^\w\s]")


def tokenize_for_bm25(text: str) -> list[str]:
    """Match embedding_cache._tokenize_for_bm25 exactly; used on the query side."""
    text = text.lower()
    text = _BM25_TOKEN_RE.sub(" ", text)
    return [t for t in text.split() if t]


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale to [0, 1] via min-max. Returns float32."""
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


@dataclass
class BM25Backbone:
    """Holds BM25 indexes for nodes and relations (length-agnostic, b=0)."""
    bm25_node: object   # BM25Okapi
    bm25_rel: object
    rel_names: list[str]

    def node_sim(self, query: str) -> np.ndarray:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return np.zeros(len(self.bm25_node.doc_freqs), dtype=np.float32)
        return minmax_normalize(self.bm25_node.get_scores(tokens))

    def rel_sim(self, query: str) -> np.ndarray:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return np.zeros(len(self.rel_names), dtype=np.float32)
        return minmax_normalize(self.bm25_rel.get_scores(tokens))


def load_backbone() -> BM25Backbone:
    """Load BM25 backbone (length-agnostic Okapi, b=0)."""
    from pathbank.embedding_cache import load_bm25_node, load_bm25_rel, CACHE_DIR
    node_data = load_bm25_node(CACHE_DIR / "bm25_node_nolen.pkl")
    rel_data = load_bm25_rel(CACHE_DIR / "bm25_rel_nolen.pkl")
    return BM25Backbone(
        bm25_node=node_data["bm25"],
        bm25_rel=rel_data["bm25"],
        rel_names=rel_data["rel_names"],
    )
