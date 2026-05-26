"""PPR engine wrapping igraph.personalized_pagerank.

Two modes:
  multi_seed     one PPR with reset = {topic: w_t, answer: w_a}
  bidirectional  two independent PPRs (topic-seeded, answer-seeded) then product

Seed weighting (multi_seed only):
  uniform              reset[topic] = reset[answer] = 0.5
  degree_inv           hub gets smaller reset (counter walk bias at high damping)
  degree_proportional  hub gets larger reset (counter dilution at low damping)

Uses igraph's standard call:
  graph.personalized_pagerank(
      vertices=range(N),
      damping=<cfg>,
      directed=False,
      weights='weight',
      reset=<np.ndarray [N]>,
      implementation='prpack',
  )
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Mode = Literal["multi_seed", "bidirectional"]
SeedWeighting = Literal["uniform", "degree_inv", "degree_proportional"]


def _build_reset(
    n: int,
    assignments: dict[int, float],
) -> np.ndarray:
    r = np.zeros(n, dtype=np.float64)
    for nid, w in assignments.items():
        r[nid] = w
    if r.sum() <= 0:
        raise ValueError("reset vector is all-zero")
    return r


def _compute_seed_weights(
    graph,
    topic_id: int,
    answer_id: int,
    seed_weighting: SeedWeighting,
) -> tuple[float, float]:
    if seed_weighting == "uniform":
        return 0.5, 0.5

    d_t = float(graph.degree(topic_id))
    d_a = float(graph.degree(answer_id))
    total = d_t + d_a
    if total == 0:
        return 0.5, 0.5

    if seed_weighting == "degree_inv":
        # hub (large deg) gets smaller reset
        return d_a / total, d_t / total
    if seed_weighting == "degree_proportional":
        # hub (large deg) gets larger reset
        return d_t / total, d_a / total

    raise ValueError(f"unknown seed_weighting: {seed_weighting}")


def _single_ppr(
    graph,
    reset: np.ndarray,
    damping: float,
) -> np.ndarray:
    scores = graph.personalized_pagerank(
        vertices=range(graph.vcount()),
        damping=damping,
        directed=False,
        weights="weight",
        reset=reset.tolist(),
        implementation="prpack",
    )
    return np.asarray(scores, dtype=np.float64)


def _combine_bidirectional(pi_t: np.ndarray, pi_a: np.ndarray, combine: str) -> np.ndarray:
    """Combine two PPR vectors → single corridor score (higher = better)."""
    if combine == "product":
        return pi_t * pi_a
    if combine == "rank_product":
        # rank: 0 = highest π value, N-1 = lowest.
        # We want score = higher is better, so invert rank product.
        rank_t = np.argsort(-pi_t).argsort().astype(np.float64)
        rank_a = np.argsort(-pi_a).argsort().astype(np.float64)
        return 1.0 / ((rank_t + 1.0) * (rank_a + 1.0))
    raise ValueError(f"unknown bidir_combine: {combine}")


def run_ppr(
    graph,
    topic_id: int,
    answer_id: int,
    mode: Mode = "bidirectional",
    damping: float = 0.15,
    seed_weighting: SeedWeighting = "uniform",
    bidir_combine: str = "product",
    return_components: bool = False,
):
    """Return per-node score vector [N] (higher = better corridor fit).

    If return_components=True, also returns (pi_t, pi_a) for PPR-aware Yen's cost.
    For multi_seed mode, pi_t / pi_a are returned as None.
    """
    n = graph.vcount()

    if mode == "multi_seed":
        w_t, w_a = _compute_seed_weights(graph, topic_id, answer_id, seed_weighting)
        reset = _build_reset(n, {topic_id: w_t, answer_id: w_a})
        score = _single_ppr(graph, reset, damping)
        if return_components:
            return score, None, None
        return score

    if mode == "bidirectional":
        reset_t = _build_reset(n, {topic_id: 1.0})
        reset_a = _build_reset(n, {answer_id: 1.0})
        pi_t = _single_ppr(graph, reset_t, damping)
        pi_a = _single_ppr(graph, reset_a, damping)
        score = _combine_bidirectional(pi_t, pi_a, bidir_combine)
        if return_components:
            return score, pi_t, pi_a
        return score

    raise ValueError(f"unknown mode: {mode}")
