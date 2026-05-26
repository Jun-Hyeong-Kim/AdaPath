"""Query-conditioned edge weighting.

w(u, r, v) = beta * s(r) + (1 - beta) * (s(u) + s(v)) / 2

Shape:
  node_sim : [N]   in [0, 1]
  rel_sim  : [R=18]  in [0, 1]    (rel_names order matches PrimeKG edge_type_dict)
  weights  : [E]   clipped to >= eps so PPR never sees zero-weight edges
"""

from __future__ import annotations

import numpy as np


def compute_edge_weights(
    edge_src: np.ndarray,          # [E] int
    edge_dst: np.ndarray,          # [E] int
    edge_rel: np.ndarray,          # [E] int (rel type id)
    node_sim: np.ndarray,          # [N] float in [0, 1]
    rel_sim: np.ndarray,           # [R] float in [0, 1]
    beta: float = 0.3,
    eps: float = 1e-6,
) -> np.ndarray:
    node_sim = node_sim.astype(np.float32)
    rel_sim = rel_sim.astype(np.float32)

    s_r = rel_sim[edge_rel]
    s_u = node_sim[edge_src]
    s_v = node_sim[edge_dst]
    w = beta * s_r + (1.0 - beta) * 0.5 * (s_u + s_v)
    return np.clip(w, eps, None)


def apply_weights_to_graph(graph, weights: np.ndarray) -> None:
    """Set graph.es['weight'] = weights (list form required by igraph)."""
    graph.es["weight"] = weights.tolist()


def extract_edge_arrays(graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache (src, dst, rel_id) arrays from an igraph. Call once per run."""
    edgelist = np.asarray(graph.get_edgelist(), dtype=np.int32)     # [E, 2]
    rel_id = np.asarray(graph.es["rel_id"], dtype=np.int32)         # [E]
    return edgelist[:, 0], edgelist[:, 1], rel_id
