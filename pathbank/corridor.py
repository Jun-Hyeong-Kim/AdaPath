"""Corridor subgraph extraction + fallback ladder + k-shortest paths.

Given PPR scores, induce the subgraph on top-K nodes (plus topic, answer).
If topic↔answer are disconnected in the subgraph, expand K along the ladder.
Final fallback: full-graph bidirectional BFS.

k-shortest paths: Yen's algorithm on the corridor subgraph using
cost = 1 / (edge_weight + eps), so high-weight (query-relevant) edges are
preferred by the shortest-path search.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def induce_corridor(
    graph,                          # igraph.Graph (full)
    ppr_score: np.ndarray,          # [N]
    topic_id: int,
    answer_id: int,
    k: int,
):
    """Return (subgraph, mapping dict {corridor_vertex_idx -> full_vertex_idx}).

    Top-K nodes by PPR score (+ topic/answer always included).
    The returned subgraph preserves edge 'weight' and 'relation' attributes.
    """
    top_k = set(np.argpartition(-ppr_score, k)[:k].tolist())
    top_k.add(topic_id)
    top_k.add(answer_id)
    vs = sorted(top_k)
    sub = graph.subgraph(vs)
    # Map corridor vertex idx -> full graph vertex idx.
    # (igraph.subgraph keeps vertex order of the given list, so position == new idx.)
    return sub, {i: v for i, v in enumerate(vs)}


def _has_topic_answer_path(subgraph, new_topic: int, new_answer: int) -> bool:
    try:
        d = subgraph.shortest_paths_dijkstra(
            source=[new_topic], target=[new_answer], weights=None
        )[0][0]
    except Exception:
        return False
    return d != float("inf")


def build_corridor(
    graph,
    ppr_score: np.ndarray,
    topic_id: int,
    answer_id: int,
    ladder: Sequence[int] = (150, 300, 500, 1000),
):
    """Return (subgraph, mapping, corridor_k_used, fallback_triggered, fallback_reason).

    Tries each K in the ladder; returns the smallest K where subgraph contains
    a topic-answer path. If all fail, `fallback_triggered=True` and the
    subgraph is None — the caller should fall back to full-graph BFS.
    """
    for k in ladder:
        sub, mapping = induce_corridor(graph, ppr_score, topic_id, answer_id, k)
        inv = {v: i for i, v in mapping.items()}
        new_t, new_a = inv[topic_id], inv[answer_id]
        if _has_topic_answer_path(sub, new_t, new_a):
            return sub, mapping, k, False, None

    return None, None, ladder[-1], True, f"disconnected_at_K<={ladder[-1]}"


def _best_edge(subgraph, u: int, v: int, costs: list[float]) -> int:
    """Among all edges between u and v (multi-edges), return the one with
    minimum cost (= maximum weight)."""
    eids = subgraph.get_eids([(u, v)], directed=False, error=False)
    # igraph returns a single eid by default; to find all multi-edges we
    # must iterate incident edges.
    incidents = set(subgraph.incident(u, mode="all"))
    best = None
    best_cost = float("inf")
    for eid in incidents:
        e = subgraph.es[eid]
        if (e.source == v) or (e.target == v):
            if costs[eid] < best_cost:
                best_cost = costs[eid]
                best = eid
    return best if best is not None else eids[0]


def _compute_edge_costs(
    subgraph,
    mode: str = "edge_weight",
    ppr_score_local: np.ndarray | None = None,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> list[float]:
    """Compute edge costs for Yen's.

    mode:
      edge_weight:   cost = 1 / edge_weight                                 (current default)
      ppr_mix:       cost = 1 / (α · edge_weight + (1-α) · ppr_edge_score)
      ppr_only:      cost = 1 / ppr_edge_score

    ppr_edge_score(u, v) = (normalize(ppr[u]) + normalize(ppr[v])) / 2
    where normalize is per-query max-normalize → [0, 1].
    """
    weights = np.asarray(subgraph.es["weight"], dtype=np.float64)

    if mode == "edge_weight":
        return [1.0 / (float(w) + eps) for w in weights]

    if ppr_score_local is None:
        raise ValueError(f"mode={mode} requires ppr_score_local")

    # Per-query max-normalize so PPR scale matches edge_weight [0,1] range.
    pm = ppr_score_local.max()
    ppr_norm = ppr_score_local / (pm + eps)           # [N_sub], [0,1]

    edgelist = np.asarray(subgraph.get_edgelist(), dtype=np.int64)
    u_idx, v_idx = edgelist[:, 0], edgelist[:, 1]
    ppr_edge = (ppr_norm[u_idx] + ppr_norm[v_idx]) / 2.0     # [E]

    if mode == "ppr_only":
        return [1.0 / (float(x) + eps) for x in ppr_edge]

    if mode == "ppr_mix":
        combined = alpha * weights + (1.0 - alpha) * ppr_edge
        return [1.0 / (float(x) + eps) for x in combined]

    raise ValueError(f"unknown yen_cost_mode: {mode}")


def yen_k_shortest(
    subgraph,
    topic_local: int,
    answer_local: int,
    k: int = 5,
    max_hops: int = 5,
    eps: float = 1e-8,
    raw_paths_multiplier: int = 3,
    cost_mode: str = "edge_weight",
    ppr_score_local: np.ndarray | None = None,
    ppr_alpha: float = 0.5,
):
    """Yen's k-shortest simple paths. Cost function selectable by `cost_mode`."""
    costs = _compute_edge_costs(
        subgraph, mode=cost_mode,
        ppr_score_local=ppr_score_local, alpha=ppr_alpha, eps=eps,
    )
    subgraph.es["cost"] = costs

    raw_paths = subgraph.get_k_shortest_paths(
        v=topic_local, to=answer_local,
        k=k * raw_paths_multiplier, weights="cost", mode="all",
    )

    seen: dict[tuple[int, ...], float] = {}
    for path in raw_paths:
        if len(path) - 1 > max_hops:
            continue
        total = 0.0
        for i in range(len(path) - 1):
            eid = _best_edge(subgraph, path[i], path[i + 1], costs)
            total += costs[eid]
        key = tuple(path)
        if key not in seen or total < seen[key]:
            seen[key] = total

    ordered = sorted(seen.items(), key=lambda kv: kv[1])[:k]
    return [(list(p), c) for p, c in ordered]


def yen_k_shortest_by_hop(
    subgraph,
    topic_local: int,
    answer_local: int,
    k_per_hop: int = 5,
    max_hops: int = 5,
    eps: float = 1e-8,
    raw_k: int = 100,
    cost_mode: str = "edge_weight",
    ppr_score_local: np.ndarray | None = None,
    ppr_alpha: float = 0.5,
    dedup: str = "none",
):
    """Run Yen's with a large k and bucket paths by hop length.

    dedup:
      none       — pure cost-order top-k (default, same as before).
      type_sig   — unique (node_type seq, relation seq) only, min-cost representative.
      first_type — diverse first-intermediate type; fill remaining by cost-order.
    """
    costs = _compute_edge_costs(
        subgraph, mode=cost_mode,
        ppr_score_local=ppr_score_local, alpha=ppr_alpha, eps=eps,
    )
    subgraph.es["cost"] = costs

    raw_paths = subgraph.get_k_shortest_paths(
        v=topic_local, to=answer_local,
        k=raw_k, weights="cost", mode="all",
    )

    # Dedup by node sequence, keep min-cost representative
    seen: dict[tuple[int, ...], float] = {}
    for path in raw_paths:
        h = len(path) - 1
        if h == 0 or h > max_hops:
            continue
        total = 0.0
        for i in range(len(path) - 1):
            eid = _best_edge(subgraph, path[i], path[i + 1], costs)
            total += costs[eid]
        key = tuple(path)
        if key not in seen or total < seen[key]:
            seen[key] = total

    # Bucket by hop length, ordered by cost within each bucket
    buckets_raw: dict[int, list] = {h: [] for h in range(1, max_hops + 1)}
    for key, cost in sorted(seen.items(), key=lambda kv: kv[1]):
        h = len(key) - 1
        if 1 <= h <= max_hops:
            buckets_raw[h].append((list(key), cost))

    # Apply dedup per bucket
    buckets: dict[int, list] = {h: [] for h in range(1, max_hops + 1)}
    for h, items in buckets_raw.items():
        if dedup == "none":
            buckets[h] = items[:k_per_hop]
            continue

        # Need type/rel sigs for dedup
        def _sig(node_path):
            types = tuple(subgraph.vs[int(v)]["type"] for v in node_path)
            rels = []
            for i in range(len(node_path) - 1):
                eid = _best_edge(subgraph, node_path[i], node_path[i + 1], costs)
                rels.append(subgraph.es[eid]["relation"] if eid >= 0 else "?")
            return types, tuple(rels)

        if dedup == "type_sig":
            chosen_sigs = set()
            for path, cost in items:
                sig = _sig(path)
                if sig in chosen_sigs:
                    continue
                chosen_sigs.add(sig)
                buckets[h].append((path, cost))
                if len(buckets[h]) >= k_per_hop:
                    break

        elif dedup == "first_type":
            chosen_first: set = set()
            used = set()
            for idx, (path, cost) in enumerate(items):
                if len(path) < 2:
                    buckets[h].append((path, cost))
                    used.add(idx)
                    if len(buckets[h]) >= k_per_hop:
                        break
                    continue
                ftype = subgraph.vs[int(path[1])]["type"]
                if ftype in chosen_first:
                    continue
                chosen_first.add(ftype)
                buckets[h].append((path, cost))
                used.add(idx)
                if len(buckets[h]) >= k_per_hop:
                    break
            # Fill remaining by cost-order
            if len(buckets[h]) < k_per_hop:
                for idx, (path, cost) in enumerate(items):
                    if idx in used:
                        continue
                    buckets[h].append((path, cost))
                    if len(buckets[h]) >= k_per_hop:
                        break
        else:
            raise ValueError(f"unknown dedup: {dedup}")

    return buckets


def _best_edge_full_graph(graph, u: int, v: int, eps: float = 1e-8) -> int:
    """Full-graph version of _best_edge — picks min-cost edge among multi-edges."""
    incidents = set(graph.incident(u, mode="all"))
    best = None
    best_cost = float("inf")
    for eid in incidents:
        e = graph.es[eid]
        other = e.target if e.source == u else e.source
        if other == v:
            w = float(e["weight"]) if "weight" in e.attributes() else 1.0
            c = 1.0 / (w + eps)
            if c < best_cost:
                best_cost = c
                best = eid
    return best if best is not None else graph.get_eid(u, v, error=False)


def translate_path(
    path_local: list[int],
    mapping: dict[int, int] | None,
    graph,
) -> tuple[list[int], list[tuple[str, str, str]]]:
    """Translate a local path → full-graph ids + (src, relation, dst) triples.

    Among multi-edges between consecutive nodes, picks the one with minimum
    cost (= maximum query-conditioned weight), matching what Yen's would have
    chosen on the full graph.

    If `mapping` is None, `path_local` is already in full-graph ids (used by
    the full-graph fallback path).
    """
    ids = path_local if mapping is None else [mapping[v] for v in path_local]
    triples: list[tuple[str, str, str]] = []
    for i in range(len(ids) - 1):
        u, v = ids[i], ids[i + 1]
        eid = _best_edge_full_graph(graph, u, v)
        rel = graph.es[eid]["relation"] if eid >= 0 else "?"
        triples.append((graph.vs[u]["name"], rel, graph.vs[v]["name"]))
    return ids, triples


# ----------------------------------------------------------------------------
# Fallback: full-graph BFS shortest path
# ----------------------------------------------------------------------------


def fallback_full_graph_path(
    graph,
    topic_id: int,
    answer_id: int,
    k: int = 5,
    max_hops: int = 5,
    eps: float = 1e-8,
):
    """When the corridor is disconnected even at K=max(ladder), run Yen's on
    the full weighted graph. Slower but guaranteed to find a path if one
    exists in the KG."""
    costs = [1.0 / (float(w) + eps) for w in graph.es["weight"]]
    graph.es["cost"] = costs
    paths = graph.get_k_shortest_paths(
        v=topic_id, to=answer_id, k=k, weights="cost", mode="all"
    )
    result: list[tuple[list[int], float]] = []
    for path in paths:
        if len(path) - 1 > max_hops:
            continue
        total = 0.0
        for i in range(len(path) - 1):
            eid = graph.get_eid(path[i], path[i + 1], error=False)
            total += costs[eid]
        result.append((path, total))
    return result
