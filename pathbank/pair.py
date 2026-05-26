"""Per-pair pipeline: PPR + corridor + k-shortest path → per-topic JSONL block."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .corridor import (
    build_corridor,
    fallback_full_graph_path,
    translate_path,
    yen_k_shortest,
    yen_k_shortest_by_hop,
)
from .pair_utils import PairSpec
from .ppr_engine import run_ppr


def build_pair_pathbank(
    pair: PairSpec,
    graph,                          # igraph with query-conditioned weights applied
    cfg,                            # argparse.Namespace
) -> dict[str, Any]:
    """Return the per-topic dict for a single (topic, answer) pair."""
    # --- PPR ---
    score, pi_t_full, pi_a_full = run_ppr(
        graph,
        topic_id=pair.topic_id,
        answer_id=pair.answer_id,
        mode=cfg.ppr_mode,
        damping=cfg.damping,
        seed_weighting=cfg.seed_weighting,
        bidir_combine=getattr(cfg, "bidir_combine", "product"),
        return_components=True,
    )

    # Top-K node ids by PPR score (for corridor_recall@K sanity metric).
    # K = max of ladder so we can compute recall at all ladder points offline.
    ladder_max = max(int(x) for x in str(cfg.corridor_ladder).split(","))
    top_n = min(ladder_max, len(score))
    ppr_top_ids = np.argpartition(-score, top_n - 1)[:top_n]
    ppr_top_ids = ppr_top_ids[np.argsort(-score[ppr_top_ids])].tolist()

    # --- Corridor + fallback ---
    ladder = tuple(int(x) for x in str(cfg.corridor_ladder).split(","))
    sub, mapping, k_used, fallback, fb_reason = build_corridor(
        graph, score, pair.topic_id, pair.answer_id, ladder=ladder
    )

    k_per_hop = int(getattr(cfg, "n_paths_per_hop", 0) or 0)

    yen_cost_mode = getattr(cfg, "yen_cost_mode", "edge_weight")
    yen_ppr_alpha = float(getattr(cfg, "yen_ppr_alpha", 0.5))

    if fallback:
        raw = fallback_full_graph_path(
            graph,
            topic_id=pair.topic_id,
            answer_id=pair.answer_id,
            k=cfg.n_paths,
            max_hops=cfg.max_hops,
        )
        translated = [translate_path(p, None, graph) for p, _ in raw]
        costs = [c for _, c in raw]
        raw_by_hop = {}
        inv_map = None
    else:
        inv = {v: i for i, v in mapping.items()}
        inv_map = inv

        # Subset score to corridor for PPR-aware cost (if any).
        if yen_cost_mode != "edge_weight":
            # Use 'score' (the combined corridor score) restricted to subgraph nodes.
            sub_node_ids = [mapping[i] for i in range(sub.vcount())]
            ppr_local = score[np.asarray(sub_node_ids, dtype=np.int64)]
        else:
            ppr_local = None

        raw = yen_k_shortest(
            sub,
            topic_local=inv[pair.topic_id],
            answer_local=inv[pair.answer_id],
            k=cfg.n_paths,
            max_hops=cfg.max_hops,
            cost_mode=yen_cost_mode,
            ppr_score_local=ppr_local,
            ppr_alpha=yen_ppr_alpha,
        )
        translated = [translate_path(p, mapping, graph) for p, _ in raw]
        costs = [c for _, c in raw]

        raw_by_hop = {}
        if k_per_hop > 0:
            dedup_mode = getattr(cfg, "path_dedup", "none")
            # Larger raw pool when dedup active (need candidates to fill after dedup)
            raw_k_eff = int(getattr(cfg, "yen_raw_k", 0)) or (
                500 if dedup_mode != "none" else max(100, cfg.n_paths * k_per_hop * 4)
            )
            raw_by_hop = yen_k_shortest_by_hop(
                sub,
                topic_local=inv[pair.topic_id],
                answer_local=inv[pair.answer_id],
                k_per_hop=k_per_hop,
                max_hops=cfg.max_hops,
                raw_k=raw_k_eff,
                cost_mode=yen_cost_mode,
                ppr_score_local=ppr_local,
                ppr_alpha=yen_ppr_alpha,
                dedup=dedup_mode,
            )

    path_node_ids = [ids for ids, _ in translated]
    triples = [tri for _, tri in translated]
    path_hops = [len(ids) - 1 for ids in path_node_ids]

    # Extract per-path type/relation sequences (for type-level match metric).
    def _path_types(ids_list: list[int]) -> list[str]:
        return [graph.vs[int(n)]["type"] for n in ids_list]

    path_types = [_path_types(ids) for ids in path_node_ids]
    path_relations = [[t[1] for t in tri] for tri in triples]

    # Hop-stratified output (optional)
    paths_by_hop: dict[str, list] = {}
    path_node_ids_by_hop: dict[str, list] = {}
    scores_by_hop: dict[str, list] = {}
    path_types_by_hop: dict[str, list] = {}
    path_relations_by_hop: dict[str, list] = {}

    if raw_by_hop:
        for h, items in raw_by_hop.items():
            if not items:
                continue
            h_key = str(h)
            translated_h = [translate_path(p, mapping, graph) for p, _ in items]
            paths_by_hop[h_key]         = [tri for _, tri in translated_h]
            path_node_ids_by_hop[h_key] = [ids for ids, _ in translated_h]
            scores_by_hop[h_key]        = [c for _, c in items]
            path_types_by_hop[h_key]    = [_path_types(ids) for ids in path_node_ids_by_hop[h_key]]
            path_relations_by_hop[h_key] = [[t[1] for t in tri] for tri in paths_by_hop[h_key]]

    return {
        "topic_name": pair.topic_name,
        "topic_type": pair.topic_type,
        "gt_subpath": pair.gt_subpath,
        "gt_relations": pair.gt_relations,
        "gt_types": pair.gt_types,
        "role": pair.role,
        "paths": triples,
        "path_node_ids": path_node_ids,
        "path_types": path_types,
        "path_relations": path_relations,
        "scores": costs,
        "path_hops": path_hops,
        "corridor_k_used": k_used,
        "fallback": fallback,
        "fallback_reason": fb_reason,
        "ppr_top_ids": ppr_top_ids,
        "paths_by_hop": paths_by_hop,
        "path_node_ids_by_hop": path_node_ids_by_hop,
        "scores_by_hop": scores_by_hop,
        "path_types_by_hop": path_types_by_hop,
        "path_relations_by_hop": path_relations_by_hop,
    }
