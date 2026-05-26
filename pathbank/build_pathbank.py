"""Build the per-query path bank.

For each record in a BioStrat-QA-style jsonl (must contain `data_id`,
`topic_entities`, and `answer_entity` / `answer_ids`):

  1. Apply query-conditioned edge weights (BM25 over node names + relation
     descriptions; β-mixed between relation and node similarities).
  2. Run bidirectional PPR seeded at (topic, answer).
  3. Extract a corridor subgraph (PPR top-N with ladder fallback).
  4. Yen's K-shortest-paths on the corridor with cost
       cost(edge) = 1 / (α · edge_weight + (1-α) · ppr_edge_score)
     keeping K=500 raw paths, bucketing by hop length, and saving the top-5
     per (topic, answer) and per hop (1..max_hops).

Output: jsonl with per-record `per_topic` dict matching the format consumed
by inference.run_inference (path patterns + node-id sequences per hop).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from .edge_weight import (
    apply_weights_to_graph,
    compute_edge_weights,
    extract_edge_arrays,
)
from .embedding_cache import load_graph
from .pair import build_pair_pathbank
from .pair_utils import PairSpec
from .similarity import load_backbone


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = _ROOT / "data" / "pathbank"


def _primary_answer_id(rec: dict):
    a = rec.get("answer_entity") or {}
    if isinstance(a, dict) and a.get("id") is not None:
        return int(a["id"])
    ans_ids = rec.get("answer_ids") or []
    if ans_ids:
        return int(ans_ids[0])
    return None


def _decompose_record(rec: dict, graph) -> List[PairSpec]:
    """Per (topic, primary_answer) PairSpec for one record."""
    pairs: List[PairSpec] = []
    primary_ans = _primary_answer_id(rec)
    if primary_ans is None or primary_ans < 0 or primary_ans >= graph.vcount():
        return pairs
    a_type = graph.vs[primary_ans]["type"]

    topics = rec.get("topic_entities") or {}
    for tid_str, tname in topics.items():
        try:
            tid = int(tid_str)
        except (ValueError, TypeError):
            continue
        if tid < 0 or tid >= graph.vcount():
            continue
        t_type = graph.vs[tid]["type"]
        t_name_g = graph.vs[tid]["name"]
        pairs.append(PairSpec(
            topic_id=tid,
            answer_id=primary_ans,
            gt_subpath=[tid, primary_ans],
            gt_relations=[],
            gt_types=[t_type, a_type],
            role="chain",
            topic_name=tname or t_name_g,
            topic_type=t_type,
        ))
    return pairs


def _process_record(record: dict, graph, edge_arrays, backbone, cfg) -> dict:
    query = record["query"] if "query" in record else record.get("explicit_query") or ""
    ns = backbone.node_sim(query)
    rs = backbone.rel_sim(query)
    es, ds, eel = edge_arrays
    w = compute_edge_weights(es, ds, eel, ns, rs, beta=cfg.beta)
    apply_weights_to_graph(graph, w)

    pairs = _decompose_record(record, graph)
    per_topic = {}
    for pair in pairs:
        info = build_pair_pathbank(pair, graph, cfg)
        per_topic[str(pair.topic_id)] = info

    return {
        "data_id": record.get("data_id"),
        "query": query,
        "topic_entities": record.get("topic_entities", {}),
        "answer_entity": record.get("answer_entity"),
        "answer_ids": record.get("answer_ids", []),
        "primary_answer_id": _primary_answer_id(record),
        "per_topic": per_topic,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Build per-query path bank.")
    p.add_argument("--input", required=True,
                   help="Input jsonl (BioStrat-QA train/dev/test or any "
                        "compatible records).")
    p.add_argument("--output", required=True,
                   help="Output jsonl path.")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--ppr_mode", default="bidirectional",
                   choices=["bidirectional", "multi_seed"])
    p.add_argument("--bidir_combine", default="product",
                   choices=["product", "rank_product"])
    p.add_argument("--seed_weighting", default="uniform",
                   choices=["uniform", "degree_inv", "degree_proportional"])
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--damping", type=float, default=0.15)
    p.add_argument("--corridor_ladder", default="150,300,500,1000")
    p.add_argument("--n_paths", type=int, default=5)
    p.add_argument("--n_paths_per_hop", type=int, default=5)
    p.add_argument("--max_hops", type=int, default=5)
    p.add_argument("--yen_cost_mode", default="ppr_mix",
                   choices=["edge_weight", "ppr_mix", "ppr_only"])
    p.add_argument("--yen_ppr_alpha", type=float, default=0.5)
    p.add_argument("--path_dedup", default="type_sig",
                   choices=["none", "type_sig", "first_type"])
    p.add_argument("--yen_raw_k", type=int, default=500)
    return p.parse_args()


def main():
    cfg = parse_args()

    # Lazy KG + cache prep
    from .prepare_kg import ensure_kg_ready
    ensure_kg_ready()

    input_path = Path(cfg.input)
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {input_path}")
    with open(input_path) as f:
        records = [json.loads(l) for l in f]
    if cfg.limit:
        records = records[:cfg.limit]
    print(f"  {len(records)} records")

    print("Loading graph + BM25 backbone...")
    t0 = time.time()
    graph = load_graph()
    edge_arrays = extract_edge_arrays(graph)
    backbone = load_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\n=== writing {output_path} ===")
    t_start = time.time()
    n_written = n_skipped = 0
    with open(output_path, "w") as fo:
        for i, rec in enumerate(records):
            out = _process_record(rec, graph, edge_arrays, backbone, cfg)
            if not out["per_topic"]:
                n_skipped += 1
            fo.write(json.dumps(out) + "\n")
            n_written += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(records) - i - 1)
                print(f"  [{i+1}/{len(records)}] elapsed={elapsed:.0f}s "
                      f"eta={eta:.0f}s skipped={n_skipped}")
    print(f"\nDone. wrote {n_written} records ({n_skipped} empty) "
          f"in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
