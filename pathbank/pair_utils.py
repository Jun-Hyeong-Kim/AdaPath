"""Pair decomposition from dataset records.

2-topic records have asymmetric structure (one chain topic + one direct-neighbor topic):
  path_node_ids = [t_chain, m_1, ..., m_{h-1}, answer, t_direct]
where h = num_hops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PairSpec:
    topic_id: int
    answer_id: int
    gt_subpath: List[int]          # node id sequence from topic to answer
    gt_relations: List[str]        # relations along gt_subpath consecutive pairs
    gt_types: List[str]            # node types along gt_subpath
    role: str                      # "chain" | "direct"
    topic_name: str
    topic_type: str


def _extract_gt_relations(
    gt_subpath: List[int],
    triplets: list,
    id_to_name: dict,
) -> List[str]:
    """For each consecutive (u, v) in gt_subpath, find matching relation from triplets.

    triplets are [[src_name, rel, dst_name], ...] (undirected match OK).
    Falls back to '?' if no match found (should not happen for well-formed data).
    """
    rels: List[str] = []
    for i in range(len(gt_subpath) - 1):
        u_name = id_to_name.get(gt_subpath[i], "")
        v_name = id_to_name.get(gt_subpath[i + 1], "")
        rel = "?"
        for t in triplets:
            if len(t) < 3:
                continue
            s, r, d = t[0], t[1], t[2]
            if (s == u_name and d == v_name) or (s == v_name and d == u_name):
                rel = r
                break
        rels.append(rel)
    return rels


def decompose_pairs(record: dict, graph=None) -> List[PairSpec]:
    """Decompose a record into (topic, answer) pairs.

    1-topic: single PairSpec with role="chain".
    2-topic: two PairSpecs — chain topic (len = num_hops+1) and direct topic (len = 2).

    If `graph` is provided (igraph), fill gt_relations/gt_types by combining
    record.triplets (for relations) + graph.vs[...]['type'] (for node types).
    """
    pnids = record["path_node_ids"]
    h = record["num_hops"]
    a_id = pnids[h]
    topics = record["topic_entities"]

    # node_metadata is a list of {id, name, type, ...} dicts
    raw_md = record.get("node_metadata") or []
    meta_by_id: dict[int, dict] = {}
    if isinstance(raw_md, list):
        meta_by_id = {int(m["id"]): m for m in raw_md if isinstance(m, dict) and "id" in m}
    elif isinstance(raw_md, dict):
        meta_by_id = {int(k): v for k, v in raw_md.items() if isinstance(v, dict)}

    def _name(nid):
        if graph is not None:
            try:
                return graph.vs[int(nid)]["name"]
            except Exception:
                pass
        return topics.get(str(nid)) or meta_by_id.get(int(nid), {}).get("name") or ""

    def _type(nid):
        if graph is not None:
            try:
                return graph.vs[int(nid)]["type"]
            except Exception:
                pass
        return meta_by_id.get(int(nid), {}).get("type") or ""

    # Build id→name map for all path nodes for relation extraction.
    id_to_name = {nid: _name(nid) for nid in pnids}

    triplets = record.get("triplets") or []

    def _build_pair(topic_id: int, gt_sub: list) -> PairSpec:
        gt_rels = _extract_gt_relations(gt_sub, triplets, id_to_name)
        gt_tys = [_type(nid) for nid in gt_sub]
        return PairSpec(
            topic_id=topic_id,
            answer_id=a_id,
            gt_subpath=list(gt_sub),
            gt_relations=gt_rels,
            gt_types=gt_tys,
            role="chain" if gt_sub[0] == pnids[0] else "direct",
            topic_name=_name(topic_id),
            topic_type=_type(topic_id),
        )

    t_chain = pnids[0]
    pairs = [_build_pair(t_chain, pnids[: h + 1])]

    if len(topics) == 2:
        t_direct = pnids[-1]
        pairs.append(_build_pair(t_direct, [t_direct, a_id]))

    return pairs
