"""State containers shared by the AdaPath inference loop
(TopicHopState / QuestionState and small load/save helpers)."""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from stark_qa import load_skb

from inference.primekg_client import PrimeKGClient
from inference.primekg_func import (
    relation_search_prune,
    entity_search,
    entity_score,
    update_history,
    entity_prune,
    reasoning,
    generate_answer,
)
from inference.utils import save_2_jsonl, generate_without_explored_paths


# ------------------------------------------------------------------ #
#  Variant config                                                     #


# ------------------------------------------------------------------ #
#  State                                                              #
# ------------------------------------------------------------------ #

@dataclass
class TopicHopState:
    """Per-original-topic-entity state for dynamic subquery tracking."""
    hops: list          # list of hop dicts from subquery data
    current_hop: int = 0
    total_hops: int = 0
    fallback_to_query: bool = False  # f2: switched to query-only


@dataclass
class QuestionState:
    """Track state for one question through the ToG pipeline."""
    idx: int
    record: dict
    query: str
    topic_entities: dict
    answer_ids: list
    cluster_chain_of_entities: list = field(default_factory=list)
    pre_relations: list = field(default_factory=list)
    pre_heads: list = field(default_factory=list)
    explored_entities: dict = field(default_factory=dict)
    current_depth: int = 0
    finished: bool = False
    result_record: Optional[dict] = None
    # Dynamic subquery state
    topic_state: dict = field(default_factory=dict)      # {origin_topic_id: TopicHopState}
    entity_to_origin: dict = field(default_factory=dict)  # {current_entity_id: origin_topic_id}
    # Intermediate logging (--log_intermediate)
    depth_logs: list = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Effective query (original + dynamic subquery)                                             #
# ------------------------------------------------------------------ #

def get_effective_query(state, entity_idx, variant_cfg):
    """
    Determine the effective query for a given entity at the current depth.
    Returns (effective_query, target_type_or_None).
    """
    use_q_plus_s, _, _, _ = variant_cfg
    origin = state.entity_to_origin.get(entity_idx, entity_idx)
    ts = state.topic_state.get(origin)

    if ts and ts.current_hop < ts.total_hops and not ts.fallback_to_query:
        hop_info = ts.hops[ts.current_hop]
        subquery = hop_info["subquery"]
        target_type = hop_info["target"]["type"]

        if use_q_plus_s:
            effective_query = f"{state.query}\nSub-question for this step: {subquery}"
        else:
            effective_query = subquery
        return effective_query, target_type
    else:
        return state.query, None


# ------------------------------------------------------------------ #
#  Core depth processing                                              #
# ------------------------------------------------------------------ #

def process_single_question_depth(state, args, client, variant_cfg):
    """Process one depth step for a single question."""
    query = state.query
    _, use_early_stop, use_type_filter, fallback_mode = variant_cfg
    _logging = getattr(args, 'log_intermediate', False)
    _depth_log = {"depth": state.current_depth} if _logging else None

    # Phase 1: Relation search & prune (per topic entity with effective query)
    current_entity_relations_list = []
    entity_relation_origins = []  # track which origin each relation group came from
    if _logging:
        _depth_log["topic_entities"] = {int(k): v for k, v in state.topic_entities.items()}
        _depth_log["relation_pruning"] = []

    for entity_idx in state.topic_entities:
        effective_query, _ = get_effective_query(state, entity_idx, variant_cfg)
        origin = state.entity_to_origin.get(entity_idx, entity_idx)

        retrieve_relations = relation_search_prune(
            entity_idx, state.topic_entities[entity_idx],
            state.pre_relations, effective_query, args, client
        )
        current_entity_relations_list.extend(retrieve_relations)
        entity_relation_origins.extend([origin] * len(retrieve_relations))

        if _logging:
            _depth_log["relation_pruning"].append({
                "topic_entity_id": int(entity_idx),
                "topic_entity_name": state.topic_entities[entity_idx],
                "effective_query": effective_query[:300],
                "selected_relations": [
                    {"relation": r["relation"], "score": float(r["score"])}
                    for r in retrieve_relations if r["entity"] == entity_idx
                ],
            })

    # Phase 2: Entity search & score
    total_candidates = []
    total_scores = []
    total_relations = []
    total_entities_id = []
    total_topic_entities = []
    total_head = []
    if _logging:
        _depth_log["entity_search_score"] = []

    for rel_idx, entity in enumerate(current_entity_relations_list):
        origin = entity_relation_origins[rel_idx]
        source_entity_idx = entity['entity']

        # Get effective query and target type for this entity's origin
        effective_query, target_type = get_effective_query(state, source_entity_idx, variant_cfg)

        entity_candidates_id, entity_candidates_name = entity_search(
            entity['entity'], entity['relation'], client
        )
        if len(entity_candidates_name) == 0:
            continue

        _pre_filter_count = len(entity_candidates_id)

        # Type filtering (V5-V8)
        if use_type_filter and target_type:
            filtered_pairs = [
                (eid, name) for eid, name in zip(entity_candidates_id, entity_candidates_name)
                if client.idx_to_type(eid) == target_type
            ]
            if filtered_pairs:
                entity_candidates_id, entity_candidates_name = zip(*filtered_pairs)
                entity_candidates_id = list(entity_candidates_id)
                entity_candidates_name = list(entity_candidates_name)
            else:
                # Fallback
                if fallback_mode == "f2":
                    state.topic_state[origin].fallback_to_query = True
                    # Re-compute effective query (now falls back to original query)
                    effective_query, _ = get_effective_query(state, source_entity_idx, variant_cfg)
                # f1: keep all candidates and keep subquery

        _pre_sbert_count = len(entity_candidates_id)
        _sbert_triggered = _pre_sbert_count >= 20
        _pre_sbert_ids = list(entity_candidates_id) if _logging and _sbert_triggered else None
        _pre_sbert_names = list(entity_candidates_name) if _logging and _sbert_triggered else None

        # Downsampling — use entity descriptions for BM25/SBERT similarity
        if len(entity_candidates_id) >= 20:
            if args.entity_sampling == "bm25":
                from inference.utils import compute_bm25_similarity
                candidate_docs = [client.get_doc_info(eid) for eid in entity_candidates_id]
                top_docs, _ = compute_bm25_similarity(
                    effective_query, candidate_docs, args.max_entity_candidates)
                indices = [candidate_docs.index(d) for d in top_docs
                           if d in candidate_docs]
            elif args.entity_sampling == "sbert":
                from inference.utils import retrieve_top_docs
                candidate_docs = [client.get_doc_info(eid) for eid in entity_candidates_id]
                top_docs, _ = retrieve_top_docs(
                    effective_query, candidate_docs, args.sbert_model, args.max_entity_candidates)
                indices = [candidate_docs.index(d) for d in top_docs
                           if d in candidate_docs]
            else:
                indices = random.sample(range(len(entity_candidates_name)),
                                        min(args.max_entity_candidates, len(entity_candidates_name)))
            entity_candidates_id = [entity_candidates_id[i] for i in indices]
            entity_candidates_name = [entity_candidates_name[i] for i in indices]

        if len(entity_candidates_id) == 0:
            continue

        scores, entity_candidates_name, entity_candidates_id = entity_score(
            effective_query, entity_candidates_id, entity_candidates_name,
            entity['score'], entity['relation'], args
        )

        if _logging:
            _log_entry = {
                "parent_entity_id": int(source_entity_idx),
                "parent_entity_name": client.idx_to_name(source_entity_idx),
                "relation": entity['relation'],
                "relation_score": float(entity['score']),
                "neighbors_total": _pre_filter_count,
                "after_type_filter": _pre_sbert_count,
                "sbert_triggered": _sbert_triggered,
                "after_sbert_filter": len(entity_candidates_id),
                "scored_candidates": [
                    {"id": int(eid), "name": str(ename), "final_score": float(sc)}
                    for eid, ename, sc in zip(entity_candidates_id, entity_candidates_name, scores)
                ],
            }
            if _sbert_triggered and _pre_sbert_ids is not None:
                _log_entry["pre_sbert_ids"] = [int(x) for x in _pre_sbert_ids]
                _log_entry["pre_sbert_names"] = [str(x) for x in _pre_sbert_names]
            _depth_log["entity_search_score"].append(_log_entry)

        (total_candidates, total_scores, total_relations,
         total_entities_id, total_topic_entities, total_head) = update_history(
            entity_candidates_name, entity, scores, entity_candidates_id,
            total_candidates, total_scores, total_relations,
            total_entities_id, total_topic_entities, total_head
        )

    # Track explored
    for eid, sc in zip(total_entities_id, total_scores):
        if eid not in state.explored_entities or sc > state.explored_entities[eid]:
            state.explored_entities[eid] = sc

    # Phase 3: Check completion
    record = state.record

    if len(total_candidates) == 0:
        if _logging and _depth_log:
            _depth_log["prune_input"] = []
            _depth_log["prune_output_top_k"] = []
            state.depth_logs.append(_depth_log)
        answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
        state.result_record = _make_result(record, query, answer_text,
                                            state.cluster_chain_of_entities,
                                            state.explored_entities, args,
                                            depth_logs=state.depth_logs if getattr(args, 'log_intermediate', False) else None)
        state.finished = True
        return state

    if _logging:
        _prune_input = sorted(
            zip(total_entities_id, total_candidates, total_scores,
                total_relations, total_topic_entities),
            key=lambda x: -x[2]
        )
        _depth_log["prune_input"] = [
            {"id": int(eid), "name": str(ename), "score": float(sc),
             "relation": rel, "parent_entity_id": int(pid)}
            for eid, ename, sc, rel, pid in _prune_input
        ]

    flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(
        total_entities_id, total_relations, total_candidates,
        total_topic_entities, total_head, total_scores, args, client
    )
    state.cluster_chain_of_entities.append(chain_of_entities)

    if _logging:
        _depth_log["prune_output_top_k"] = [
            {"id": int(eid), "name": client.idx_to_name(eid)}
            for eid in entities_id
        ] if flag else []
        state.depth_logs.append(_depth_log)

    if flag:
        # Update entity_to_origin mapping and advance hop counters
        _advance_hop_counters(state, entities_id, total_entities_id, total_topic_entities)

        # Early stop check
        if use_early_stop and _all_subqueries_exhausted(state):
            answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
            state.result_record = _make_result(record, query, answer_text,
                                                state.cluster_chain_of_entities,
                                                state.explored_entities, args,
                                                depth_logs=state.depth_logs if getattr(args, 'log_intermediate', False) else None)
            state.finished = True
            return state

        stop, reasoning_response = reasoning(query, state.cluster_chain_of_entities, args)
        if stop:
            answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
            state.result_record = _make_result(record, query, answer_text,
                                                state.cluster_chain_of_entities,
                                                state.explored_entities, args,
                                                depth_logs=state.depth_logs if getattr(args, 'log_intermediate', False) else None)
            state.finished = True
            return state
        else:
            state.topic_entities = {
                eid: client.idx_to_name(eid) for eid in entities_id
            }
            state.pre_relations = pre_relations
            state.pre_heads = pre_heads
    else:
        answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
        state.result_record = _make_result(record, query, answer_text,
                                            state.cluster_chain_of_entities,
                                            state.explored_entities, args,
                                            depth_logs=state.depth_logs if getattr(args, 'log_intermediate', False) else None)
        state.finished = True

    return state


def _advance_hop_counters(state, pruned_entity_ids, total_entities_id, total_topic_entities):
    """
    After entity_prune, update entity_to_origin mapping and advance hop counters.
    total_topic_entities[i] = the source entity that produced total_entities_id[i].
    """
    # Build candidate -> source mapping
    candidate_to_source = {}
    for cand_id, src_id in zip(total_entities_id, total_topic_entities):
        if cand_id not in candidate_to_source:
            candidate_to_source[cand_id] = src_id

    new_entity_to_origin = {}
    advanced_origins = set()
    for eid in pruned_entity_ids:
        src = candidate_to_source.get(eid, eid)
        origin = state.entity_to_origin.get(src, src)
        new_entity_to_origin[eid] = origin

        # Advance hop counter once per origin
        if origin not in advanced_origins and origin in state.topic_state:
            ts = state.topic_state[origin]
            if ts.current_hop < ts.total_hops:
                ts.current_hop += 1
            advanced_origins.add(origin)

    state.entity_to_origin = new_entity_to_origin


def _make_result(record, query, answer_text, chains, explored, args, depth_logs=None):
    result = {
        "path_node_ids": record["path_node_ids"],
        "template_id": record["template_id"],
        "query_type": record.get("query_type", ""),
        "question": query,
        "answer_ids": [record["answer_entity"]["id"]],
        "results": answer_text,
        "reasoning_chains": chains,
        "explored_entities": {str(k): v for k, v in explored.items()},
        "variant": getattr(args, "variant", None),
    }
    if hasattr(record, 'data_id'):
        result["data_id"] = record["data_id"]
    elif "data_id" in record:
        result["data_id"] = record["data_id"]
    if depth_logs:
        result["depth_logs"] = depth_logs
    return result


# ------------------------------------------------------------------ #
#  Subquery loading                                                   #
def init_topic_state(record, subquery_data):
    """Initialize per-topic state from subquery data."""
    topic_state = {}
    entity_to_origin = {}

    if subquery_data and "topic_subqueries" in subquery_data:
        for topic_id_str, info in subquery_data["topic_subqueries"].items():
            tid = int(topic_id_str)
            topic_state[tid] = TopicHopState(
                hops=info["hops"],
                total_hops=len(info["hops"]),
            )
            entity_to_origin[tid] = tid
    else:
        # No subquery data: all topics map to themselves
        for tid_str in record.get("topic_entities", {}):
            tid = int(tid_str)
            entity_to_origin[tid] = tid

    return topic_state, entity_to_origin


# ------------------------------------------------------------------ #
#  Data loading                                                       #
# ------------------------------------------------------------------ #

def load_biokgqa(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------------ #
#  Main                                                               #
