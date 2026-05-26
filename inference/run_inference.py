"""AdaPath inference: LLM-guided path-finding over a biomedical knowledge graph.

At each hop the LLM is given the question, the current node (name + type),
and the hop index, and it produces a dynamic sub-question that is used to
prune relations and rank candidate entities. Final answers are produced
once the path-finder reaches the target hop count.
"""

import argparse
import json
import os
import random
import re
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from stark_qa import load_skb

# State containers + helpers
from inference._tog_state import (
    TopicHopState,
    QuestionState,
    _advance_hop_counters,
    _make_result,
    load_biokgqa,
)
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
from inference.utils import save_2_jsonl, generate_without_explored_paths, run_llm
from inference.prompts import (
    dynamic_subquery_prompt_with_type,
    dynamic_subquery_prompt_no_type,
)
import pickle

# ------------------------------------------------------------------ #
#  Inference-time pathbank matching
# ------------------------------------------------------------------ #

PRIMEKG_NODE_TYPES = [
    "drug", "gene/protein", "disease", "effect/phenotype", "anatomy",
    "biological_process", "cellular_component", "exposure", "molecular_function", "pathway"
]

ANSWER_TYPE_PROMPT = """Given a biomedical question, predict the type of the answer entity.

Available entity types in the knowledge graph:
- drug: medications, compounds, therapeutics
- gene/protein: genes, proteins, enzymes, receptors
- disease: diseases, disorders, syndromes, conditions
- effect/phenotype: symptoms, side effects, phenotypic traits
- anatomy: organs, tissues, body parts, cell types
- biological_process: biological processes (e.g., apoptosis, cell division)
- cellular_component: cell structures (e.g., mitochondria, nucleus)
- exposure: environmental exposures (e.g., chemicals, radiation)
- molecular_function: molecular functions (e.g., binding, catalysis)
- pathway: biological pathways (e.g., signaling, metabolic)

Q: Which drugs are indicated for the treatment of Alzheimer's disease?
Answer type: drug

Q: What symptoms are associated with long-term use of Metformin?
Answer type: effect/phenotype

Q: {}
Answer type:"""


def _predict_answer_type(query, args):
    """Use LLM to predict the answer entity type. Returns (predicted_type, raw_response)."""
    prompt = ANSWER_TYPE_PROMPT.format(query)
    resp = run_llm(prompt, getattr(args, 'temperature_select', 0.0), 256, args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
    resp_lower = resp.strip().lower()
    for t in PRIMEKG_NODE_TYPES:
        if t in resp_lower:
            return t, resp
    return None, resp  # parse failure


class InferenceTimePathbankMatcher:
    """Matches test queries to train pathbank paths at inference time using
    topic type + answer type filtering + hybrid similarity."""

    def __init__(self, train_records, train_pathbank_by_id, qtype, sbert_device="cpu"):
        """
        Args:
            train_records: list of train record dicts (from generated_gpt54/train.jsonl)
            train_pathbank_by_id: dict data_id -> {hop: [path_dicts]} from train pathbank
            qtype: 'explicit' / 'implicit' / 'bare'
            sbert_device: device for SBERT encoding
        """
        self.train_records = train_records
        self.train_pathbank = train_pathbank_by_id
        self.qtype = qtype
        self._query_field = {"explicit": "explicit_query", "implicit": "implicit_query", "bare": "bare_query"}.get(qtype, "query")

        # Pre-compute train types
        self._train_topic_types = []
        self._train_answer_types = []
        self._train_queries = []
        self._train_data_ids = []
        self._train_hops = []
        for r in train_records:
            # topic type
            pnids = r["path_node_ids"]
            topic_type = ""
            for md in r.get("node_metadata") or []:
                if isinstance(md, dict) and md.get("id") == pnids[0]:
                    topic_type = md.get("type", "")
                    break
            ans_type = r["answer_entity"].get("type", "")
            self._train_topic_types.append(topic_type)
            self._train_answer_types.append(ans_type)
            self._train_queries.append(r[self._query_field])
            self._train_data_ids.append(r["data_id"])
            self._train_hops.append(r.get("num_hops", len(r.get("triplets", []))))

        # Pre-compute SBERT embeddings + BM25 index for train
        import re as _re
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b", device=sbert_device)
        with torch.no_grad():
            self._train_embs = _sbert.encode(
                self._train_queries, batch_size=256, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False)
        self._sbert = _sbert

        def _tokenize(text):
            text = _re.sub(r"[^\w\s]", " ", text.lower())
            return [t for t in text.split() if t]

        self._tokenize = _tokenize
        train_tokens = [_tokenize(q) for q in self._train_queries]
        self._bm25 = BM25Okapi(train_tokens, b=0.0)

    def match(self, test_query, topic_type, answer_type, hops_to_use=None):
        """Find best matching train record and return its pathbank paths.

        Args:
            test_query: test query string
            topic_type: test topic entity type (always known)
            answer_type: predicted or GT answer type (None = skip answer type filter)
            hops_to_use: list of hop ints to include paths from, or None for all 1-3

        Returns:
            (paths: list of path dicts, match_info: dict with matching details)
        """
        if hops_to_use is None:
            hops_to_use = [1, 2, 3]

        # Build candidate mask
        candidates = []
        for i in range(len(self.train_records)):
            if self._train_topic_types[i] != topic_type:
                continue
            if answer_type and self._train_answer_types[i] != answer_type:
                continue
            candidates.append(i)

        if not candidates:
            # Fallback: topic type only
            candidates = [i for i in range(len(self.train_records))
                          if self._train_topic_types[i] == topic_type]

        if not candidates:
            return [], {"matched": False, "reason": "no_candidates"}

        # Compute similarity for this test query against candidates
        with torch.no_grad():
            test_emb = self._sbert.encode([test_query], convert_to_tensor=True,
                                          normalize_embeddings=True, show_progress_bar=False)
        cand_embs = self._train_embs[candidates]
        sbert_scores = (test_emb @ cand_embs.T).cpu().numpy()[0]

        # BM25
        toks = self._tokenize(test_query)
        if toks:
            all_bm25 = self._bm25.get_scores(toks)
            bm25_scores = np.array([all_bm25[i] for i in candidates], dtype=np.float32)
        else:
            bm25_scores = np.zeros(len(candidates), dtype=np.float32)

        # Min-max normalize
        def _mm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(x)

        hybrid = 0.5 * _mm(sbert_scores) + 0.5 * _mm(bm25_scores)

        # Per-hop pool: find best match per hop, collect paths
        paths_out = []
        match_details = {}
        for hop in hops_to_use:
            hop_cands = [(j, candidates[j]) for j in range(len(candidates))
                         if self._train_hops[candidates[j]] == hop]
            if not hop_cands:
                match_details[f"pool_{hop}h"] = None
                continue
            best_j = max(hop_cands, key=lambda x: hybrid[x[0]])[0]
            best_idx = candidates[best_j]
            best_did = self._train_data_ids[best_idx]
            best_sim = float(hybrid[best_j])

            # Get pathbank paths for this train record at this hop
            pb = self.train_pathbank.get(str(best_did), {}).get(str(hop), [])
            for p in pb:
                paths_out.append({
                    "path_types": p["path_types"],
                    "path_relations": p["path_relations"],
                    "hop": hop,
                })
            match_details[f"pool_{hop}h"] = {
                "train_data_id": best_did,
                "sim_score": best_sim,
                "n_paths": len(pb),
            }

        return paths_out, {"matched": True, "details": match_details}

    def match_top_k(self, test_query, topic_type, answer_type, k=5, hops_to_use=None):
        """Find top-k matching train queries and collect ALL their pathbank paths.

        Unlike match() which picks per-hop best-1, this picks global top-k train
        queries and collects all their paths across all hops.

        Returns:
            (paths_by_hop: {hop_int: [path_dicts]}, match_info: dict)
        """
        if hops_to_use is None:
            hops_to_use = [1, 2, 3]

        # Build candidate mask (topic_type filter only, answer_type applied at path level)
        candidates = [i for i in range(len(self.train_records))
                      if self._train_topic_types[i] == topic_type]
        if not candidates:
            return {}, {"matched": False, "reason": "no_candidates"}

        # Similarity
        with torch.no_grad():
            test_emb = self._sbert.encode([test_query], convert_to_tensor=True,
                                          normalize_embeddings=True, show_progress_bar=False)
        cand_embs = self._train_embs[candidates]
        sbert_scores = (test_emb @ cand_embs.T).cpu().numpy()[0]

        toks = self._tokenize(test_query)
        if toks:
            all_bm25 = self._bm25.get_scores(toks)
            bm25_scores = np.array([all_bm25[i] for i in candidates], dtype=np.float32)
        else:
            bm25_scores = np.zeros(len(candidates), dtype=np.float32)

        def _mm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(x)

        hybrid = 0.5 * _mm(sbert_scores) + 0.5 * _mm(bm25_scores)

        # Global top-k (deduplicate by data_id to get k distinct train queries)
        sorted_idx = np.argsort(-hybrid)
        seen_dids = set()
        top_k_dids = []
        top_k_sims = []
        for j in sorted_idx:
            did = self._train_data_ids[candidates[j]]
            if did not in seen_dids:
                seen_dids.add(did)
                top_k_dids.append(did)
                top_k_sims.append(float(hybrid[j]))
                if len(top_k_dids) >= k:
                    break

        # Collect all pathbank paths from top-k train queries
        paths_by_hop = {h: [] for h in hops_to_use}
        for did in top_k_dids:
            pb_by_hop = self.train_pathbank.get(str(did), {})
            for hop in hops_to_use:
                for p in pb_by_hop.get(str(hop), []):
                    pt = p["path_types"]
                    pr = p["path_relations"]
                    # Answer type filter at path level
                    if answer_type and pt and pt[-1] != answer_type:
                        continue
                    paths_by_hop[hop].append({
                        "path_types": pt,
                        "path_relations": pr,
                        "hop": hop,
                    })

        match_info = {
            "matched": True,
            "top_k_data_ids": top_k_dids,
            "top_k_sims": top_k_sims,
            "paths_per_hop_raw": {h: len(v) for h, v in paths_by_hop.items()},
        }
        return paths_by_hop, match_info


def _dedup_path_sigs(paths_by_hop):
    """Remove duplicate path signatures within each hop pool."""
    result = {}
    for hop, paths in paths_by_hop.items():
        seen = set()
        deduped = []
        for p in paths:
            sig = (tuple(p["path_types"]), tuple(p["path_relations"]))
            if sig not in seen:
                seen.add(sig)
                deduped.append(p)
        result[hop] = deduped
    return result


def _filter_and_cap_paths(paths_by_hop, topic_id, client, caps):
    """Apply traversable filter + cap to each hop pool."""
    result = {}
    for hop, paths in paths_by_hop.items():
        cap = caps.get(hop, 8)
        traversable = [p for p in paths
                       if _check_traversable(topic_id, p["path_types"], p["path_relations"], client)]
        if len(traversable) > cap:
            traversable = random.sample(traversable, cap)
        result[hop] = traversable
    return result


def _dedup_triplets(all_chains):
    """Remove duplicate triplets from merged chains, preserving depth structure."""
    seen = set()
    result = []
    for depth_data in all_chains:
        deduped = []
        for triplet in depth_data:
            key = tuple(str(x) for x in triplet)
            if key not in seen:
                seen.add(key)
                deduped.append(triplet)
        if deduped:
            result.append(deduped)
    return result


_LLM_HOP_SELECT_PROMPT = """Given a biomedical question and candidate path patterns from a knowledge graph, select which hop count (path length) is most appropriate for answering the question.

Question: {query}

Available path pools:
{pool_descriptions}

Which hop count ({hop_options}) is most likely to reach the answer? Consider:
- Simpler questions (direct relationships) → 1-hop
- Questions requiring one intermediate step → 2-hop
- Complex multi-step reasoning → 3-hop

Answer with just the number (e.g. 1, 2, or 3):"""


def _format_path_example(p):
    """Format a path dict as readable string: 'drug →[indication]→ disease'."""
    types, rels = p["path_types"], p["path_relations"]
    parts = []
    for i, r in enumerate(rels):
        parts.append(f"{types[i]} →[{r}]→")
    parts.append(types[-1] if types else "?")
    return " ".join(parts)


def _llm_select_hop(paths_by_hop, query, args):
    """Ask LLM which hop pool to use. Returns int (1/2/3) or None on failure."""
    pool_descs = []
    available_hops = []
    for hop in [1, 2, 3]:
        paths = paths_by_hop.get(hop, [])
        if not paths:
            continue
        available_hops.append(hop)
        examples = random.sample(paths, min(3, len(paths)))
        ex_strs = "\n".join(f"  - {_format_path_example(e)}" for e in examples)
        pool_descs.append(f"## {hop}-hop paths ({len(paths)} unique patterns):\n{ex_strs}")

    if not available_hops:
        return None

    prompt = _LLM_HOP_SELECT_PROMPT.format(
        query=query,
        pool_descriptions="\n\n".join(pool_descs),
        hop_options="/".join(str(h) for h in available_hops),
    )
    resp = run_llm(prompt, getattr(args, 'temperature_select', 0.0), 32, args.opeani_api_keys, args.LLM_type,
                   base_url=getattr(args, 'base_url', None))
    # Parse: find first digit that's a valid hop
    for ch in resp.strip():
        if ch.isdigit() and int(ch) in available_hops:
            return int(ch)
    return None


def _get_desc_best_chunk(entity_id, question, node_info, bm25_tokenize=None):
    """Get BM25 best sentence chunk from entity description. Returns name-only if no description."""
    info = node_info.get(entity_id, {})
    name = info.get('name', str(entity_id))
    details = info.get('details', {})

    if not isinstance(details, dict):
        return name, False

    # Collect description text
    text_parts = []
    for k, v in details.items():
        if isinstance(v, str) and len(v) > 10:
            text_parts.append(v)
    if not text_parts:
        return name, False

    full_text = ' '.join(text_parts)
    sentences = [s.strip() for s in full_text.replace('\n', '. ').split('. ') if len(s.strip()) > 10]
    if not sentences:
        return name, False

    # BM25 scoring
    if bm25_tokenize is not None:
        from rank_bm25 import BM25Okapi
        tokenized = [s.lower().split() for s in sentences]
        bm25 = BM25Okapi(tokenized)
        q_tokens = question.lower().split()
        scores = bm25.get_scores(q_tokens)
        best_idx = scores.argmax()
        return sentences[best_idx], True
    else:
        # Fallback: first sentence
        return sentences[0], True


def _load_train_pathbank_by_id_and_hop(pathbank_dir, qtype):
    """Load train pathbank: {data_id_str: {hop_str: [path_dicts]}}"""
    fpath = os.path.join(pathbank_dir, qtype, "train_gtHopTop5.jsonl")
    result = {}
    with open(fpath) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            did = str(rec["data_id"])
            by_hop = {}
            for tid, tinfo in rec["per_topic"].items():
                if "paths" in tinfo:
                    gt_hop = tinfo.get("gt_hop", 1)
                    hop_key = str(gt_hop)
                    if hop_key not in by_hop:
                        by_hop[hop_key] = []
                    for i in range(len(tinfo["paths"])):
                        by_hop[hop_key].append({
                            "path_types": tinfo["path_types"][i],
                            "path_relations": tinfo["path_relations"][i],
                        })
            result[did] = by_hop
    return result


# ------------------------------------------------------------------ #
#  Entity scoring mode helpers (sbert_top5_llm / sbert_only)         #
# ------------------------------------------------------------------ #

def _get_desc_for_sbert(nid):
    """Get DESC_STRATEGY-based description for SBERT scoring."""
    ni = _get_node_info()
    info = ni.get(int(nid), {})
    details = info.get('details', {})
    if not isinstance(details, dict):
        return info.get('name', str(nid))
    fields = DESC_STRATEGY.get(info.get('type', ''), [])
    parts = []
    total = 0
    for key in fields:
        val = details.get(key, '')
        if val and isinstance(val, str):
            remaining = 3000 - total
            if remaining <= 0:
                break
            parts.append(val[:remaining])
            total += len(val[:remaining])
    desc = ' '.join(parts)
    if not desc:
        desc = info.get('name', str(nid))
    return desc


def _sbert_score_entities(query, entity_ids, sbert_model, top_k=None):
    """Score entities with SBERT using DESC_STRATEGY descriptions.
    Returns (entity_ids, scores) sorted by score descending.
    If top_k given, returns only top_k. Scores are normalized to sum=1.
    """
    from sentence_transformers import util as st_util
    descs = [_get_desc_for_sbert(eid) for eid in entity_ids]
    query_emb = sbert_model.encode(query)
    doc_emb = sbert_model.encode(descs)
    raw_scores = st_util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    pairs = sorted(zip(entity_ids, raw_scores), key=lambda x: x[1], reverse=True)
    if top_k is not None and top_k < len(pairs):
        pairs = pairs[:top_k]

    ids = [p[0] for p in pairs]
    scores = [max(p[1], 0.0) for p in pairs]  # clamp negatives
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    else:
        scores = [1.0 / len(scores)] * len(scores)
    return ids, scores


def _get_entity_desc_for_llm(entity_id, query, mode):
    """Get entity description for LLM scoring prompt.
    mode='chunk': BM25 best 150-char chunk from DESC_STRATEGY text.
    mode='full': full DESC_STRATEGY text (max 3000 chars).
    mode='none': returns empty string.
    """
    if mode == 'none':
        return ''
    desc_text = _get_desc(entity_id)
    if not desc_text:
        return ''
    if mode == 'chunk':
        return _get_best_chunk_bm25(query, desc_text, max_len=150)
    elif mode == 'full':
        return desc_text[:3000]
    return ''


def _format_entities_with_desc(names, ids, query, desc_mode):
    """Format entity list for LLM prompt, optionally with descriptions.
    Returns a string like:
      name_only: "Entity1; Entity2; Entity3"
      with desc:  "Entity1 (description1); Entity2 (description2)"
    """
    if desc_mode == 'none':
        return "; ".join(names)
    parts = []
    for name, eid in zip(names, ids):
        desc = _get_entity_desc_for_llm(eid, query, desc_mode)
        if desc:
            parts.append(f"{name} ({desc})")
        else:
            parts.append(name)
    return "; ".join(parts)


# ------------------------------------------------------------------ #
#  Entity lookup cache (pre-computed relations + types for speedup)   #
# ------------------------------------------------------------------ #

_ENTITY_LOOKUP_CACHE = None
def _get_entity_lookup():
    global _ENTITY_LOOKUP_CACHE
    if _ENTITY_LOOKUP_CACHE is None:
        lookup_path = 'data/kg/prime/processed/entity_lookup.json'
        if os.path.exists(lookup_path):
            with open(lookup_path) as f:
                _ENTITY_LOOKUP_CACHE = json.load(f)
        else:
            _ENTITY_LOOKUP_CACHE = {}
    return _ENTITY_LOOKUP_CACHE

def _fast_get_relations(entity_id):
    """O(1) relation lookup from pre-computed JSON."""
    lookup = _get_entity_lookup()
    rels = lookup.get('entity_relations', {}).get(str(entity_id), None)
    if rels is not None:
        return rels
    return None  # fallback to client.get_all_relations_of_entity()

def _fast_get_type(entity_id):
    """O(1) type lookup from pre-computed JSON."""
    lookup = _get_entity_lookup()
    return lookup.get('entity_types', {}).get(str(entity_id), None)


# ------------------------------------------------------------------ #
#  Experiment helpers (Process 2-5 prompt modifications)              #
# ------------------------------------------------------------------ #

_NODE_INFO_CACHE = None
def _get_node_info():
    global _NODE_INFO_CACHE
    if _NODE_INFO_CACHE is None:
        with open('data/kg/prime/processed/node_info.pkl', 'rb') as f:
            _NODE_INFO_CACHE = pickle.load(f)
    return _NODE_INFO_CACHE

DESC_STRATEGY = {
    'drug': ['description', 'mechanism_of_action', 'indication', 'pharmacodynamics', 'category', 'group'],
    'gene/protein': ['summary', 'name'],
    'disease': ['mondo_definition', 'umls_description', 'orphanet_clinical_description',
                'mayo_symptoms', 'orphanet_definition', 'mondo_name'],
    'pathway': ['summation', 'displayName'],
}

def _get_desc(nid):
    ni = _get_node_info()
    info = ni.get(int(nid), {})
    details = info.get('details', {})
    if not isinstance(details, dict): return ''
    fields = DESC_STRATEGY.get(info.get('type', ''), [])
    parts = []; total = 0
    for key in fields:
        val = details.get(key, '')
        if val and isinstance(val, str):
            remaining = 3000 - total
            if remaining <= 0: break
            parts.append(val[:remaining]); total += len(val[:remaining])
    return ' '.join(parts)

def _get_best_chunk_bm25(query, text, max_len=150):
    if not text: return ''
    from rank_bm25 import BM25Okapi
    sentences = re.split(r'(?<=[.;])\s+', text)
    chunks = []
    for s in sentences:
        s = s.strip()
        if not s: continue
        while len(s) > max_len:
            b = s[:max_len].rfind(', ')
            if b < 50: b = max_len
            chunks.append(s[:b].strip()); s = s[b:].strip(', ')
        if s: chunks.append(s)
    if not chunks: return text[:max_len]
    tokenized = [re.findall(r'[a-z0-9]+', c.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(re.findall(r'[a-z0-9]+', query.lower()))
    return chunks[scores.argmax()]

def _build_type_evidence_path(triplets, node_info, path_node_ids=None):
    """Build type-based evidence path: drug →[indication]→ disease →[phenotype absent]→ effect/phenotype
    If path_node_ids provided, use them for type lookup (avoids name-based disambiguation errors).
    """
    if not triplets: return '', []

    if path_node_ids and len(path_node_ids) >= len(triplets) + 1:
        # Use path_node_ids for accurate type lookup
        types = [node_info.get(path_node_ids[i], {}).get('type', '?') for i in range(len(triplets) + 1)]
    else:
        # Fallback: name-based lookup
        def find_type(name):
            for nid, info in node_info.items():
                if info.get('name', '').lower() == name.lower():
                    return info.get('type', '?')
            return '?'
        types = [find_type(triplets[0][0])]
        for t in triplets:
            types.append(find_type(t[2]))

    rels = []
    for t in triplets:
        rels.append(t[1])

    parts = [types[0]]
    for i, rel in enumerate(rels):
        parts.append(f'→[{rel}]→ {types[i+1]}')
    evidence = ' '.join(parts)

    sub_evidences = []
    for i in range(len(rels)):
        sub_evidences.append(f'{types[i]} →[{rels[i]}]→ {types[i+1]}')

    return evidence, sub_evidences

def _get_desc_by_name(entity_name):
    """Get description by entity name (slower, for leaf nodes in reasoning check)."""
    ni = _get_node_info()
    for nid, info in ni.items():
        if info.get('name', '').lower() == entity_name.lower():
            return _get_desc(nid)
    return ''

def _get_relation_target_types(entity_idx, relations, client):
    """For each relation, find connected node types (sample max 50 neighbors)."""
    ni = _get_node_info()
    mapping = {}
    for rel in relations:
        try:
            nids, _ = client.get_neighbors(entity_idx, rel)
            types = set()
            for nid in nids[:50]:
                t = ni.get(nid, {}).get('type', '')
                if t: types.add(t)
            mapping[rel] = sorted(types)
        except:
            mapping[rel] = []
    return mapping


# ------------------------------------------------------------------ #
#  Variant config                                                     #
# ------------------------------------------------------------------ #
# (use_target_type, type_filter, fallback_mode)
# All variants use Q+S together (effective query = original query + dynamic subquery)
# All variants use ToG reasoning_check for stopping (no early-stop on hop exhaustion)
DYNAMIC_VARIANT_CONFIG = {
    "VD-N": (False, False, None),
}


# ------------------------------------------------------------------ #
#  Subquery generation (lazy, per-hop)                                #
# ------------------------------------------------------------------ #

def _format_evidence_path(triplets):
    """Format gold triplets as a numbered list for the LLM prompt."""
    if not triplets:
        return "(no evidence path available)"
    lines = []
    for i, t in enumerate(triplets, 1):
        if isinstance(t, (list, tuple)) and len(t) == 3:
            lines.append(f"  {i}. {t[0]} --[{t[1]}]--> {t[2]}")
    return "\n".join(lines)


def _parse_subquery_response(text):
    """Extract subquery (and optional target_type) from LLM JSON response.
    Tolerant to minor format issues — finds first {...} and parses.
    """
    if not text:
        return None, None
    # Find first JSON-like object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
        sq = obj.get("subquery", "").strip() if isinstance(obj, dict) else ""
        tt = obj.get("target_type", "").strip() if isinstance(obj, dict) else ""
        return sq if sq else None, tt if tt else None
    except json.JSONDecodeError:
        return None, None


def _strip_mcq_choices(query):
    """Strip MCQ choices (A/B/C/D block) from query if present."""
    import re
    m = re.search(r'\nA: .+\nB: .+\nC: .+\nD: ', query)
    if m:
        return query[:m.start()].strip()
    return query

def generate_dynamic_subquery(query, triplets, current_name, current_type,
                               hop_idx, total_hops, with_type, args,
                               path_node_ids=None):
    """Call LLM to generate a sub-question for the current hop.

    Returns: (subquery_text, target_type_or_None)
    """
    # Strip MCQ choices so subquery doesn't repeat A/B/C/D options
    query = _strip_mcq_choices(query)
    _exp = getattr(args, 'experiment', 'none')

    if _exp == '4' and triplets:
        # Process 4: type-based evidence path + sub-evidence + fallback
        ni = _get_node_info()
        ev_path, sub_evs = _build_type_evidence_path(triplets, ni, path_node_ids=path_node_ids)
        hop_0idx = hop_idx - 1  # hop_idx is 1-based
        sub_ev = sub_evs[hop_0idx] if hop_0idx < len(sub_evs) else ''

        # Check fallback: current type vs expected source type
        path_ids = []  # we don't have path_ids here, infer from triplets
        expected_src_type = '?'
        if sub_ev:
            expected_src_type = sub_ev.split(' →')[0].strip()
        is_fallback = (current_type != expected_src_type) if sub_ev and current_type else True

        if not is_fallback and sub_ev:
            # Extract relation and target type from sub-evidence
            parts = sub_ev.split('→')
            relation = ''
            target_type = ''
            if len(parts) >= 2:
                rel_match = re.search(r'\[(.+?)\]', parts[1])
                if rel_match:
                    relation = rel_match.group(1)
                target_type = parts[-1].strip() if len(parts) >= 2 else ''

            prompt = f"""You are guiding a multi-hop knowledge graph path-finder for biomedical question answering.

Original question: {query}

Evidence path: {ev_path}
Current step: {sub_ev}
- Relation to follow: "{relation}"
- Target node type: {target_type}
- Current node: "{current_name}" ({current_type})
- Hop: {hop_idx} of {total_hops}

Generate ONE focused sub-question for finding a "{target_type}" connected to "{current_name}" via "{relation}".
Output JSON only, no commentary, no markdown:
{{"subquery": "<your sub-question>", "target_type": "{target_type}"}}
"""
        else:
            prompt = f"""You are guiding a multi-hop knowledge graph path-finder for biomedical question answering.

Original question: {query}

Evidence path: {ev_path}
- Current node: "{current_name}" ({current_type})
- Hop: {hop_idx} of {total_hops}

Generate ONE focused sub-question to help find the answer.
Output JSON only, no commentary, no markdown:
{{"subquery": "<your sub-question>", "target_type": "<expected node type>"}}
"""
    else:
        # Original VD-N/VD-T prompt
        template = dynamic_subquery_prompt_with_type if with_type else dynamic_subquery_prompt_no_type
        prompt = template.format(
            query=query,
            evidence_path=_format_evidence_path(triplets),
            current_name=current_name,
            current_type=current_type or "?",
            hop_idx=hop_idx,
            total_hops=total_hops,
        )

    response = run_llm(
        prompt,
        temperature=args.temperature_exploration,
        max_tokens=256,
        opeani_api_keys=args.opeani_api_keys,
        engine=args.LLM_type,
        base_url=args.base_url,
    )

    sq, tt = _parse_subquery_response(response)
    if sq is None:
        sq = query
    return sq, (tt if with_type else None)


def ensure_subquery_for_current_hop(state, entity_idx, args, client, variant_cfg):
    """Lazily generate the subquery for the current hop of this entity's origin
    if it hasn't been generated yet. Mutates state.topic_state[origin].hops.
    """
    use_type, _, _ = variant_cfg
    origin = state.entity_to_origin.get(entity_idx, entity_idx)
    ts = state.topic_state.get(origin)
    if ts is None or ts.fallback_to_query:
        return
    if ts.current_hop >= ts.total_hops:
        return
    hop_info = ts.hops[ts.current_hop]
    if hop_info.get("subquery") is not None:
        return  # already generated

    # Look up the actual current entity at this hop (the entity_idx we're at now)
    # — this may differ from the original topic if we've advanced.
    current_name = client.idx_to_name(entity_idx)
    current_type = client.idx_to_type(entity_idx)

    # In pathbank fallback mode, don't pass GT triplets/path_node_ids to subquery generation
    _pb_mode = getattr(args, 'pathbank_mode', 'none')
    _pb_ev = getattr(state, '_pathbank_evidence', None)
    if _pb_mode.startswith('pb_') and _pb_ev is None:
        # Fallback in pathbank mode — no GT info
        _sq_triplets = []
        _sq_pnids = []
    else:
        _sq_triplets = state.gt_triplets
        _sq_pnids = state.record.get('path_node_ids', [])

    sq, tt = generate_dynamic_subquery(
        query=state.query,
        triplets=_sq_triplets,
        current_name=current_name,
        current_type=current_type,
        hop_idx=ts.current_hop + 1,
        total_hops=ts.total_hops,
        with_type=use_type,
        args=args,
        path_node_ids=_sq_pnids,
    )
    hop_info["subquery"] = sq
    if use_type and tt:
        hop_info["target"] = {"type": tt}
    elif use_type:
        hop_info["target"] = {"type": None}
    # Save info for analysis later
    hop_info["current_entity_at_gen"] = current_name
    hop_info["current_type_at_gen"] = current_type


# ------------------------------------------------------------------ #
#  Effective query for current hop (always Q+S in dynamic variants)   #
# ------------------------------------------------------------------ #

def get_effective_query_dynamic(state, entity_idx, variant_cfg):
    """Return (effective_query, target_type) — assumes subquery already generated."""
    _args = getattr(state, '_args_ref', None)
    origin = state.entity_to_origin.get(entity_idx, entity_idx)
    ts = state.topic_state.get(origin)
    if ts and ts.current_hop < ts.total_hops and not ts.fallback_to_query:
        hop_info = ts.hops[ts.current_hop]
        sq = hop_info.get("subquery")
        target_type = (hop_info.get("target") or {}).get("type")
        if sq:
            if _args and getattr(_args, 'subquery_only', False):
                return sq, target_type  # subquery only, no original query
            elif _args and getattr(_args, 'no_subquery', False):
                return state.query, None  # original query only
            return f"{state.query}\nSub-question for this step: {sq}", target_type
    return state.query, None


# ------------------------------------------------------------------ #
#  Init topic state (no pre-loaded subqueries — empty hop slots)      #
# ------------------------------------------------------------------ #

def init_topic_state_dynamic(record, max_depth):
    """Create empty hop slots so subqueries can be generated lazily.
    total_hops = max_depth (we don't know in advance how many hops will be needed).
    """
    topic_state = {}
    entity_to_origin = {}
    for tid_str in record.get("topic_entities", {}):
        tid = int(tid_str)
        topic_state[tid] = TopicHopState(
            hops=[{"subquery": None, "target": None} for _ in range(max_depth)],
            total_hops=max_depth,
        )
        entity_to_origin[tid] = tid
    return topic_state, entity_to_origin


# ------------------------------------------------------------------ #
#  Per-depth processing (with dynamic subquery generation hook)   #
# ------------------------------------------------------------------ #

def process_single_question_depth_dynamic(state, args, client, variant_cfg):
    """One depth step. Differences from the base version:
      1. Lazy-generate subquery for current hop before each entity is processed
      2. No early-stop on subquery exhaustion (always relies on reasoning_check)
    """
    query = state.query
    use_target_type, use_type_filter, fallback_mode = variant_cfg

    # Phase 1: Relation search & prune (per topic entity with effective query)
    current_entity_relations_list = []
    entity_relation_origins = []

    # Pre-compute experiment context if needed
    _exp = getattr(args, 'experiment', 'none')
    _ev_path = ''
    _sub_evs = []
    _path_types = []
    _path_history = getattr(state, '_path_history', [])
    _pb_ev = getattr(state, '_pathbank_evidence', None)
    if _pb_ev:
        # Pathbank mode: build evidence from pathbank path
        _path_types = _pb_ev['path_types']
        types, rels = _pb_ev['path_types'], _pb_ev['path_relations']
        _ev_path = " \u2192".join(f"{t} \u2192[{r}]" for t, r in zip(types[:-1], rels)) + f"\u2192 {types[-1]}"
        _sub_evs = [f"{types[i]} \u2192[{rels[i]}]\u2192 {types[i+1]}" for i in range(len(rels))]
    elif _exp in ('2A', '2B', '2C', '3A', '3B', '3C', '4', '5A', '5B'):
        # If in pathbank mode (pb_*) but no pathbank evidence (fallback),
        # do NOT use GT information — leave _path_types and _ev_path empty
        _pb_mode = getattr(args, 'pathbank_mode', 'none')
        if _pb_mode.startswith('pb_'):
            pass  # No GT usage in pathbank fallback — clean inference
        else:
            ni = _get_node_info()
            path_ids = state.record.get('path_node_ids', [])
            _path_types = [ni.get(pid, {}).get('type', '?') for pid in path_ids]
            gt_triplets = getattr(state, 'gt_triplets', [])
            if gt_triplets:
                _ev_path, _sub_evs = _build_type_evidence_path(gt_triplets, ni, path_node_ids=path_ids)

    # ============================================================
    # Evidence-based relation/entity variants (V0/V1/V2/V3)
    # ============================================================
    _rsm = getattr(args, 'relation_scoring_mode', 'default')
    _rsel = getattr(args, 'relation_select_mode', 'default')
    _ev_variant = _rsm in ('ev_rel', 'ev_rel_typefilt', 'ev_rel_typefilt_nextrel') or _rsel == 'target_type_single'

    if _ev_variant and not getattr(state, '_ev_fallback', False):
        depth_idx = state.current_depth - 1
        gt_triplets = getattr(state, 'gt_triplets', [])
        ni = _get_node_info()
        path_ids = state.record.get('path_node_ids', [])

        # Get evidence relation and target type for this depth
        if _pb_ev:
            # Pathbank mode: extract from pathbank path
            ev_rel = _pb_ev['path_relations'][depth_idx] if depth_idx < len(_pb_ev['path_relations']) else None
            ev_target_type = _pb_ev['path_types'][depth_idx + 1] if depth_idx + 1 < len(_pb_ev['path_types']) else None
            ev_next_rel = _pb_ev['path_relations'][depth_idx + 1] if depth_idx + 1 < len(_pb_ev['path_relations']) else None
        else:
            # Extract from GT triplets (non-pathbank experiments)
            _pb_mode_local2 = getattr(args, 'pathbank_mode', 'none')
            if _pb_mode_local2.startswith('pb_'):
                # Pathbank fallback — no GT usage
                ev_rel = None
                ev_target_type = None
                ev_next_rel = None
            else:
                ev_rel = gt_triplets[depth_idx][1] if depth_idx < len(gt_triplets) else None
                ev_target_type = _path_types[depth_idx + 1] if depth_idx + 1 < len(_path_types) else None
                ev_next_rel = gt_triplets[depth_idx + 1][1] if depth_idx + 1 < len(gt_triplets) else None

        if _rsel == 'target_type_single':
            # V1: Filter relations by target type → LLM picks 1
            for entity_idx in state.topic_entities:
                ensure_subquery_for_current_hop(state, entity_idx, args, client, variant_cfg)
                effective_query, _ = get_effective_query_dynamic(state, entity_idx, variant_cfg)
                origin = state.entity_to_origin.get(entity_idx, entity_idx)
                entity_name = state.topic_entities[entity_idx]
                entity_type = client.idx_to_type(entity_idx)

                all_rels = client.get_all_relations_of_entity(entity_idx)
                all_rels = [r for r in all_rels if r not in state.pre_relations]

                if not all_rels:
                    continue

                # Filter by target type
                if ev_target_type:
                    filtered_rels = []
                    for rel in all_rels:
                        neighbor_ids, _ = entity_search(entity_idx, rel, client)
                        if any((_fast_get_type(nid) or ni.get(nid, {}).get('type', '')) == ev_target_type for nid in neighbor_ids):
                            filtered_rels.append(rel)
                    if filtered_rels:
                        all_rels = filtered_rels
                    else:
                        # Fallback: no relation leads to target type
                        state._ev_fallback = True

                if getattr(state, '_ev_fallback', False):
                    break

                if len(all_rels) == 1:
                    selected_rel = all_rels[0]
                else:
                    # LLM picks 1 (no score, just relation name)
                    prompt = f"Select the ONE most relevant relation for answering the question.\n"
                    if _ev_path:
                        prompt += f"\nEvidence path: {_ev_path}\n"
                        if depth_idx < len(_sub_evs):
                            prompt += f"Current step: {_sub_evs[depth_idx]}\n"
                    prompt += f"\nQ: {effective_query}\nCurrent Entity: {entity_name} ({entity_type})\n"
                    prompt += f"Relations: {'; '.join(all_rels)}\n"
                    prompt += "Selected relation: "
                    result = run_llm(prompt, args.temperature_exploration, 256,
                                    args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
                    # Parse: find first relation name in response
                    selected_rel = None
                    for rel in all_rels:
                        if rel.lower() in result.lower():
                            selected_rel = rel
                            break
                    if not selected_rel:
                        selected_rel = all_rels[0]  # fallback to first

                current_entity_relations_list.append({
                    "entity": entity_idx, "relation": selected_rel, "score": 1.0, "head": True
                })
                entity_relation_origins.append(origin)

        elif _rsm in ('ev_rel', 'ev_rel_typefilt', 'ev_rel_typefilt_nextrel'):
            # V0/V2/V3: Force evidence relation
            if ev_rel is None:
                state._ev_fallback = True
            else:
                any_entity_has_rel = False
                for entity_idx in state.topic_entities:
                    origin = state.entity_to_origin.get(entity_idx, entity_idx)
                    all_rels = client.get_all_relations_of_entity(entity_idx)
                    if ev_rel in all_rels:
                        any_entity_has_rel = True
                        current_entity_relations_list.append({
                            "entity": entity_idx, "relation": ev_rel, "score": 1.0, "head": True
                        })
                        entity_relation_origins.append(origin)

                if not any_entity_has_rel:
                    state._ev_fallback = True

        # If fallback triggered, clear and let default flow handle it
        if getattr(state, '_ev_fallback', False):
            current_entity_relations_list = []
            entity_relation_origins = []

    if not _ev_variant or getattr(state, '_ev_fallback', False):
        # Default per-entity relation pruning loop (existing code)
        pass

    for entity_idx in state.topic_entities:
        if _ev_variant and not getattr(state, '_ev_fallback', False):
            break  # already handled above

        # Lazy generation of dynamic subquery for this hop
        ensure_subquery_for_current_hop(state, entity_idx, args, client, variant_cfg)
        effective_query, _ = get_effective_query_dynamic(state, entity_idx, variant_cfg)
        origin = state.entity_to_origin.get(entity_idx, entity_idx)

        if _exp.startswith('3'):
            # --- Process 3: Modified relation pruning ---
            entity_name = state.topic_entities[entity_idx]
            entity_type = client.idx_to_type(entity_idx)
            all_rels = client.get_all_relations_of_entity(entity_idx)
            all_rels = [r for r in all_rels if r not in state.pre_relations]
            all_rels.sort()

            if not all_rels:
                continue
            if len(all_rels) <= args.width:
                sc = 1.0 / len(all_rels)
                retrieve_relations = [{"entity": entity_idx, "relation": r, "score": sc, "head": True} for r in all_rels]
            else:
                depth_idx = state.current_depth - 1
                sub_ev = _sub_evs[depth_idx] if depth_idx < len(_sub_evs) else ''
                expected_src = _path_types[depth_idx] if depth_idx < len(_path_types) else '?'
                is_fallback = (entity_type != expected_src) if sub_ev else True

                # Build variant-specific prompt with dedicated few-shot examples.
                # Each variant has its own few-shot that demonstrates how to
                # use the additional context (evidence path, path history, type map).
                w = args.width

                # --- Few-shot construction per variant ---
                if _exp == '3A':
                    prompt = f"""Please retrieve up to {w} relations that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores is 1).
You are given an evidence path (the expected node-type sequence from topic to answer) and the current step (the segment to follow at this hop). Prioritize relations that match the current step.

Evidence path: disease →[indication]→ drug →[target]→ gene/protein
Current step: disease →[indication]→ drug

Q: Which drugs are indicated for treating Alzheimer's disease?
Current Entity: Alzheimer disease (disease)
Relations:
1. associated with
2. contraindication
3. indication
4. linked to
5. parent-child
6. phenotype present
A: 1. {{indication (Score: 0.6)}}: Matches the current step relation "indication", directly connecting diseases to their indicated drugs.
2. {{associated with (Score: 0.3)}}: May reveal gene/protein associations useful for downstream hops.
3. {{linked to (Score: 0.1)}}: Provides additional contextual links.

Evidence path: drug →[side effect]→ effect/phenotype
Current step: drug →[side effect]→ effect/phenotype

Q: What are the known side effects of Metformin?
Current Entity: Metformin (drug)
Relations:
1. carrier
2. contraindication
3. enzyme
4. indication
5. off-label use
6. side effect
7. synergistic interaction
8. target
9. transporter
A: 1. {{side effect (Score: 0.8)}}: Directly matches the current step relation "side effect".
2. {{contraindication (Score: 0.1)}}: Contraindications may overlap with side effects.
3. {{target (Score: 0.1)}}: Drug targets can explain mechanism-related side effects.

"""
                elif _exp == '3B':
                    prompt = f"""Please retrieve up to {w} relations that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores is 1).
You are given an evidence path, the path traversed so far, and the current step. Prioritize relations that match the current step.

Evidence path: disease →[indication]→ drug →[target]→ gene/protein
Path so far: Alzheimer disease (disease) →[indication]→ Donepezil (drug)
Current step: drug →[target]→ gene/protein

Q: Which proteins are targeted by drugs indicated for Alzheimer's disease?
Current Entity: Donepezil (drug)
Relations:
1. carrier
2. contraindication
3. enzyme
4. indication
5. off-label use
6. side effect
7. synergistic interaction
8. target
9. transporter
A: 1. {{target (Score: 0.8)}}: Matches the current step relation "target", connecting the drug to its protein targets.
2. {{enzyme (Score: 0.1)}}: Enzymes involved in drug metabolism may be relevant.
3. {{carrier (Score: 0.1)}}: Carrier proteins can be related to drug targets.

Evidence path: drug →[side effect]→ effect/phenotype
Current step: drug →[side effect]→ effect/phenotype

Q: What are the known side effects of Metformin?
Current Entity: Metformin (drug)
Relations:
1. carrier
2. contraindication
3. enzyme
4. indication
5. off-label use
6. side effect
7. synergistic interaction
8. target
9. transporter
A: 1. {{side effect (Score: 0.8)}}: Directly matches the current step relation "side effect".
2. {{contraindication (Score: 0.1)}}: Contraindications may overlap with side effects.
3. {{target (Score: 0.1)}}: Drug targets can explain mechanism-related side effects.

"""
                else:  # 3C
                    prompt = f"""Please retrieve up to {w} relations that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores is 1).
You are given an evidence path, the path traversed so far, and the current step. Each relation shows the node types it leads to. Prioritize relations whose target types match the current step.

Evidence path: disease →[indication]→ drug →[target]→ gene/protein
Path so far: Alzheimer disease (disease) →[indication]→ Donepezil (drug)
Current step: drug →[target]→ gene/protein

Q: Which proteins are targeted by drugs indicated for Alzheimer's disease?
Current Entity: Donepezil (drug)
Relations (→ leads to node types):
1. carrier → gene/protein
2. contraindication → disease
3. enzyme → gene/protein
4. indication → disease
5. off-label use → disease
6. side effect → effect/phenotype
7. synergistic interaction → drug
8. target → gene/protein
9. transporter → gene/protein
A: 1. {{target (Score: 0.7)}}: Matches the current step "target" and leads to gene/protein, the expected target type.
2. {{carrier (Score: 0.15)}}: Also leads to gene/protein, may include relevant protein targets.
3. {{enzyme (Score: 0.15)}}: Leads to gene/protein, enzymes involved in drug metabolism.

Evidence path: drug →[side effect]→ effect/phenotype
Current step: drug →[side effect]→ effect/phenotype

Q: What are the known side effects of Metformin?
Current Entity: Metformin (drug)
Relations (→ leads to node types):
1. carrier → gene/protein
2. contraindication → disease
3. enzyme → gene/protein
4. indication → disease
5. off-label use → disease
6. side effect → effect/phenotype
7. synergistic interaction → drug
8. target → gene/protein
9. transporter → gene/protein
A: 1. {{side effect (Score: 0.8)}}: Matches the current step and leads to effect/phenotype.
2. {{contraindication (Score: 0.1)}}: Contraindications may overlap with side effects.
3. {{target (Score: 0.1)}}: Drug targets can explain mechanism-related side effects.

"""
                # --- Append experiment context for the actual query ---
                if _ev_path:
                    prompt += f"Evidence path: {_ev_path}\n"
                if not is_fallback and sub_ev:
                    if _exp in ('3B', '3C') and _path_history:
                        ph_str = ' → '.join([f"{h[0]} ({h[1]}) →[{h[2]}]→ {h[3]} ({h[4]})" for h in _path_history])
                        prompt += f"Path so far: {ph_str}\n"
                    prompt += f"Current step: {sub_ev}\n"
                else:
                    _pb_mode_pr = getattr(args, 'pathbank_mode', 'none')
                    _pb_ev_pr = getattr(state, '_pathbank_evidence', None)
                    if _pb_mode_pr.startswith('pb_') and _pb_ev_pr is None:
                        prompt += f"Current hop: {state.current_depth}\n"
                    else:
                        prompt += f"Current hop: {state.current_depth} of {len(getattr(state, 'gt_triplets', []))}\n"
                    prompt += f"Current node: {entity_name} ({entity_type})\n"

                prompt += f"\nQ: {effective_query}\nCurrent Entity: {entity_name} ({entity_type})\n"

                if _exp == '3C':
                    rel_type_map = _get_relation_target_types(entity_idx, all_rels, client)
                    prompt += "Relations (→ leads to node types):\n"
                    for i, rel in enumerate(all_rels, 1):
                        types = rel_type_map.get(rel, [])
                        prompt += f"{i}. {rel} → {', '.join(types) if types else '?'}\n"
                else:
                    prompt += "Relations:\n"
                    for i, rel in enumerate(all_rels, 1):
                        prompt += f"{i}. {rel}\n"
                prompt += "A: "

                result = run_llm(prompt, args.temperature_exploration, args.max_length,
                                args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
                from inference.primekg_func import clean_relations
                flag, retrieve_relations = clean_relations(result, entity_idx, all_rels)
                if not flag:
                    retrieve_relations = []
        else:
            retrieve_relations = relation_search_prune(
                entity_idx, state.topic_entities[entity_idx],
                state.pre_relations, effective_query, args, client
            )

        current_entity_relations_list.extend(retrieve_relations)
        entity_relation_origins.extend([origin] * len(retrieve_relations))

    # ============================================================
    # Relation scoring mode
    # ============================================================
    _rsm = getattr(args, 'relation_scoring_mode', 'default')

    if _rsm == 'cascade' and current_entity_relations_list:
        # Cascade = rescore + source entity score multiplication
        # First: apply rescore (same as below)
        _rsm = 'rescore'  # fall through to rescore, then multiply source scores after

    if _rsm == 'rescore' and current_entity_relations_list:
        # Cross-relation re-scoring: collect unique relations, LLM scores them together
        unique_rels = list(set(e['relation'] for e in current_entity_relations_list))

        if len(unique_rels) > 1:
            # Build re-scoring prompt
            depth_idx = state.current_depth - 1
            effective_query_for_rescore = query  # use original query for cross-relation comparison
            # Try to get subquery
            for eidx in state.topic_entities:
                eq, _ = get_effective_query_dynamic(state, eidx, variant_cfg)
                effective_query_for_rescore = eq
                break

            rescore_prompt = f"Please score the following relations for their contribution to answering the question on a scale from 0 to 1 (the sum of all scores is 1).\n"
            if _ev_path:
                rescore_prompt += f"\nEvidence path: {_ev_path}\n"
                if depth_idx < len(_sub_evs):
                    rescore_prompt += f"Current step: {_sub_evs[depth_idx]}\n"

            # List starting entities
            ent_names = [f"{client.idx_to_name(eidx)} ({client.idx_to_type(eidx)})" for eidx in state.topic_entities]
            rescore_prompt += f"\nStarting entities: {', '.join(ent_names)}\n"
            rescore_prompt += f"\nQ: {effective_query_for_rescore}\n"
            rescore_prompt += f"Relations:\n"
            for i, rel in enumerate(unique_rels, 1):
                rescore_prompt += f"{i}. {rel}\n"
            rescore_prompt += "Score: "

            rescore_result = run_llm(rescore_prompt, args.temperature_exploration, args.max_length,
                                     args.opeani_api_keys, args.LLM_type, base_url=args.base_url)

            # Parse scores
            from inference.utils import clean_scores as _clean_scores_util
            rel_scores_raw = _clean_scores_util(rescore_result, unique_rels)
            n_rels = len(unique_rels)
            is_uniform = all(abs(s - 1.0/n_rels) < 1e-6 for s in rel_scores_raw) if n_rels > 0 else True

            if not is_uniform:
                total_s = sum(rel_scores_raw)
                rel_score_map = {rel: (s / total_s if total_s > 0 else 1.0/n_rels) for rel, s in zip(unique_rels, rel_scores_raw)}
            else:
                rel_score_map = {rel: 1.0/n_rels for rel in unique_rels}

            # Apply re-scored relation scores
            for e in current_entity_relations_list:
                e['score'] = rel_score_map.get(e['relation'], 1.0/n_rels)

            # Log rescore results for analysis
            if not hasattr(state, '_rescore_log'):
                state._rescore_log = []
            state._rescore_log.append({
                'depth': state.current_depth,
                'unique_rels': unique_rels,
                'raw_scores': rel_scores_raw,
                'normalized': dict(rel_score_map),
                'is_uniform_fallback': is_uniform,
                'prompt_snippet': rescore_prompt[-200:],
            })

    # Phase 2: Entity search & score
    total_candidates = []
    total_scores = []
    total_relations = []
    total_entities_id = []
    total_topic_entities = []
    total_head = []

    # ============================================================
    # Entity scoring — branch by entity_scoring_mode
    # ============================================================
    _esm = getattr(args, 'entity_scoring_mode', 'default')

    if _esm in ('sbert_only', 'sbert_top5_llm', 'sbert_top10_llm'):
        # --- Two-pass approach for sbert_only / sbert_top5_llm ---
        # Pass 1: Collect all candidates per (entity, relation) pair
        _all_groups = []  # list of {entity, relation, origin, cand_ids, cand_names, effective_query, rel_score}
        for rel_idx, entity in enumerate(current_entity_relations_list):
            origin = entity_relation_origins[rel_idx]
            source_entity_idx = entity['entity']
            effective_query, target_type = get_effective_query_dynamic(state, source_entity_idx, variant_cfg)

            cand_ids, cand_names = entity_search(entity['entity'], entity['relation'], client)
            if len(cand_names) == 0:
                continue

            # Type filtering (same as default)
            _do_type_filter = use_type_filter and target_type
            if _exp == '2C':
                depth_idx = state.current_depth - 1
                gt_target_type = _path_types[depth_idx + 1] if depth_idx + 1 < len(_path_types) else None
                if gt_target_type:
                    fp = [(eid, nm) for eid, nm in zip(cand_ids, cand_names)
                          if _get_node_info().get(eid, {}).get('type', '') == gt_target_type]
                    if fp: cand_ids, cand_names = zip(*fp); cand_ids = list(cand_ids); cand_names = list(cand_names)
            elif _do_type_filter:
                fp = [(eid, nm) for eid, nm in zip(cand_ids, cand_names) if client.idx_to_type(eid) == target_type]
                if fp: cand_ids, cand_names = zip(*fp); cand_ids = list(cand_ids); cand_names = list(cand_names)

            # V2/V3: Evidence-based entity type filtering
            _rsm_ev = getattr(args, 'relation_scoring_mode', 'default')
            if _rsm_ev in ('ev_rel_typefilt', 'ev_rel_typefilt_nextrel') and not getattr(state, '_ev_fallback', False):
                _depth_idx = state.current_depth - 1
                _ev_target_type = _path_types[_depth_idx + 1] if _depth_idx + 1 < len(_path_types) else None
                if _ev_target_type:
                    fp = [(eid, nm) for eid, nm in zip(cand_ids, cand_names)
                          if ni.get(eid, {}).get('type', '') == _ev_target_type]
                    if fp:
                        cand_ids, cand_names = zip(*fp)
                        cand_ids, cand_names = list(cand_ids), list(cand_names)
                    else:
                        # Type filtering yielded 0 → fallback for this and future depths
                        state._ev_fallback = True

                # V3: Additional filtering by next-depth relation
                if _rsm_ev == 'ev_rel_typefilt_nextrel' and not getattr(state, '_ev_fallback', False):
                    _pb_ev_v3 = getattr(state, '_pathbank_evidence', None)
                    if _pb_ev_v3:
                        _pb_rels_v3 = _pb_ev_v3.get('path_relations', [])
                        _next_rel = _pb_rels_v3[_depth_idx + 1] if _depth_idx + 1 < len(_pb_rels_v3) else None
                    elif not getattr(args, 'pathbank_mode', 'none').startswith('pb_'):
                        _gt = getattr(state, 'gt_triplets', [])
                        _next_rel = _gt[_depth_idx + 1][1] if _depth_idx + 1 < len(_gt) else None
                    else:
                        _next_rel = None  # pathbank fallback — no GT
                    if _next_rel and cand_ids:
                        source_eid = entity['entity']
                        fp_nr = []
                        for eid, nm in zip(cand_ids, cand_names):
                            eid_rels = _fast_get_relations(eid) or client.get_all_relations_of_entity(eid)
                            # Has next-depth relation, excluding the relation back to source
                            if _next_rel in eid_rels:
                                fp_nr.append((eid, nm))
                        if fp_nr:
                            cand_ids, cand_names = zip(*fp_nr)
                            cand_ids, cand_names = list(cand_ids), list(cand_names)
                        # If 0 after next-rel filtering → keep V2 results (don't fallback)

            _all_groups.append({
                'entity_info': entity,
                'origin': origin,
                'cand_ids': cand_ids,
                'cand_names': cand_names,
                'effective_query': effective_query,
                'rel_score': entity['score'],
                'relation': entity['relation'],
            })

        if _esm == 'sbert_only':
            # --- SBERT-only: score ALL neighbors per group, normalize per group ---
            for grp in _all_groups:
                scored_ids, scored_scores = _sbert_score_entities(
                    grp['effective_query'], grp['cand_ids'], args.sbert_model, top_k=None)
                scored_names = [client.idx_to_name(eid) for eid in scored_ids]
                scores = [s * grp['rel_score'] for s in scored_scores]
                (total_candidates, total_scores, total_relations,
                 total_entities_id, total_topic_entities, total_head) = update_history(
                    scored_names, grp['entity_info'], scores, scored_ids,
                    total_candidates, total_scores, total_relations,
                    total_entities_id, total_topic_entities, total_head)

        elif _esm in ('sbert_top5_llm', 'sbert_top10_llm'):
            # Pass 2: SBERT top-k per group, check if any group < k
            _sbert_k = 10 if _esm == 'sbert_top10_llm' else 5
            _group_top5 = []
            any_under5 = False
            for grp in _all_groups:
                top_k = min(_sbert_k, len(grp['cand_ids']))
                top_ids, sbert_sc = _sbert_score_entities(
                    grp['effective_query'], grp['cand_ids'], args.sbert_model, top_k=top_k)
                top_names = [client.idx_to_name(eid) for eid in top_ids]
                _group_top5.append({**grp, 'top_ids': top_ids, 'top_names': top_names, 'sbert_scores': sbert_sc})
                if len(grp['cand_ids']) < _sbert_k:
                    any_under5 = True

            _desc_mode = getattr(args, 'entity_desc_mode', 'none')

            if any_under5 and len(_group_top5) > 1:
                # --- Merge mode: one prompt for all candidates ---
                all_ids, all_names, all_rels, all_sbert = [], [], [], []
                eq = _group_top5[0]['effective_query']  # use first query (same for all in VD-N)
                from inference.prompts import score_entity_candidates_prompt_bio
                from inference.primekg_func import clean_scores
                prompt = "Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).\n"
                prompt += f"\nQ: {eq}\n"
                for ginfo in _group_top5:
                    prompt += f"\nRelation: {ginfo['relation']}\n"
                    entities_str = _format_entities_with_desc(
                        ginfo['top_names'], ginfo['top_ids'], eq, _desc_mode)
                    prompt += "Entities: " + entities_str + "\n"
                    all_ids.extend(ginfo['top_ids'])
                    all_names.extend(ginfo['top_names'])
                    all_rels.extend([ginfo['relation']] * len(ginfo['top_ids']))
                    all_sbert.extend(ginfo['sbert_scores'])

                prompt += f"\nScore all {len(all_names)} entities (sum = 1): "
                result = run_llm(prompt, args.temperature_exploration, args.max_length,
                                args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
                llm_scores = clean_scores(result, all_names)

                n = len(all_names)
                is_uniform = all(abs(s - 1.0/n) < 1e-6 for s in llm_scores) if n > 0 else True

                if is_uniform:
                    # Fallback: SBERT scores, normalize across all
                    total_s = sum(all_sbert)
                    final_scores = [s / total_s for s in all_sbert] if total_s > 0 else [1.0/n]*n
                else:
                    total_s = sum(llm_scores)
                    final_scores = [s / total_s for s in llm_scores] if total_s > 0 else [1.0/n]*n

                # Assign back to groups for update_history
                idx = 0
                for ginfo in _group_top5:
                    gn = len(ginfo['top_ids'])
                    grp_scores = final_scores[idx:idx+gn]
                    # Note: no rel_score multiplication in merge mode — scores are already cross-relation
                    (total_candidates, total_scores, total_relations,
                     total_entities_id, total_topic_entities, total_head) = update_history(
                        ginfo['top_names'], ginfo['entity_info'], grp_scores, ginfo['top_ids'],
                        total_candidates, total_scores, total_relations,
                        total_entities_id, total_topic_entities, total_head)
                    idx += gn

            else:
                # --- Normal mode: per-group LLM scoring ---
                from inference.prompts import score_entity_candidates_prompt_bio
                from inference.primekg_func import clean_scores
                for ginfo in _group_top5:
                    prompt = score_entity_candidates_prompt_bio.format(ginfo['effective_query'], ginfo['relation'])
                    entities_str = _format_entities_with_desc(
                        ginfo['top_names'], ginfo['top_ids'], ginfo['effective_query'], _desc_mode)
                    prompt += entities_str + '\nScore: '
                    result = run_llm(prompt, args.temperature_exploration, args.max_length,
                                    args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
                    llm_scores = clean_scores(result, ginfo['top_names'])

                    n = len(ginfo['top_names'])
                    is_uniform = all(abs(s - 1.0/n) < 1e-6 for s in llm_scores) if n > 0 else True

                    if is_uniform:
                        final_scores = ginfo['sbert_scores']
                    else:
                        total_s = sum(llm_scores)
                        final_scores = [s / total_s for s in llm_scores] if total_s > 0 else [1.0/n]*n

                    scores = [s * ginfo['rel_score'] for s in final_scores]
                    (total_candidates, total_scores, total_relations,
                     total_entities_id, total_topic_entities, total_head) = update_history(
                        ginfo['top_names'], ginfo['entity_info'], scores, ginfo['top_ids'],
                        total_candidates, total_scores, total_relations,
                        total_entities_id, total_topic_entities, total_head)

        # Skip the default per-entity loop below (already handled above)

    else:
        # --- Default entity scoring path (unchanged) ---
        pass

    for rel_idx, entity in enumerate(current_entity_relations_list):
        if _esm in ('sbert_only', 'sbert_top5_llm', 'sbert_top10_llm'):
            break  # already scored above

        origin = entity_relation_origins[rel_idx]
        source_entity_idx = entity['entity']

        effective_query, target_type = get_effective_query_dynamic(state, source_entity_idx, variant_cfg)

        entity_candidates_id, entity_candidates_name = entity_search(
            entity['entity'], entity['relation'], client
        )
        if len(entity_candidates_name) == 0:
            continue

        # Type filtering
        _do_type_filter = use_type_filter and target_type
        if _exp == '2C':
            depth_idx = state.current_depth - 1
            gt_target_type = _path_types[depth_idx + 1] if depth_idx + 1 < len(_path_types) else None
            if gt_target_type:
                filtered_pairs = [
                    (eid, name) for eid, name in zip(entity_candidates_id, entity_candidates_name)
                    if _get_node_info().get(eid, {}).get('type', '') == gt_target_type
                ]
                if filtered_pairs:
                    entity_candidates_id, entity_candidates_name = zip(*filtered_pairs)
                    entity_candidates_id = list(entity_candidates_id)
                    entity_candidates_name = list(entity_candidates_name)
        elif _do_type_filter:
            filtered_pairs = [
                (eid, name) for eid, name in zip(entity_candidates_id, entity_candidates_name)
                if client.idx_to_type(eid) == target_type
            ]
            if filtered_pairs:
                entity_candidates_id, entity_candidates_name = zip(*filtered_pairs)
                entity_candidates_id = list(entity_candidates_id)
                entity_candidates_name = list(entity_candidates_name)

        # --- Default: existing pipeline ---
        # Downsampling
        if len(entity_candidates_id) >= 20:
            if args.entity_sampling == "bm25":
                from inference.utils import compute_bm25_similarity
                candidate_docs = [client.get_doc_info(eid) for eid in entity_candidates_id]
                top_docs, _ = compute_bm25_similarity(
                    effective_query, candidate_docs, args.max_entity_candidates)
                indices = [candidate_docs.index(d) for d in top_docs if d in candidate_docs]
            elif args.entity_sampling == "sbert":
                from inference.utils import retrieve_top_docs
                candidate_docs = [client.get_doc_info(eid) for eid in entity_candidates_id]
                top_docs, _ = retrieve_top_docs(
                    effective_query, candidate_docs, args.sbert_model, args.max_entity_candidates)
                indices = [candidate_docs.index(d) for d in top_docs if d in candidate_docs]
            else:
                indices = random.sample(
                    range(len(entity_candidates_name)),
                    min(args.max_entity_candidates, len(entity_candidates_name)),
                )
            entity_candidates_id = [entity_candidates_id[i] for i in indices]
            entity_candidates_name = [entity_candidates_name[i] for i in indices]

        if len(entity_candidates_id) == 0:
            continue

        if _exp in ('2A', '2B'):
            # --- Process 2A/2B: Modified entity scoring prompt ---
            ni = _get_node_info()
            depth_idx = state.current_depth - 1
            sub_ev = _sub_evs[depth_idx] if _exp == '2B' and depth_idx < len(_sub_evs) else ''
            src_type = client.idx_to_type(source_entity_idx)
            expected_src = _path_types[depth_idx] if depth_idx < len(_path_types) else '?'
            is_fallback = (src_type != expected_src) if sub_ev else True
            gt_target_type = _path_types[depth_idx + 1] if _exp == '2B' and depth_idx + 1 < len(_path_types) else ''

            sorted_pairs = sorted(zip(entity_candidates_name, entity_candidates_id))
            cand_names_sorted = [p[0] for p in sorted_pairs]
            cand_ids_sorted = [p[1] for p in sorted_pairs]

            prompt = "Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).\n"

            if _exp == '2B' and gt_target_type and not is_fallback:
                prompt += f'Prioritize entities of type "{gt_target_type}".\n'

            if _exp == '2B' and _ev_path:
                prompt += f"\nEvidence path: {_ev_path}\n"
                if not is_fallback and sub_ev:
                    prompt += f"Current step: {sub_ev}\n"
                else:
                    _pb_mode_pr = getattr(args, 'pathbank_mode', 'none')
                    _pb_ev_pr = getattr(state, '_pathbank_evidence', None)
                    if _pb_mode_pr.startswith('pb_') and _pb_ev_pr is None:
                        prompt += f"Current hop: {state.current_depth}\n"
                    else:
                        prompt += f"Current hop: {state.current_depth} of {len(getattr(state, 'gt_triplets', []))}\n"

            prompt += f"\nQ: {effective_query}\nRelation: {entity['relation']}\n"

            if _exp == '2A':
                from rank_bm25 import BM25Okapi
                descs = [_get_desc(eid) for eid in cand_ids_sorted]
                chunks = [_get_best_chunk_bm25(effective_query, d) if d else '' for d in descs]
                tokenized = [re.findall(r'[a-z0-9]+', d.lower()) for d in descs]
                if all(tokenized):
                    bm25 = BM25Okapi(tokenized)
                    bm25_scores = bm25.get_scores(re.findall(r'[a-z0-9]+', effective_query.lower())).tolist()
                else:
                    bm25_scores = [0.0] * len(cand_ids_sorted)
                combined = sorted(zip(cand_names_sorted, cand_ids_sorted, bm25_scores, chunks), key=lambda x: -x[2])
                prompt += "Entities (sorted by relevance to the question):\n"
                for i, (name, eid, bscore, chunk) in enumerate(combined, 1):
                    etype = ni.get(int(eid), {}).get('type', '?')
                    prompt += f'{i}. {name} ({etype}) — "{chunk[:150]}"\n'
            else:
                prompt += "Entities: "
                entries = []
                for name, eid in zip(cand_names_sorted, cand_ids_sorted):
                    etype = ni.get(int(eid), {}).get('type', '?')
                    entries.append(f"{name} ({etype})")
                prompt += "; ".join(entries) + "\n"

            prompt += "Score: "
            result = run_llm(prompt, args.temperature_exploration, args.max_length,
                            args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
            from inference.primekg_func import clean_scores
            entity_scores = clean_scores(result, cand_names_sorted)
            if all(s == 0 for s in entity_scores):
                scores = [1/len(cand_names_sorted) * entity['score']] * len(cand_names_sorted)
            else:
                scores = [float(x) * entity['score'] for x in entity_scores]
            entity_candidates_name = cand_names_sorted
            entity_candidates_id = cand_ids_sorted
        else:
            scores, entity_candidates_name, entity_candidates_id = entity_score(
                effective_query, entity_candidates_id, entity_candidates_name,
                entity['score'], entity['relation'], args
            )
        (total_candidates, total_scores, total_relations,
         total_entities_id, total_topic_entities, total_head) = update_history(
            entity_candidates_name, entity, scores, entity_candidates_id,
            total_candidates, total_scores, total_relations,
            total_entities_id, total_topic_entities, total_head
        )

    # Cascade: multiply scores by normalized source entity scores
    _rsm_orig = getattr(args, 'relation_scoring_mode', 'default')
    if _rsm_orig == 'cascade' and hasattr(state, '_source_entity_scores') and total_scores:
        src_scores = state._source_entity_scores
        for i in range(len(total_scores)):
            src_eid = total_topic_entities[i]
            src_sc = src_scores.get(src_eid, 1.0 / max(len(src_scores), 1))
            total_scores[i] = total_scores[i] * src_sc

    # Track explored
    for eid, sc in zip(total_entities_id, total_scores):
        if eid not in state.explored_entities or sc > state.explored_entities[eid]:
            state.explored_entities[eid] = sc

    record = state.record
    if len(total_candidates) == 0:
        answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
        state.result_record = _make_result_dynamic(
            record, query, answer_text,
            state.cluster_chain_of_entities,
            state.explored_entities, state.topic_state, args)
        if hasattr(state, "_rescore_log"): state.result_record["rescore_log"] = state._rescore_log
        state.finished = True
        return state

    # Width override for pathbank mode (width=1 per path)
    _eff_width = getattr(state, '_effective_width', None)
    if _eff_width is not None:
        import copy
        _args_prune = copy.copy(args)
        _args_prune.width = _eff_width
    else:
        _args_prune = args
    flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(
        total_entities_id, total_relations, total_candidates,
        total_topic_entities, total_head, total_scores, _args_prune, client
    )
    state.cluster_chain_of_entities.append(chain_of_entities)

    if flag:
        _advance_hop_counters(state, entities_id, total_entities_id, total_topic_entities)

        # Store entity scores for cascade mode (normalized, sum=1)
        _rsm_check = getattr(args, 'relation_scoring_mode', 'default')
        if _rsm_check == 'cascade':
            eid_to_score = {}
            for eid, sc in zip(total_entities_id, total_scores):
                if eid in entities_id:
                    eid_to_score[eid] = max(eid_to_score.get(eid, 0), sc)
            total_sc = sum(eid_to_score.values())
            if total_sc > 0:
                state._source_entity_scores = {eid: sc / total_sc for eid, sc in eid_to_score.items()}
            else:
                state._source_entity_scores = {eid: 1.0 / len(entities_id) for eid in entities_id}

        # Track path history for 3B
        if _exp in ('3B', '3C'):
            for eid in entities_id:
                ename = client.idx_to_name(eid)
                etype = client.idx_to_type(eid)
                for ci, ti in zip(total_entities_id, total_topic_entities):
                    if ci == eid:
                        tname = client.idx_to_name(ti)
                        ttype = client.idx_to_type(ti)
                        for ri, ei in zip(total_relations, total_entities_id):
                            if ei == eid:
                                _path_history.append((tname, ttype, ri, ename, etype))
                                break
                        break
            state._path_history = _path_history

        # Skip reasoning check in pathbank mode (always traverse full hop count)
        if getattr(state, '_skip_reasoning_check', False):
            state.topic_entities = {
                eid: client.idx_to_name(eid) for eid in entities_id
            }
            state.pre_relations = pre_relations
            state.pre_heads = pre_heads
            return state

        # ToG reasoning_check (5A/5B: modified prompt; others: original)
        if _exp in ('5A', '5B'):
            # Build path-structured reasoning check prompt
            from collections import defaultdict
            all_depths = []
            for depth_data in state.cluster_chain_of_entities:
                triplets = []
                for item in depth_data:
                    if isinstance(item, (list, tuple)):
                        if len(item) == 3 and all(isinstance(x, str) for x in item):
                            triplets.append(item)
                        else:
                            for t in item:
                                if isinstance(t, (list, tuple)) and len(t) == 3:
                                    triplets.append(t)
                all_depths.append(triplets)

            # Build valid paths
            paths = [[(t[0], t[1], t[2])] for t in all_depths[0]] if all_depths else []
            for d in range(1, len(all_depths)):
                new_paths = []
                dst_map = defaultdict(list)
                for i, p in enumerate(paths):
                    dst_map[p[-1][2].lower()].append(i)
                extended = set()
                for t in all_depths[d]:
                    if t[0].lower() in dst_map:
                        for pi in dst_map[t[0].lower()]:
                            new_paths.append(paths[pi] + [tuple(t)])
                            extended.add(pi)
                for i, p in enumerate(paths):
                    if i not in extended:
                        new_paths.append(p)
                paths = new_paths

            prompt = "Given a question and the explored knowledge graph paths, you are asked to answer whether it's sufficient for you to answer the question with these paths and your knowledge (Yes or No).\n\n"
            prompt += f"Q: {query}\nExplored Paths:\n"
            for i, path in enumerate(paths, 1):
                parts = [path[0][0]]
                for t in path:
                    parts.append(f'→[{t[1]}]→ {t[2]}')
                line = f"  Path {i}: {' '.join(parts)}"
                if _exp == '5B':
                    leaf = path[-1][2]
                    desc = _get_desc_by_name(leaf)
                    if desc:
                        chunk = _get_best_chunk_bm25(query, desc)
                        if chunk:
                            line += f'\n    └ {leaf}: "{chunk[:150]}"'
                prompt += line + "\n"
            prompt += "A: "

            resp = run_llm(prompt, args.temperature_reasoning, args.max_length,
                          args.opeani_api_keys, args.LLM_type, base_url=args.base_url)
            from inference.primekg_func import extract_answer, if_true
            stop = if_true(extract_answer(resp))
        else:
            stop, reasoning_response = reasoning(query, state.cluster_chain_of_entities, args)
        if stop:
            answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
            state.result_record = _make_result_dynamic(
                record, query, answer_text,
                state.cluster_chain_of_entities,
                state.explored_entities, state.topic_state, args)
            if hasattr(state, "_rescore_log"): state.result_record["rescore_log"] = state._rescore_log
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
        state.result_record = _make_result_dynamic(
            record, query, answer_text,
            state.cluster_chain_of_entities,
            state.explored_entities, state.topic_state, args)
        if hasattr(state, "_rescore_log"): state.result_record["rescore_log"] = state._rescore_log
        state.finished = True

    return state


def _make_result_dynamic(record, query, answer_text, chains, explored, topic_state, args):
    """Like _make_result but also persists the dynamically generated subqueries."""
    result = _make_result(record, query, answer_text, chains, explored, args)
    # Strip TopicHopState → JSON-friendly dict
    result["dynamic_subqueries"] = {
        str(origin): {
            "current_hop": ts.current_hop,
            "total_hops": ts.total_hops,
            "hops": ts.hops,
        }
        for origin, ts in (topic_state or {}).items()
    }
    result["variant"] = getattr(args, 'variant', None)
    result["entity_scoring_mode"] = getattr(args, 'entity_scoring_mode', 'default')
    result["relation_scoring_mode"] = getattr(args, 'relation_scoring_mode', 'default')
    return result


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="AdaPath inference")
    # ----- I/O -----
    parser.add_argument("--input", required=True,
                        help="Path to QA jsonl with 'query', 'topic_entities', 'answer_entity'.")
    parser.add_argument("--triplets_file", default=None,
                        help="Optional separate file for GT triplets (joined by data_id). "
                             "If omitted, triplets are read from --input record itself.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skb_root", type=str, default="data/primekg")

    # ----- Path-finding -----
    parser.add_argument("--width", type=int, default=3,
                        help="Beam width per hop (default 3 = AdaPath).")
    parser.add_argument("--depth", type=int, default=3,
                        help="Maximum number of hops (default 3 = AdaPath).")
    parser.add_argument("--max_entity_candidates", type=int, default=10)

    # ----- LLM -----
    parser.add_argument("--LLM_type", type=str,
                        default="meta-llama/Llama-3.1-70B-Instruct",
                        help="HuggingFace model id loaded via AutoModelForCausalLM.")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--temperature_exploration", type=float, default=0.4)
    parser.add_argument("--temperature_reasoning", type=float, default=0)
    parser.add_argument("--temperature_select", type=float, default=0.0,
                        help="Temperature for selection-type LLM calls "
                             "(hop selection, answer type, etc.). 0.0 = greedy.")

    # ----- Scoring backends -----
    parser.add_argument("--sbert_device", type=str, default="cuda:0")
    parser.add_argument("--sbert_model_name", type=str,
                        default="msmarco-distilbert-base-tas-b")
    parser.add_argument("--sbert_max_seq_length", type=int, default=512)

    # ----- Path-bank -----
    parser.add_argument("--train_pathbank_dir", type=str, default="",
                        help="Directory with the per-query path bank "
                             "(produced by `pathbank.build_pathbank`).")
    parser.add_argument("--pathbank_file", type=str, default="",
                        help="Optional single pathbank jsonl (overrides --train_pathbank_dir).")
    parser.add_argument("--pathbank_hop_cap", type=str, default="2,4,8",
                        help="Width caps per hop as comma-separated 1h,2h,3h.")
    parser.add_argument("--pathbank_match_k", type=int, default=5,
                        help="Top-k matched train queries to use (default: 5).")

    parser.add_argument("--pathbank_ablation", action="store_true",
                        help="Path-bank ablation: bypass the matched pathbank "
                             "evidence and run the free-BFS fallback for every "
                             "query (used to measure the contribution of the "
                             "matched path bank).")

    parser.add_argument("--cot_fallback", action="store_true", default=True,
                        help="If answer generation fails to produce {answer}, "
                             "retry with the CoT prompt (no triplets).")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=20)

    args = parser.parse_args()

    # ----- Fixed defaults (AdaPath single mode) -----
    args.variant = "VD-N"
    args.experiment = "3C"
    args.pathbank_mode = "pb_llmhop_w3"
    args.prune_tools = "llm"
    args.entity_sampling = "sbert"
    args.entity_scoring_mode = "sbert_top5_llm"
    args.relation_scoring_mode = "ev_rel_typefilt_nextrel"
    args.relation_select_mode = "default"
    args.entity_desc_mode = "none"
    # Backward-compat shims (unused; HF backend)
    args.opeani_api_keys = "EMPTY"
    args.base_url = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse pathbank hop caps
    _cap_vals = [int(x) for x in args.pathbank_hop_cap.split(',')]
    args._pb_caps = {1: _cap_vals[0], 2: _cap_vals[1], 3: _cap_vals[2]}

    variant_cfg = DYNAMIC_VARIANT_CONFIG[args.variant]

    # Parse paths
    input_parts = args.input.split('/')
    split_name = os.path.splitext(input_parts[-1])[0]
    query_type = input_parts[-2]
    dataset_model = input_parts[-3]

    if args.output_dir is None:
        exp_tag = args.experiment if args.experiment != "none" else "VDN_baseline"
        if args.entity_scoring_mode != "default" or args.relation_scoring_mode != "default":
            # Scoring variant → separate directory
            proc_dir = f"entity_scoring_variants"
        else:
            proc_dir_map = {
                'none': 'process_vdn_baseline',
                '2A': 'process2_entity_scoring', '2B': 'process2_entity_scoring', '2C': 'process2_entity_scoring',
                '3A': 'process3_relation_pruning', '3B': 'process3_relation_pruning', '3C': 'process3_relation_pruning',
                '4': 'process4_dynamic_subquery',
                '5A': 'process5_reasoning_check', '5B': 'process5_reasoning_check',
            }
    os.makedirs(args.output_dir, exist_ok=True)

    pb_tag = "_adapath"
    if args.pathbank_ablation:
        pb_tag += "_pbAbl"
    output_file = os.path.join(
        args.output_dir,
        f"adapath{pb_tag}_{query_type}.jsonl"
    )

    args._node_info = _get_node_info()

    print(f"Variant: {args.variant}")
    print(f"Input: {args.input}")
    print(f"Output: {output_file}")
    print(f"Workers: {args.num_workers}")

    records = load_biokgqa(args.input)

    # Optional: load triplets from separate file (for v2/gpt54 datasets where
    # subquery files don't carry triplets — they live in v2/generated_*/test.jsonl)
    triplets_by_data_id = {}
    if args.triplets_file and os.path.exists(args.triplets_file):
        with open(args.triplets_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "data_id" in r and "triplets" in r:
                        triplets_by_data_id[r["data_id"]] = r["triplets"]
        print(f"Loaded triplets for {len(triplets_by_data_id)} records from {args.triplets_file}")

    # ---- Inference-time pathbank matcher ----
    pathbank_by_id = {}
    _pb_mode = args.pathbank_mode
    _pb_matcher = None
    if True:
        print(f"Setting up inference-time pathbank matcher...")
        # Load train records
        train_qa_path = os.path.join(os.path.dirname(args.triplets_file), 'train.jsonl')
        train_records_raw = []
        with open(train_qa_path) as f:
            for line in f:
                if line.strip():
                    train_records_raw.append(json.loads(line))
        print(f"  Train records: {len(train_records_raw)}")

        # Load train pathbank
        _qtype_for_pb = os.path.basename(os.path.dirname(args.input))  # explicit/implicit/bare
        train_pb = _load_train_pathbank_by_id_and_hop(args.train_pathbank_dir, _qtype_for_pb)
        print(f"  Train pathbank: {len(train_pb)} records")

        # Determine SBERT device for matcher (use first available SBERT GPU or CPU)
        _matcher_sbert_dev = getattr(args, 'sbert_device', 'cpu')
        _pb_matcher = InferenceTimePathbankMatcher(
            train_records_raw, train_pb, _qtype_for_pb, sbert_device=_matcher_sbert_dev)
        print(f"  Matcher ready (sbert={_matcher_sbert_dev})")

    print("Loading PrimeSKB...")
    skb = load_skb('prime', root=args.skb_root, download_processed=True)
    client = PrimeKGClient(skb)
    args._client = client

    rel_desc_path = os.path.join(os.path.dirname(__file__), 'qa_construction', 'relation_descriptions.json')
    if os.path.exists(rel_desc_path):
        with open(rel_desc_path) as f:
            args.relation_descriptions = json.load(f)
    else:
        args.relation_descriptions = {}

    need_sbert = (args.prune_tools == "sentencebert" or args.entity_sampling == "sbert"
                  or args.entity_scoring_mode in ("sbert_top5_llm", "sbert_top10_llm", "sbert_only"))
    if need_sbert:
        # Limit per-process GPU memory to prevent OOM when sharing GPU
        sbert_gpu_idx = int(args.sbert_device.split(':')[1]) if ':' in args.sbert_device else 0
        # Set per-GPU memory fraction based on available memory
        _gpu_free = torch.cuda.mem_get_info(sbert_gpu_idx)[0] / (1024**3)  # free GB
        _frac = min(0.23, (_gpu_free - 2) / (torch.cuda.get_device_properties(sbert_gpu_idx).total_memory / (1024**3)))
        _frac = max(_frac, 0.10)  # minimum 10%
        torch.cuda.set_per_process_memory_fraction(_frac, sbert_gpu_idx)
        from sentence_transformers import SentenceTransformer
        args.sbert_model = SentenceTransformer(args.sbert_model_name, device=args.sbert_device)
        args.sbert_model.max_seq_length = args.sbert_max_seq_length

    # Resume
    done_keys = set()
    done_ids = set()
    done_pnids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("data_id") is not None:
                        done_ids.add(r["data_id"])
                    done_pnids.add(tuple(r.get("path_node_ids", [])))
        print(f"Resuming: {len(done_ids)} done (by data_id), {len(done_pnids)} (by pnids)")

    def _is_done(r):
        if r.get("data_id") is not None and r["data_id"] in done_ids:
            return True
        if not done_ids and tuple(r["path_node_ids"]) in done_pnids:
            return True
        return False

    remaining = [r for r in records if not _is_done(r)]
    print(f"Remaining: {len(remaining)} queries")

    def _run_single_pass(record, pathbank_evidence=None, effective_width=None,
                         last_depth_width=None, skip_reasoning=False, subgraph=None):
        """Run one pass of the pipeline. If pathbank_evidence is set, use it as evidence source."""
        query = record["query"]
        topic_entities = {int(k): v for k, v in record["topic_entities"].items()}
        answer_ids = [record["answer_entity"]["id"]]

        if not topic_entities:
            result = generate_without_explored_paths(query, args)
            return _make_result_dynamic(record, query, result, [], {}, {}, args)

        max_depth = pathbank_evidence["hop"] if pathbank_evidence else args.depth
        topic_state, entity_to_origin = init_topic_state_dynamic(record, max_depth)

        state = QuestionState(
            idx=0,
            record=record,
            query=query,
            topic_entities=topic_entities,
            answer_ids=answer_ids,
            pre_heads=[-1] * len(topic_entities),
            topic_state=topic_state,
            entity_to_origin=entity_to_origin,
        )
        # Attach gold triplets for subquery generation prompt
        gt_triplets = record.get("triplets", [])
        if not gt_triplets and triplets_by_data_id:
            gt_triplets = triplets_by_data_id.get(record.get("data_id"), [])
        state.gt_triplets = gt_triplets

        # Args reference for get_effective_query_dynamic
        state._args_ref = args

        # Pathbank overrides
        # --pathbank_ablation: skip setting evidence so the code falls into
        # the `_pb_ev is None` fallback branch for every query.
        if pathbank_evidence and not args.pathbank_ablation:
            state._pathbank_evidence = pathbank_evidence
        if effective_width is not None:
            state._effective_width = effective_width
        if last_depth_width is not None:
            state._last_depth_width = last_depth_width
        if skip_reasoning:
            state._skip_reasoning_check = True
        if subgraph:
            state._subgraph = subgraph

        for depth in range(1, max_depth + 1):
            state.current_depth = depth
            state = process_single_question_depth_dynamic(state, args, client, variant_cfg)
            if state.finished:
                return state
        return state

    def _do_cot_fallback(result):
        """Apply CoT fallback inside worker thread."""
        if not getattr(args, 'cot_fallback', False):
            return result
        results_text = result.get("results", "")
        import re as _re
        _m = _re.findall(r"\{([^}]+)\}", results_text)
        extracted = _m[1].strip() if len(_m) >= 2 and _m[0].strip().lower() in ("yes", "no") else (_m[0].strip() if _m else "")
        if extracted:
            return result
        query = result.get("question", "")

        cot_result = generate_without_explored_paths(query, args)
        result["results_original"] = results_text
        result["results"] = cot_result
        result["cot_fallback_used"] = True
        return result

    def _process_one_inner(record):
        query = record["query"]
        _pb_mode_local = getattr(args, 'pathbank_mode', 'none')
        data_id = str(record.get("data_id", ""))

        # Get pathbank paths: either from pre-loaded dict or inference-time matching
        pb_paths = []
        _pred_answer_type = None
        _pred_answer_type_raw = None
        _match_info = None

        if _pb_matcher is not None:
            topic_entities = {int(k): v for k, v in record["topic_entities"].items()}
            topic_id = list(topic_entities.keys())[0] if topic_entities else None
            topic_type = client.idx_to_type(topic_id) if topic_id else ""

            _pred_answer_type, _pred_answer_type_raw = _predict_answer_type(query, args)
            answer_type = _pred_answer_type

            # Top-k match
            _match_k = getattr(args, 'pathbank_match_k', 5)
            paths_by_hop, _match_info = _pb_matcher.match_top_k(
                query, topic_type, answer_type, k=_match_k, hops_to_use=[1, 2, 3])

            # Sig dedup per hop + traversable filter + cap
            paths_by_hop = _dedup_path_sigs(paths_by_hop)
            caps = getattr(args, '_pb_caps', {1: 2, 2: 4, 3: 8})
            paths_by_hop = _filter_and_cap_paths(paths_by_hop, topic_id, client, caps)

            paths_by_hop_counts = {str(h): len(v) for h, v in paths_by_hop.items()}

            # LLM hop selection (fallback: hop with the most matched paths)
            _chosen_hop = _llm_select_hop(paths_by_hop, query, args)
            if _chosen_hop is None or not paths_by_hop.get(_chosen_hop):
                _chosen_hop = max([1, 2, 3], key=lambda h: len(paths_by_hop.get(h, [])))
            pb_paths = paths_by_hop.get(_chosen_hop, [])
            exec_width = 3

            _llm_ps_used = False

            # --pathbank_ablation: empty pb_paths so the fallback branch below
            # (free exploration with pathbank_fallback=True) is taken.
            if args.pathbank_ablation:
                pb_paths = []

            if not pb_paths:
                # No traversable paths → fallback to vanilla VD-N
                state = _run_single_pass(record)
                if state.finished:
                    result = state.result_record
                else:
                    answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
                    result = _make_result_dynamic(record, query, answer_text,
                                                  state.cluster_chain_of_entities,
                                                  state.explored_entities, state.topic_state, args)
                result["pathbank_mode"] = _pb_mode_local
                result["pathbank_paths_by_hop"] = paths_by_hop_counts
                result["pathbank_paths_used"] = 0
                result["pathbank_paths_success"] = 0
                result["pathbank_fallback"] = True
                result["predicted_answer_type"] = _pred_answer_type
                result["predicted_answer_type_raw"] = _pred_answer_type_raw
                if _match_info is not None:
                    result["match_info"] = _match_info
                return result

            # Step 5: Execute each path
            all_chains = []
            all_explored = {}
            paths_success = 0
            for pb_path in pb_paths:
                state = _run_single_pass(record, pathbank_evidence=pb_path,
                                         effective_width=exec_width,
                                         skip_reasoning=True)
                if state.finished and not state.cluster_chain_of_entities:
                    continue
                if len(state.cluster_chain_of_entities) == pb_path["hop"]:
                    all_chains.extend(state.cluster_chain_of_entities)
                    paths_success += 1
                for eid, sc in state.explored_entities.items():
                    if eid not in all_explored or sc > all_explored[eid]:
                        all_explored[eid] = sc

            if not all_chains:
                # All paths failed → fallback
                state = _run_single_pass(record)
                if state.finished:
                    result = state.result_record
                else:
                    answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
                    result = _make_result_dynamic(record, query, answer_text,
                                                  state.cluster_chain_of_entities,
                                                  state.explored_entities, state.topic_state, args)
                result["pathbank_mode"] = _pb_mode_local
                result["pathbank_paths_by_hop"] = paths_by_hop_counts
                result["pathbank_chosen_hop"] = _chosen_hop
                result["pathbank_exec_width"] = exec_width
                result["pathbank_paths_used"] = len(pb_paths)
                result["pathbank_paths_success"] = 0
                result["pathbank_fallback"] = True
                result["predicted_answer_type"] = _pred_answer_type
                result["predicted_answer_type_raw"] = _pred_answer_type_raw
                if _match_info is not None:
                    result["match_info"] = _match_info
                return result

            # Step 6: Triplet dedup
            triplets_before = sum(len(d) for d in all_chains)
            all_chains = _dedup_triplets(all_chains)
            triplets_after = sum(len(d) for d in all_chains)

            # Step 7: Generate answer
            answer_text = generate_answer(query, all_chains, args)
            result = _make_result_dynamic(record, query, answer_text,
                                          all_chains, all_explored, None, args)
            result["pathbank_mode"] = _pb_mode_local
            result["pathbank_paths_by_hop"] = paths_by_hop_counts
            result["pathbank_chosen_hop"] = _chosen_hop
            result["pathbank_exec_width"] = exec_width
            result["pathbank_paths_used"] = len(pb_paths)
            result["pathbank_paths_success"] = paths_success
            result["pathbank_fallback"] = False
            result["triplets_before_dedup"] = triplets_before
            result["triplets_after_dedup"] = triplets_after
            result["predicted_answer_type"] = _pred_answer_type
            result["predicted_answer_type_raw"] = _pred_answer_type_raw
            if _llm_ps_used:
                result["llm_path_selection_used"] = True
            if _match_info is not None:
                result["match_info"] = _match_info
            return result

        elif _pb_mode_local != 'none':
            pb_paths = pathbank_by_id.get(data_id, [])

        # --pathbank_ablation: empty pb_paths so the loop is skipped and
        # we fall through to the free-exploration branch below, taken
        # exactly once per query.
        if args.pathbank_ablation:
            pb_paths = []

        if _pb_mode_local == 'none' or not pb_paths:
            # Original flow (no pathbank)
            state = _run_single_pass(record)
            if state.finished:
                return state.result_record
            answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
            result = _make_result_dynamic(record, query, answer_text,
                                          state.cluster_chain_of_entities,
                                          state.explored_entities, state.topic_state, args)
            if _pb_mode_local != 'none':
                result["pathbank_mode"] = _pb_mode_local
                result["pathbank_paths_used"] = 0
                result["pathbank_paths_success"] = 0
                result["pathbank_fallback"] = True
            return result

        # ---- Pathbank mode: iterate over paths, width=1 each ----
        all_chains = []
        all_explored = {}
        paths_success = 0

        for pb_path in pb_paths:
            state = _run_single_pass(record, pathbank_evidence=pb_path,
                                     effective_width=1, skip_reasoning=True)
            # If state.finished with no candidates (dead end) → path failed
            if state.finished and not state.cluster_chain_of_entities:
                continue
            # Check if chain has expected depth (path not truncated)
            if len(state.cluster_chain_of_entities) == pb_path["hop"]:
                all_chains.extend(state.cluster_chain_of_entities)
                paths_success += 1
            # Collect explored entities regardless
            for eid, sc in state.explored_entities.items():
                if eid not in all_explored or sc > all_explored[eid]:
                    all_explored[eid] = sc

        if not all_chains:
            # All paths failed → fallback: VD-N + sbert5llm (no evidence, width=3)
            state = _run_single_pass(record)
            if state.finished:
                result = state.result_record
            else:
                answer_text = generate_answer(query, state.cluster_chain_of_entities, args)
                result = _make_result_dynamic(record, query, answer_text,
                                              state.cluster_chain_of_entities,
                                              state.explored_entities, state.topic_state, args)
            result["pathbank_mode"] = _pb_mode_local
            result["pathbank_paths_used"] = len(pb_paths)
            result["pathbank_paths_success"] = 0
            result["pathbank_fallback"] = True
            if _pred_answer_type is not None or _pred_answer_type_raw is not None:
                result["predicted_answer_type"] = _pred_answer_type
                result["predicted_answer_type_raw"] = _pred_answer_type_raw
            if _match_info is not None:
                result["match_info"] = _match_info
            return result

        answer_text = generate_answer(query, all_chains, args)
        result = _make_result_dynamic(record, query, answer_text,
                                      all_chains, all_explored, None, args)
        result["pathbank_mode"] = _pb_mode_local
        result["pathbank_paths_used"] = len(pb_paths)
        result["pathbank_paths_success"] = paths_success
        result["pathbank_fallback"] = False
        # Inference-time matching info
        if _pred_answer_type is not None or _pred_answer_type_raw is not None:
            result["predicted_answer_type"] = _pred_answer_type
            result["predicted_answer_type_raw"] = _pred_answer_type_raw
        if _match_info is not None:
            result["match_info"] = _match_info
        return result

    def process_one(record):
        """Wrapper: run pipeline + CoT fallback inside worker thread."""
        result = _process_one_inner(record)
        return _do_cot_fallback(result)

    completed = 0
    with open(output_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one, r): r for r in remaining}
            for future in tqdm(as_completed(futures), total=len(remaining),
                               desc=f"Dynamic {args.variant} ({args.num_workers}w)"):
                try:
                    result = future.result()
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    completed += 1
                    if completed % 100 == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error: {e}")

    print(f"\nDone! {completed} results saved to {output_file}")


if __name__ == "__main__":
    main()
