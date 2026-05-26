"""
PrimeKG-specific functions for AdaPath.
Mirrors ToG/wiki_func.py but simplified for PrimeKG:
- No head/tail distinction (undirected graph)
- No value entities (all neighbors are proper nodes)
- Only 18 biomedical relation types (no abandon_rels filtering needed)
"""

import json
import random
import re

from inference.prompts import (
    extract_entities_prompt_bio,
    extract_relation_prompt_bio,
    score_entity_candidates_prompt_bio,
    prompt_evaluate_bio,
    answer_prompt_bio,
)
from inference.utils import (
    run_llm,
    clean_scores,
    extract_answer,
    if_true,
    save_2_jsonl,
    compute_bm25_similarity,
    retrieve_top_docs,
    clean_relations_bm25_sent,
)


# ------------------------------------------------------------------ #
#  Entity Linking (NEW — STaRK QA has no pre-annotated topic entities)#
# ------------------------------------------------------------------ #

def entity_linking(question, args, client):
    """
    Extract topic entities from the question using LLM + PrimeKG name matching.

    Returns:
        dict: {node_idx: node_name} or empty dict if no entities found.
    """
    # Step 1: Ask LLM to extract entity names
    prompt = extract_entities_prompt_bio % question
    result = run_llm(prompt, args.temperature_exploration, args.max_length,
                     args.opeani_api_keys, args.LLM_type,
                     base_url=getattr(args, 'base_url', None))

    # Parse JSON list from LLM output
    try:
        # Try to find JSON list in the response
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            entity_names = json.loads(match.group())
        else:
            entity_names = []
    except (json.JSONDecodeError, ValueError):
        entity_names = []

    if not entity_names:
        return {}

    # Step 2: Match entity names to PrimeKG nodes
    topic_entity = {}
    for name in entity_names:
        if not isinstance(name, str) or not name.strip():
            continue

        # Try exact match first, then fuzzy search
        matched = client.name_to_idx(name)
        if matched:
            idx = matched[0]  # Take the first exact match
            topic_entity[idx] = client.idx_to_name(idx)
        else:
            # Fuzzy search
            results = client.search_entity(name, top_k=1)
            if results:
                idx, ename = results[0]
                topic_entity[idx] = ename

    return topic_entity


# ------------------------------------------------------------------ #
#  Relation Search & Prune                                           #
# ------------------------------------------------------------------ #

def clean_relations(string, entity_idx, all_relations):
    """Parse LLM output to extract relations and scores."""
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations = []
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in all_relations:
            relations.append({
                "entity": entity_idx,
                "relation": relation,
                "score": score,
                "head": True,  # PrimeKG is undirected, always head
            })
    if not relations:
        return False, "No relations found"
    return True, relations


def construct_relation_prune_prompt(question, entity_name, entity_type, total_relations, args):
    """Build the relation pruning prompt."""
    return (
        extract_relation_prompt_bio % (args.width, args.width)
        + question
        + '\nTopic Entity: ' + entity_name + ' (' + entity_type + ')'
        + '\nRelations:\n'
        + '\n'.join([f"{i}. {item}" for i, item in enumerate(total_relations, start=1)])
        + '\nA: '
    )


def relation_search_prune(entity_idx, entity_name, pre_relations, question, args, client):
    """
    Fetch all relations of an entity, filter already-explored ones,
    then use LLM/BM25/SentenceBERT to score and select top-width relations.
    """
    all_relations = client.get_all_relations_of_entity(entity_idx)
    entity_type = client.idx_to_type(entity_idx)

    # Remove previously explored relations
    all_relations = [r for r in all_relations if r not in pre_relations]
    all_relations.sort()

    if not all_relations:
        return []

    # If fewer relations than width, adjust
    if len(all_relations) <= args.width:
        # Return all with equal scores
        score = 1.0 / len(all_relations)
        return [{"entity": entity_idx, "relation": r, "score": score, "head": True}
                for r in all_relations]

    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, entity_type, all_relations, args)
        result = run_llm(prompt, args.temperature_exploration, args.max_length,
                         args.opeani_api_keys, args.LLM_type,
                     base_url=getattr(args, 'base_url', None))
        flag, retrieve_relations = clean_relations(result, entity_idx, all_relations)
        if flag:
            return retrieve_relations
        else:
            return []
    elif args.prune_tools == "bm25":
        # Use relation name + description for better matching
        rel_descs = getattr(args, 'relation_descriptions', {})
        docs = [f"{r}: {rel_descs[r]}" if r in rel_descs else r for r in all_relations]
        doc_to_rel = {doc: rel for doc, rel in zip(docs, all_relations)}
        topn_docs, topn_scores = compute_bm25_similarity(question, docs, args.width)
        topn_relations = [doc_to_rel[d] for d in topn_docs]
        _, relations = clean_relations_bm25_sent(topn_relations, topn_scores, entity_idx)
        return relations
    elif args.prune_tools == "sentencebert":
        rel_descs = getattr(args, 'relation_descriptions', {})
        docs = [f"{r}: {rel_descs[r]}" if r in rel_descs else r for r in all_relations]
        doc_to_rel = {doc: rel for doc, rel in zip(docs, all_relations)}
        topn_docs, topn_scores = retrieve_top_docs(question, docs, args.sbert_model, args.width)
        topn_relations = [doc_to_rel[d] for d in topn_docs]
        _, relations = clean_relations_bm25_sent(topn_relations, topn_scores, entity_idx)
        return relations
    else:
        return []


# ------------------------------------------------------------------ #
#  Entity Search                                                      #
# ------------------------------------------------------------------ #

def entity_search(entity_idx, relation, client):
    """
    Retrieve candidate entities connected to entity_idx via relation.
    Returns (id_list, name_list).
    """
    return client.get_neighbors(entity_idx, relation)


# ------------------------------------------------------------------ #
#  Entity Scoring                                                     #
# ------------------------------------------------------------------ #

def construct_entity_score_prompt(question, relation, entity_candidates):
    """Build the entity scoring prompt."""
    return (
        score_entity_candidates_prompt_bio.format(question, relation)
        + "; ".join(entity_candidates)
        + '\nScore: '
    )


def entity_score(question, entity_candidates_id, entity_candidates, score, relation, args):
    """Score entity candidates for relevance to the question."""
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id

    # Sort alphabetically so prompt order is deterministic
    zipped = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)
        result = run_llm(prompt, args.temperature_exploration, args.max_length,
                         args.opeani_api_keys, args.LLM_type,
                     base_url=getattr(args, 'base_url', None))
        entity_scores = clean_scores(result, entity_candidates)
    elif args.prune_tools == "bm25":
        if not getattr(args, 'no_description', False) and hasattr(args, '_client'):
            scoring_docs = [args._client.get_doc_info(eid) for eid in entity_candidates_id]
        else:
            scoring_docs = entity_candidates
        _, entity_scores = compute_bm25_similarity(question, scoring_docs, len(scoring_docs))
    elif args.prune_tools == "sentencebert":
        if not getattr(args, 'no_description', False) and hasattr(args, '_client'):
            scoring_docs = [args._client.get_doc_info(eid) for eid in entity_candidates_id]
        else:
            scoring_docs = entity_candidates
        _, entity_scores = retrieve_top_docs(question, scoring_docs, args.sbert_model, len(scoring_docs))
    else:
        entity_scores = [1 / len(entity_candidates)] * len(entity_candidates)

    if all(s == 0 for s in entity_scores):
        return [1 / len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    else:
        return [float(x) * score for x in entity_scores], entity_candidates, entity_candidates_id


# ------------------------------------------------------------------ #
#  History Update                                                     #
# ------------------------------------------------------------------ #

def update_history(entity_candidates, entity, scores, entity_candidates_id,
                   total_candidates, total_scores, total_relations,
                   total_entities_id, total_topic_entities, total_head):
    """Accumulate entity candidates across all relations at current depth."""
    n = len(entity_candidates)
    candidates_relation = [entity['relation']] * n
    topic_entities = [entity['entity']] * n
    head_num = [entity['head']] * n

    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)

    return (total_candidates, total_scores, total_relations,
            total_entities_id, total_topic_entities, total_head)


# ------------------------------------------------------------------ #
#  Entity Prune                                                       #
# ------------------------------------------------------------------ #

def entity_prune(total_entities_id, total_relations, total_candidates,
                 total_topic_entities, total_head, total_scores, args, client):
    """
    Select top-width entities by score.
    Returns (flag, cluster_chain_of_entities, entities_id, relations, heads).
    """
    zipped = list(zip(total_entities_id, total_relations, total_candidates,
                      total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)

    entities_id = [x[0] for x in sorted_zipped[:args.width]]
    relations = [x[1] for x in sorted_zipped[:args.width]]
    candidates = [x[2] for x in sorted_zipped[:args.width]]
    topics = [x[3] for x in sorted_zipped[:args.width]]
    heads = [x[4] for x in sorted_zipped[:args.width]]
    scores = [x[5] for x in sorted_zipped[:args.width]]

    # Filter out zero-scored entities
    filtered = [(eid, rel, cand, top, hd, sc)
                for eid, rel, cand, top, hd, sc
                in zip(entities_id, relations, candidates, topics, heads, scores)
                if sc != 0]

    if not filtered:
        return False, [], [], [], []

    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered))

    # Convert topic entity indices to names
    tops_names = [client.idx_to_name(t) for t in tops]

    # Build chain of entities as triplets
    cluster_chain_of_entities = [
        [(tops_names[i], relations[i], candidates[i]) for i in range(len(candidates))]
    ]
    return True, cluster_chain_of_entities, entities_id, relations, heads


# ------------------------------------------------------------------ #
#  Reasoning                                                          #
# ------------------------------------------------------------------ #

def reasoning(question, cluster_chain_of_entities, args):
    """Check if accumulated knowledge triplets suffice to answer the question."""
    prompt = prompt_evaluate_bio + question
    chain_prompt = '\n'.join([
        ', '.join([str(x) for x in chain])
        for sublist in cluster_chain_of_entities
        for chain in sublist
    ])
    prompt += "\nKnowledge Triplets: " + chain_prompt + '\nA: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length,
                       args.opeani_api_keys, args.LLM_type,
                     base_url=getattr(args, 'base_url', None))
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response


# ------------------------------------------------------------------ #
#  Answer Generation                                                  #
# ------------------------------------------------------------------ #

def generate_answer(question, cluster_chain_of_entities, args):
    """Generate final answer using LLM with accumulated knowledge triplets."""
    prompt = answer_prompt_bio % question + '\n'
    chain_prompt = '\n'.join([
        ', '.join([str(x) for x in chain])
        for sublist in cluster_chain_of_entities
        for chain in sublist
    ])
    prompt += "Knowledge Triplets: " + chain_prompt + '\nA: '
    return run_llm(prompt, args.temperature_reasoning, args.max_length,
                   args.opeani_api_keys, args.LLM_type,
                   base_url=getattr(args, 'base_url', None))


def half_stop(question, cluster_chain_of_entities, depth, args,
              q_id=None, answer_ids=None, explored_entities=None,
              output_dir=None):
    """Handle early termination when no new knowledge is added."""
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities,
                 file_name=args.split, q_id=q_id, answer_ids=answer_ids,
                 explored_entities=explored_entities,
                 output_dir=output_dir)
