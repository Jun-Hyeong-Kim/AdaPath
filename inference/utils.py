"""Shared utilities for AdaPath inference."""

import json
import os
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from inference.prompts import cot_prompt_bio
from inference.llm import generate_text


# ------------------------------------------------------------------ #
#  LLM                                                                #
# ------------------------------------------------------------------ #

def run_llm(prompt, temperature, max_tokens, opeani_api_keys=None,
            engine="meta-llama/Llama-3.1-70B-Instruct",
            base_url=None):
    """Sequential HuggingFace generation.

    `opeani_api_keys` and `base_url` are accepted for backward compatibility
    with the call sites but are not used by the local HF backend.
    """
    device = os.environ.get("ADAPATH_DEVICE", "cuda:0")
    dtype_str = os.environ.get("ADAPATH_DTYPE", "bfloat16")
    import torch
    dtype = getattr(torch, dtype_str)
    return generate_text(
        prompt, model_name=engine,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=device, dtype=dtype,
    )


# ------------------------------------------------------------------ #
#  Score / answer extraction                                          #
# ------------------------------------------------------------------ #

def clean_scores(string, entity_candidates):
    """Extract float scores from LLM output."""
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(x) for x in scores]
    if len(scores) == len(entity_candidates):
        return scores
    return [1 / len(entity_candidates)] * len(entity_candidates)


def extract_answer(text):
    """Extract text between { and }."""
    start = text.find("{")
    end = text.find("}")
    if start != -1 and end != -1:
        return text[start + 1:end].strip()
    return ""


def if_true(prompt):
    """Check if response is 'yes'."""
    return prompt.lower().strip().replace(" ", "") == "yes"


# ------------------------------------------------------------------ #
#  Output                                                             #
# ------------------------------------------------------------------ #

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name,
                 q_id=None, answer_ids=None, explored_entities=None,
                 output_dir=None):
    record = {
        "question": question,
        "results": answer,
        "reasoning_chains": cluster_chain_of_entities,
    }
    if q_id is not None:
        record["q_id"] = q_id
    if answer_ids is not None:
        record["answer_ids"] = answer_ids
    if explored_entities is not None:
        record["explored_entities"] = explored_entities

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "adapath_{}.jsonl".format(file_name))
    else:
        filepath = "adapath_{}.jsonl".format(file_name)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ #
#  Fallback generation                                                #
# ------------------------------------------------------------------ #

def generate_without_explored_paths(question, args):
    """Generate answer without KG exploration (CoT fallback)."""
    prompt = cot_prompt_bio + "\n\nQ: " + question + "\nA:"
    return run_llm(prompt, args.temperature_reasoning, args.max_length,
                   getattr(args, 'opeani_api_keys', None), args.LLM_type,
                   base_url=getattr(args, 'base_url', None))


# ------------------------------------------------------------------ #
#  BM25 / SentenceBERT utilities                                      #
# ------------------------------------------------------------------ #

def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [p[0] for p in pairs[:width]]
    top_scores = [p[1] for p in pairs[:width]]
    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]
    return relations, doc_scores


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id):
    relations = []
    if all(s == 0 for s in topn_scores):
        topn_scores = [1 / len(topn_scores)] * len(topn_scores)
    for rel, score in zip(topn_relations, topn_scores):
        relations.append({
            "entity": entity_id,
            "relation": rel,
            "score": score,
            "head": True,
        })
    return True, relations
