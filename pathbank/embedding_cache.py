"""Build / load BM25 indexes for nodes and relations.

Caches live under pathbank/cache/:
  graph.pickle              igraph.Graph (undirected, attrs: name, type, relation)
  node_text.json            dict[node_id -> preprocessed text]
  bm25_node_nolen.pkl       BM25Okapi (b=0) over tokenized node_text
  bm25_rel_nolen.pkl        BM25Okapi (b=0) over relation descriptions

All caches are produced once and reused across runs/queries.
PrimeKG processed tensors are expected under data/primekg/processed/.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = _ROOT / "pathbank" / "cache"
DEFAULT_PRIMEKG_DIR = _ROOT / "data" / "primekg" / "processed"


# ---------------------------------------------------------------------------
# Relation descriptions (18 PrimeKG relations) — embedded inline
# ---------------------------------------------------------------------------

RELATION_DESCRIPTIONS: Dict[str, str] = {
    "ppi": "A protein-protein interaction where one gene/protein physically binds to, interacts with, or functionally associates with another gene/protein. This gene/protein has a direct molecular interaction with another gene/protein.",
    "carrier": "A gene/protein acts as a carrier for a drug, meaning the protein transports or carries the drug molecule within the body. The drug is carried, bound, or transported by this gene/protein for distribution.",
    "enzyme": "A gene/protein acts as an enzyme that metabolizes, catalyzes the breakdown of, or biochemically transforms a drug. The drug is metabolized, processed, or biotransformed by this enzyme.",
    "target": "A drug targets, acts on, or modulates a specific gene/protein to exert its therapeutic or pharmacological effect. The gene/protein is the molecular target, receptor, or site of action of the drug.",
    "transporter": "A gene/protein acts as a transporter that actively moves a drug across cell membranes, mediating the drug uptake, efflux, absorption, or excretion. The drug is transported into or out of cells by this transporter protein.",
    "contraindication": "A drug is contraindicated for a disease, meaning the drug should not be used, is unsafe, or is not recommended for patients with this disease or condition.",
    "indication": "A drug is indicated for treating, managing, or preventing a disease. The drug is prescribed, recommended, approved, or used therapeutically for this condition.",
    "off-label use": "A drug is used off-label for a disease, meaning the drug is used to treat a condition for which it is not officially approved but has evidence of clinical benefit.",
    "synergistic interaction": "Two drugs have a synergistic interaction, meaning they enhance each others therapeutic effect when used together. The combined effect is greater than the sum of their individual effects.",
    "associated with": "A gene/protein is associated with a disease or an effect/phenotype, meaning the gene/protein plays a role in, contributes to, is implicated in, or is linked to the pathology of the disease or the manifestation of the phenotype.",
    "parent-child": "A hierarchical ontological relationship where one entity is a parent category (broader concept) and the other is a child (narrower, more specific subtype). The child entity is a subtype of or belongs to the parent entity.",
    "phenotype absent": "A disease does NOT present, exhibit, or manifest a specific effect/phenotype. Patients with this disease typically do not show this symptom, sign, or phenotypic feature.",
    "phenotype present": "A disease presents, exhibits, or manifests a specific effect/phenotype. Patients with this disease typically show this symptom, sign, clinical feature, or phenotypic manifestation.",
    "side effect": "A drug causes, induces, or is associated with an adverse effect/phenotype as a side effect. The effect/phenotype is an unwanted, unintended, or adverse reaction that occurs when taking this drug.",
    "interacts with": "A gene/protein functionally interacts with, participates in, or is involved in a biological process, pathway, cellular component, molecular function, or exposure. Also covers interactions between biological processes, cellular components, molecular functions, and exposures.",
    "linked to": "A disease is epidemiologically or causally linked to an environmental or lifestyle exposure. The exposure is a risk factor, contributing factor, or causative agent associated with the disease.",
    "expression present": "A gene/protein is expressed in, actively produced in, or found in a specific anatomical structure or tissue. The anatomy is a site where this gene/protein is expressed, transcribed, or translated.",
    "expression absent": "A gene/protein is NOT expressed in, not detected in, or absent from a specific anatomical structure or tissue. The anatomy is a site where this gene/protein is not found or not actively produced.",
}


# ---------------------------------------------------------------------------
# Node description strategy (per type)
# ---------------------------------------------------------------------------

DESC_STRATEGY = {
    "drug": {
        "primary": ["description", "mechanism_of_action", "indication"],
        "fallback": ["pharmacodynamics", "category", "group"],
        "total_limit": 3000,
    },
    "gene/protein": {
        "primary": ["summary"],
        "fallback": ["name"],
        "total_limit": 3000,
    },
    "disease": {
        "primary": ["mondo_definition", "umls_description", "orphanet_clinical_description"],
        "fallback": ["mayo_symptoms", "orphanet_definition", "mondo_name"],
        "total_limit": 3000,
    },
    "pathway": {
        "primary": ["summation"],
        "fallback": ["displayName"],
        "total_limit": 3000,
    },
}


def _is_real(val) -> bool:
    if not val:
        return False
    s = str(val).strip()
    return len(s) > 5 and s.lower() not in ("nan", "none")


def _get_node_description(info: dict, node_type: str) -> str:
    details = info.get("details") or {}
    if not details:
        return ""
    config = DESC_STRATEGY.get(node_type)
    if not config:
        return ""
    parts: List[str] = []
    total_len = 0
    for key in config["primary"]:
        val = details.get(key, "")
        if not _is_real(val):
            continue
        val_str = str(val).strip()
        if total_len + len(val_str) > config["total_limit"]:
            break
        parts.append(val_str)
        total_len += len(val_str)
    if parts:
        return " ".join(parts)
    for key in config["fallback"]:
        val = details.get(key, "")
        if _is_real(val):
            return str(val).strip()[: config["total_limit"]]
    return ""


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_igraph(
    primekg_data_dir: Path = DEFAULT_PRIMEKG_DIR,
    cache_path: Path = CACHE_DIR / "graph.pickle",
) -> "igraph.Graph":
    """Load PrimeKG tensors -> undirected igraph.Graph, save pickle."""
    import igraph as ig
    import torch

    primekg_data_dir = Path(primekg_data_dir)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    edge_index = torch.load(primekg_data_dir / "edge_index.pt", weights_only=True)
    edge_types = torch.load(primekg_data_dir / "edge_types.pt", weights_only=True)
    node_types = torch.load(primekg_data_dir / "node_types.pt", weights_only=True)
    with open(primekg_data_dir / "edge_type_dict.pkl", "rb") as f:
        edge_type_dict = pickle.load(f)
    with open(primekg_data_dir / "node_type_dict.pkl", "rb") as f:
        node_type_dict = pickle.load(f)
    with open(primekg_data_dir / "node_info.pkl", "rb") as f:
        node_info = pickle.load(f)

    num_nodes = int(edge_index.max()) + 1

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    rels = edge_types.tolist()

    seen: set = set()
    u_list: list = []
    v_list: list = []
    r_list: list = []
    for s, d, r in zip(src, dst, rels):
        key = (s, d, r) if s <= d else (d, s, r)
        if key in seen:
            continue
        seen.add(key)
        u_list.append(key[0])
        v_list.append(key[1])
        r_list.append(r)

    g = ig.Graph(n=num_nodes, directed=False)
    g.vs["name"] = [node_info[i].get("name", f"Node_{i}") for i in range(num_nodes)]
    g.vs["type"] = [node_type_dict[int(node_types[i].item())] for i in range(num_nodes)]
    g.add_edges(list(zip(u_list, v_list)))
    g.es["relation"] = [edge_type_dict[r] for r in r_list]
    g.es["rel_id"] = r_list
    g.write_pickle(str(cache_path))
    return g


def build_node_text(
    primekg_data_dir: Path = DEFAULT_PRIMEKG_DIR,
    cache_path: Path = CACHE_DIR / "node_text.json",
) -> Dict[int, str]:
    """Produce per-node text: DESC_STRATEGY description if available, else node name."""
    primekg_data_dir = Path(primekg_data_dir)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(primekg_data_dir / "node_info.pkl", "rb") as f:
        node_info = pickle.load(f)

    node_text: Dict[int, str] = {}
    desc_coverage = 0
    for idx in range(len(node_info)):
        info = node_info[idx]
        name = info.get("name", f"Node_{idx}")
        node_type = info.get("type", "")
        desc = _get_node_description(info, node_type)
        if desc:
            node_text[idx] = f"{name}. {desc}"
            desc_coverage += 1
        else:
            node_text[idx] = name

    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in node_text.items()}, f)
    print(f"node_text: {len(node_text):,} total, {desc_coverage:,} with description "
          f"({desc_coverage / len(node_text):.1%})")
    return node_text


def _tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def build_bm25_indexes(
    node_text: Dict[int, str] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Build BM25 indexes (length-agnostic, b=0) for nodes and relations.

    Saves:
      cache_dir/bm25_node_nolen.pkl   (b=0; node texts)
      cache_dir/bm25_rel_nolen.pkl    (b=0; 18 relation descriptions)
    """
    from rank_bm25 import BM25Okapi

    if node_text is None:
        node_text = load_node_text()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = len(node_text)
    node_ids = list(range(n))
    node_tokens = [_tokenize_for_bm25(node_text[i]) for i in node_ids]
    bm25_node = BM25Okapi(node_tokens, b=0.0)
    with open(cache_dir / "bm25_node_nolen.pkl", "wb") as f:
        pickle.dump({"node_ids": node_ids, "tokens": node_tokens, "bm25": bm25_node, "b": 0.0}, f)

    rel_names = list(RELATION_DESCRIPTIONS.keys())
    rel_tokens = [_tokenize_for_bm25(f"{name} {RELATION_DESCRIPTIONS[name]}") for name in rel_names]
    bm25_rel = BM25Okapi(rel_tokens, b=0.0)
    with open(cache_dir / "bm25_rel_nolen.pkl", "wb") as f:
        pickle.dump({"rel_names": rel_names, "tokens": rel_tokens, "bm25": bm25_rel, "b": 0.0}, f)

    print(f"bm25_node (b=0): {n:,} docs")
    print(f"bm25_rel  (b=0): {len(rel_names)} docs")
    return {"bm25_node": bm25_node, "bm25_rel": bm25_rel,
            "node_ids": node_ids, "rel_names": rel_names}


def load_bm25_node(cache_path: Path = CACHE_DIR / "bm25_node_nolen.pkl"):
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_bm25_rel(cache_path: Path = CACHE_DIR / "bm25_rel_nolen.pkl"):
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_graph(cache_path: Path = CACHE_DIR / "graph.pickle") -> "igraph.Graph":
    import igraph as ig
    return ig.Graph.Read_Pickle(str(cache_path))


def load_node_text(cache_path: Path = CACHE_DIR / "node_text.json") -> Dict[int, str]:
    with open(cache_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
