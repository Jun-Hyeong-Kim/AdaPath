"""Prepare PrimeKG processed tensors from STaRK SKB.

On first call:
  1. Download PrimeKG via `stark_qa.load_skb("prime")` (auto-fetched by stark_qa).
  2. Extract edge_index, edge_types, node_types, type-name dicts, node_info.
  3. Save as torch tensors + pickles under data/primekg/processed/.
  4. Build BM25 indexes via pathbank.embedding_cache.

Subsequent calls are no-ops (cache hit).
"""

from __future__ import annotations

import pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PRIMEKG_DIR = _ROOT / "data" / "primekg" / "processed"


def _has_all_tensors() -> bool:
    needed = [
        "edge_index.pt", "edge_types.pt", "node_types.pt",
        "edge_type_dict.pkl", "node_type_dict.pkl", "node_info.pkl",
    ]
    return all((PRIMEKG_DIR / f).exists() for f in needed)


def extract_primekg_tensors(out_dir: Path = PRIMEKG_DIR) -> None:
    """Run STaRK SKB load (auto-downloads) and dump tensors / pickles."""
    import torch
    from stark_qa import load_skb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PrimeKG via stark_qa.load_skb('prime') (auto-download if needed)...")
    skb = load_skb("prime")

    print("Extracting tensors...")
    torch.save(skb.edge_index, out_dir / "edge_index.pt")
    torch.save(skb.edge_types, out_dir / "edge_types.pt")
    torch.save(skb.node_types, out_dir / "node_types.pt")

    # type-id -> name dicts
    edge_type_dict = {i: name for i, name in enumerate(skb.rel_type_lst())}
    node_type_dict = {i: name for i, name in enumerate(skb.node_type_lst())}
    with open(out_dir / "edge_type_dict.pkl", "wb") as f:
        pickle.dump(edge_type_dict, f)
    with open(out_dir / "node_type_dict.pkl", "wb") as f:
        pickle.dump(node_type_dict, f)

    # node_info: {idx: {name, type, details}}
    node_info = {}
    n = skb.edge_index.max().item() + 1
    for i in range(n):
        info = skb.node_info[i] if hasattr(skb, "node_info") else {}
        node_info[i] = info if isinstance(info, dict) else {}
    with open(out_dir / "node_info.pkl", "wb") as f:
        pickle.dump(node_info, f)

    print(f"Saved processed tensors under {out_dir}/")
    print(f"  Nodes: {n:,}  Edges: {skb.edge_index.shape[1]:,}  "
          f"Node types: {len(node_type_dict)}  Rel types: {len(edge_type_dict)}")


def ensure_kg_ready() -> Path:
    """Idempotent: build PrimeKG processed tensors + BM25 indexes if missing."""
    from pathbank.embedding_cache import (
        CACHE_DIR, build_igraph, build_node_text, build_bm25_indexes,
    )

    if not _has_all_tensors():
        extract_primekg_tensors()
    else:
        print(f"PrimeKG tensors already present at {PRIMEKG_DIR}/")

    if not (CACHE_DIR / "graph.pickle").exists():
        print("Building igraph cache...")
        build_igraph()
    if not (CACHE_DIR / "node_text.json").exists():
        print("Building node_text cache...")
        build_node_text()
    if not (CACHE_DIR / "bm25_node_nolen.pkl").exists():
        print("Building BM25 indexes...")
        build_bm25_indexes()

    print("KG + caches ready.")
    return PRIMEKG_DIR


if __name__ == "__main__":
    ensure_kg_ready()
