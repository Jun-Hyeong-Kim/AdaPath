"""
PrimeKG Client — wraps STaRK's PrimeSKB to provide the interface that AdaPath needs.

Key design decisions:
- PrimeKG is treated as undirected (SKB applies to_undirected()).
- All relations are returned as a flat list (no head/tail distinction).
- Entity IDs are integer node indices.
"""

from collections import defaultdict
from rank_bm25 import BM25Okapi


class PrimeKGClient:
    def __init__(self, skb):
        self.skb = skb
        self._relation_types = skb.rel_type_lst()
        self._build_name_index()

    # ------------------------------------------------------------------ #
    #  Name index for entity linking                                      #
    # ------------------------------------------------------------------ #

    def _build_name_index(self):
        """Build lowercase name -> [node_idx, ...] mapping."""
        self.name_to_ids = defaultdict(list)
        self._all_names = []       # parallel to node indices
        self._all_names_lower = [] # for BM25 search

        for idx in range(len(self.skb)):
            name = self.skb.node_info[idx].get('name', '')
            self.name_to_ids[name.lower().strip()].append(idx)
            self._all_names.append(name)
            self._all_names_lower.append(name.lower().strip())

    # ------------------------------------------------------------------ #
    #  Graph query methods                                                #
    # ------------------------------------------------------------------ #

    def get_all_relations_of_entity(self, node_idx: int) -> list:
        """Return relation type names for which this node has at least one neighbor."""
        if not hasattr(self, '_rel_cache'):
            self._rel_cache = {}
        if node_idx in self._rel_cache:
            return self._rel_cache[node_idx]
        relations = []
        for rel_type in self._relation_types:
            neighbors = self.skb.get_neighbor_nodes(node_idx, rel_type)
            if len(neighbors) > 0:
                relations.append(rel_type)
        self._rel_cache[node_idx] = relations
        return relations

    def get_neighbors(self, node_idx: int, relation: str):
        """Return (id_list, name_list) of neighbors connected by the given relation."""
        neighbor_ids = self.skb.get_neighbor_nodes(node_idx, relation)
        neighbor_names = [
            self.skb.node_info[nid].get('name', f'Node_{nid}')
            for nid in neighbor_ids
        ]
        return neighbor_ids, neighbor_names

    # ------------------------------------------------------------------ #
    #  Node info methods                                                  #
    # ------------------------------------------------------------------ #

    def idx_to_name(self, idx: int) -> str:
        return self.skb.node_info[idx].get('name', f'Node_{idx}')

    def idx_to_type(self, idx: int) -> str:
        return self.skb.get_node_type_by_id(idx)

    def get_doc_info(self, idx: int) -> str:
        return self.skb.get_doc_info(idx, add_rel=False)

    # ------------------------------------------------------------------ #
    #  Entity linking methods                                             #
    # ------------------------------------------------------------------ #

    def name_to_idx(self, name: str) -> list:
        """Exact case-insensitive name match. Returns list of node indices."""
        return self.name_to_ids.get(name.lower().strip(), [])

    def search_entity(self, name: str, top_k: int = 5) -> list:
        """
        Fuzzy entity search. Returns list of (node_idx, node_name) tuples.

        Strategy:
        1. Exact match (case-insensitive)
        2. Substring containment match
        3. BM25 fallback over all node names
        """
        query = name.lower().strip()

        # 1. Exact match
        exact = self.name_to_ids.get(query, [])
        if exact:
            return [(idx, self.idx_to_name(idx)) for idx in exact[:top_k]]

        # 2. Substring match
        substring_matches = []
        for idx, n in enumerate(self._all_names_lower):
            if query in n or n in query:
                substring_matches.append((idx, self._all_names[idx]))
            if len(substring_matches) >= top_k * 2:
                break
        if substring_matches:
            # Prefer shorter names (more specific matches)
            substring_matches.sort(key=lambda x: abs(len(x[1]) - len(name)))
            return substring_matches[:top_k]

        # 3. BM25 fallback (search over all node names)
        tokenized_corpus = [n.split() for n in self._all_names_lower]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        results = [
            (int(idx), self._all_names[idx])
            for idx in top_indices
            if scores[idx] > 0
        ]
        return results

    # ------------------------------------------------------------------ #
    #  Metadata                                                           #
    # ------------------------------------------------------------------ #

    @property
    def num_nodes(self) -> int:
        return len(self.skb)

    @property
    def relation_types(self) -> list:
        return self._relation_types
