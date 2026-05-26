"""Exact-match scoring for path-finding results.

Parses the answer enclosed in `{...}` from the model response and matches it
against the set of valid answer names (multi-answer expansion via
`valid_answer_ids` resolved through PrimeKG node_info).
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_answer(text: str) -> str:
    """Pull the first `{...}` from the response. If the first brace is yes/no,
    use the second. Returns "" if no brace found."""
    if not text:
        return ""
    matches = re.findall(r"\{([^}]+)\}", text)
    if len(matches) >= 2 and matches[0].strip().lower() in ("yes", "no"):
        return matches[1].strip()
    if matches:
        return matches[0].strip()
    return ""


def _norm(s: str) -> str:
    return s.strip().replace(" ", "").lower()


def em_match(response: str, ans_name: str) -> bool:
    if not response or not ans_name:
        return False
    return _norm(response) == _norm(ans_name) or \
           _norm(response) in _norm(ans_name) or \
           _norm(ans_name) in _norm(response)


def valid_names_for(test_rec: dict, node_info: dict) -> List[str]:
    """Resolve `valid_answer_ids` -> list of node names; fall back to canonical."""
    vids = test_rec.get("valid_answer_ids") or test_rec.get("answer_ids") or []
    names: List[str] = []
    for vid in vids:
        info = node_info.get(vid)
        if isinstance(info, dict) and info.get("name"):
            names.append(info["name"])
    if not names:
        canonical = (test_rec.get("answer_entity") or {}).get("name", "")
        if canonical:
            names = [canonical]
    return names


def compute_em_per_hop(
    result_records: List[dict],
    test_by_data_id: Dict,
    node_info: Dict,
) -> Dict[int, List[int]]:
    """Bucket by `num_hops`; returns {hop: [correct, total]} for hop in [1, 2, 3]."""
    results = {h: [0, 0] for h in [1, 2, 3]}
    for r in result_records:
        td = test_by_data_id.get(r.get("data_id"))
        if not td:
            continue
        h = td.get("num_hops")
        if h not in [1, 2, 3]:
            continue
        results[h][1] += 1
        extracted = extract_answer(r.get("results", ""))
        text = extracted if extracted else r.get("results", "")
        names = valid_names_for(td, node_info)
        if any(em_match(text, n) for n in names):
            results[h][0] += 1
    return results


def summarize(em_dict: Dict[int, List[int]]) -> Tuple[float, Dict[int, float]]:
    """Convert {hop: [correct, total]} -> (overall_pct, {hop: pct})."""
    per_hop = {h: (100 * c / t) if t else 0.0 for h, (c, t) in em_dict.items()}
    tot_c = sum(c for c, _ in em_dict.values())
    tot_t = sum(t for _, t in em_dict.values())
    overall = (100 * tot_c / tot_t) if tot_t else 0.0
    return overall, per_hop


# ---------------------------------------------------------------------------
# CLI: python -m eval.metrics --result_jsonl ... --test_jsonl ...
# ---------------------------------------------------------------------------

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True,
                    help="Reference test jsonl (BioStrat-QA test/dev split).")
    ap.add_argument("--node_info_pkl", required=True,
                    help="data/primekg/processed/node_info.pkl")
    args = ap.parse_args()

    test_by_id = {}
    with open(args.test_jsonl) as f:
        for line in f:
            r = json.loads(line)
            test_by_id[r["data_id"]] = r

    with open(args.node_info_pkl, "rb") as f:
        node_info = pickle.load(f)

    results = []
    with open(args.result_jsonl) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                continue

    em = compute_em_per_hop(results, test_by_id, node_info)
    overall, per_hop = summarize(em)
    print(f"Records evaluated: {sum(t for _, t in em.values())}")
    for h in (1, 2, 3):
        c, t = em[h]
        print(f"  {h}-hop  {c}/{t}  ({per_hop[h]:.2f}%)")
    print(f"  Overall {overall:.2f}%")


if __name__ == "__main__":
    _main()
