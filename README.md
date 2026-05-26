# AdaPath

Adaptive path-finding over a biomedical knowledge graph for question
answering. Builds a per-query path bank with query-conditioned BM25 edge
weighting + Personalized PageRank + Yen's K-shortest paths.

## Layout

```
AdaPath/
├── data/
│   ├── primekg/            (auto-populated on first run)
│   └── biostrat_qa/        BioStrat-QA dataset
│       ├── train.jsonl     (2491)
│       ├── dev.jsonl       (898)
│       ├── test.jsonl      (1179)
│       └── templates.jsonl (60 metapaths)
├── pathbank/               Path-bank construction (BM25-weighted PPR + Yen's K-shortest)
├── inference/              AdaPath inference (LLM-guided path-finding)
└── eval/                   Exact-match scoring
```

## Install

```bash
pip install -r requirements.txt
```

## Build the path bank

```bash
python -m pathbank.build_pathbank \
    --input  data/biostrat_qa/test.jsonl \
    --output data/pathbank/test_paths.jsonl
```

The first invocation auto-prepares the knowledge graph and BM25 indexes
under `data/primekg/processed/` and `pathbank/cache/`; subsequent runs reuse
the caches.

## Run AdaPath inference

```bash
python -m inference.run_inference \
    --input               data/biostrat_qa/test.jsonl \
    --triplets_file       data/biostrat_qa/test.jsonl \
    --train_pathbank_dir  data/pathbank \
    --LLM_type            "meta-llama/Llama-3.1-70B-Instruct" \
    --output_dir          results/adapath
```

The LLM runs locally via HuggingFace `transformers`
(`AutoModelForCausalLM`, sequential). Set the GPU via the `ADAPATH_DEVICE`
env var (default `cuda:0`) and the dtype via `ADAPATH_DTYPE`
(`bfloat16` by default).

## Evaluate

```bash
python -m eval.metrics \
    --result_jsonl   path/to/results.jsonl \
    --test_jsonl     data/biostrat_qa/test.jsonl \
    --node_info_pkl  data/primekg/processed/node_info.pkl
```

## Dataset

BioStrat-QA covers 1-, 2-, and 3-hop biomedical questions built from 60
hand-curated metapaths over PrimeKG. Each record provides three query
variants per (topic, answer) pair: `explicit_query`, `implicit_query`,
`bare_query`.

## License

See `LICENSE`.
