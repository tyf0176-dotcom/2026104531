"""Host-side fallback for `bench search`, for machines without Docker.

The course container (see compose.yaml) is the authoritative measurement
environment; this script reproduces the same numbers from the same frozen
arrays so the index code can be developed and swept outside the container.
It mirrors benchmark/runner.py: same dev/test split, same k=50 call, same
Recall@10 against data/exact_top50.npy, same Hit@5 through the BEIR helper,
same warm-up (5 queries) and same 3 timed repetitions.

Usage (from the repository root):
    python tools/offline_bench.py graph  --split dev  --build M=16,efConstruction=100 \
        --search efSearch=64 --search efSearch=128 > results/graph-dev.jsonl
    python tools/offline_bench.py cluster --split dev --build nlist=64,m=16,nbits=4 \
        --search nprobe=1 --search nprobe=4
"""

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from benchmark.third_party.beir.topk import top_k_accuracy  # noqa: E402


def rows(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf8").splitlines()]


def parameters(raw):
    try:
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        result = {}
        for field in raw.split(","):
            key, value = field.split("=", 1)
            result[key.strip()] = int(value)
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a JSON object or name=integer") from exc


def main():
    p = argparse.ArgumentParser(description="host-side ANN benchmark")
    p.add_argument("index", choices=["exact", "graph", "cluster"])
    p.add_argument("--build", action="append", type=parameters)
    p.add_argument("--search", action="append", type=parameters)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--detail", action="store_true", help="also print per-query rows")
    args = p.parse_args()

    queries, chunks = rows("questions.jsonl"), rows("chunks.jsonl")
    qrels = defaultdict(dict)
    for r in rows("test.jsonl"):
        if r["score"] > 0:
            qrels[r["query-id"]][r["corpus-id"]] = r["score"]
    x, q, exact = [np.load(DATA / (name + ".npy")) for name in ("documents", "questions", "exact_top50")]
    ids = [i for i, row in enumerate(queries) if (int(row["id"]) % 5 == 0) == (args.split == "dev")]

    cls = importlib.import_module("indexes." + args.index).Index
    for build_params in args.build or [{}]:
        start = time.perf_counter_ns()
        index = cls(x, **build_params)
        build_ms = (time.perf_counter_ns() - start) / 1e6
        extra = index.size_bytes() if hasattr(index, "size_bytes") else 0

        for search_params in args.search or [{}]:
            for i in ids[:5]:  # warm-up, mirrors runner.py
                index.search(q[i], 50, **search_params)

            searches, recalls, hits = [], [], []
            for rep in range(args.reps):
                ranked, judged = {}, {}
                for i in np.random.default_rng(42 + rep).permutation(ids):
                    begin = time.perf_counter_ns()
                    found = list(index.search(q[int(i)], 50, **search_params))
                    searches.append((time.perf_counter_ns() - begin) / 1e6)
                    if len(found) > 50 or len(set(found)) != len(found):
                        raise ValueError("ANN must return distinct, <=50 ids")
                    files = []
                    for j in found:
                        if chunks[j]["file"] not in files:
                            files.append(chunks[j]["file"])
                    ranked[queries[int(i)]["id"]] = {d: -r for r, d in enumerate(files[:5])}
                    judged[queries[int(i)]["id"]] = qrels[queries[int(i)]["id"]]
                    recalls.append(len(set(found[:10]) & set(exact[int(i), :10])) / 10)
                    if args.detail:
                        print(json.dumps(dict(type="query", query=queries[int(i)]["id"], files=files[:5],
                                              relevant=sorted(judged[queries[int(i)]["id"]]),
                                              chunk_ids=found[:10]), ensure_ascii=False))
                hits.append(top_k_accuracy(judged, ranked, [5])["Accuracy@5"])

            result = dict(
                type="summary",
                mode="search",
                index=args.index,
                split=args.split,
                build=build_params,
                search=search_params,
                build_ms=build_ms,
                queries=len(ids),
                repetitions=args.reps,
                ann_recall10=float(np.mean(recalls)),
                file_hit5=float(sum(hits) / len(hits)),
                search_p50_ms=float(np.median(searches)),
                search_p95_ms=float(np.percentile(searches, 95)),
                index_bytes=int(extra),
                host="darwin-arm64-host",
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
