"""Fixed LangGraph file search workflow. Run --help for benchmark and RAG options."""
import argparse
from collections import defaultdict
import importlib
import os
import json
from pathlib import Path
import time
import sys
from typing import TypedDict
from urllib.request import Request, urlopen

# Use one BLAS thread by default for reproducible per-query timings.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = os.environ.get("LAB_BLAS_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langgraph.graph import StateGraph, START, END
import numpy as np

from benchmark.third_party.beir.topk import top_k_accuracy

ROOT = Path(__file__).resolve().parents[1] / "data"


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf8").splitlines()]


class State(TypedDict, total=False):
    i: int
    query_vector: object
    embed_ms: float
    chunk_ids: list
    file_chunks: list
    search_ms: float
    llm_ms: float
    answer: str
    completion_tokens: int


def workflow(index, params, queries, vectors, chunks, generate, url, encoder):

    def embed(s):
        start = time.perf_counter_ns()
        query = encoder([queries[s["i"]]["question"]])[0] if encoder else vectors[s["i"]]
        return {"query_vector": query, "embed_ms": (time.perf_counter_ns() - start) / 1e6}

    def search(s):
        query = s["query_vector"]
        start = time.perf_counter_ns()
        found = list(index.search(query, 50, **params))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if (len(found) > 50 or len(set(found)) != len(found) or
            any(not isinstance(j, (int, np.integer)) or not 0 <= j < len(chunks) for j in found)):
            raise ValueError("ANN must return distinct, valid, ranked chunk IDs")
        return {"chunk_ids": [int(j) for j in found], "search_ms": elapsed}

    def rank_files(s):
        ranked = {}
        for j in s["chunk_ids"]:
            ranked.setdefault(chunks[j]["file"], j)
        return {"file_chunks": list(ranked.values())[:5]}

    def answer(s):
        facts = " ".join(queries[s["i"]]["question"].split()[:160])
        evidence = "\n".join("[{}] {}".format(chunks[j]["file"], " ".join(chunks[j]["text"].split()[:80]))
                             for j in s["file_chunks"][:3])
        body = dict(model="rag-lab", temperature=0, seed=42, max_tokens=64, messages=[
            dict(role="system", content="Use only the given excerpts. Give one reason to consult a file, citing its bracketed ID. If insufficient, say so."),
            dict(role="user", content=f"Case description:\n{facts}\nRetrieved excerpts:\n{evidence}")])
        req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.perf_counter_ns()
        with urlopen(req, timeout=180) as resp:
            reply = json.load(resp)
        return {"answer": reply["choices"][0]["message"]["content"], "completion_tokens": reply["usage"]["completion_tokens"],
                "llm_ms": (time.perf_counter_ns() - start) / 1e6}

    graph = StateGraph(State)
    graph.add_node("ann", search)
    graph.add_node("rank_files", rank_files)
    graph.add_node("embed", embed)
    graph.add_edge(START, "embed")
    graph.add_edge("embed", "ann")
    graph.add_edge("ann", "rank_files")
    if generate:
        graph.add_node("explain", answer)
        graph.add_edge("rank_files", "explain")
        graph.add_edge("explain", END)
    else:
        graph.add_edge("rank_files", END)
    return graph.compile()


def parameters(raw):
    """Accept a JSON object or comma-separated integer settings on any shell."""
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


def measure(index, query_params, queries, chunks, q, exact, qrels, ids, online, encoder):
    app = workflow(index, query_params, queries, q, chunks, online,
                   "http://127.0.0.1:8080/v1/chat/completions", encoder)
    if online:
        ids = [ids[0], ids[len(ids) // 2], ids[-1]]
    for i in ids[:1 if online else 5]:
        app.invoke({"i": i})

    searches, elapsed, recalls, hits = [], [], [], []
    encodings, generations = [], []
    for rep in range(1 if online else 3):
        ranked, judged = {}, {}
        for i in np.random.default_rng(42 + rep).permutation(ids):
            start = time.perf_counter_ns()
            state = app.invoke({"i": int(i)})
            end_to_end = (time.perf_counter_ns() - start) / 1e6
            files = [chunks[j]["file"] for j in state["file_chunks"]]
            qid = queries[i]["id"]
            ranked[qid] = {doc: -rank for rank, doc in enumerate(files)}
            judged[qid] = qrels[qid]
            recalls.append(len(set(state["chunk_ids"][:10]) & set(exact[i, :10])) / 10)
            searches.append(state["search_ms"])
            elapsed.append(end_to_end)
            if online:
                encodings.append(state["embed_ms"])
                generations.append(state["llm_ms"])
                print(json.dumps(dict(type="query", query=qid, files=files, relevant=list(judged[qid]),
                                      evidence_chunk_ids=state["file_chunks"][:3], embed_ms=state["embed_ms"],
                                      search_ms=state["search_ms"], llm_ms=state["llm_ms"], workflow_ms=end_to_end,
                                      tokens=state["completion_tokens"], answer=state["answer"]), ensure_ascii=False))
        hits.append(top_k_accuracy(judged, ranked, [5])["Accuracy@5"])
    result = dict(queries=len(ids), repetitions=1 if online else 3, ann_recall10=np.mean(recalls),
                  file_hit5=sum(hits) / len(hits), search_p50_ms=np.median(searches),
                  search_p95_ms=np.percentile(searches, 95), workflow_p50_ms=np.median(elapsed))
    if online:
        result.update(embed_p50_ms=np.median(encodings), llm_p50_ms=np.median(generations))
    return result


def main():
    p = argparse.ArgumentParser(description="CPU vector-index benchmark")
    p.add_argument("mode", choices=["embed", "search", "online"])
    p.add_argument("index", choices=["exact", "graph", "cluster"], nargs="?", default="exact")
    p.add_argument("--build", action="append", type=parameters, help="repeat for different construction settings")
    p.add_argument("--search", action="append", type=parameters, help="repeat for different query settings")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    args = p.parse_args()
    if args.mode == "embed":
        from benchmark.encode import main as encode_corpus
        sys.argv = [sys.argv[0]]
        encode_corpus()
        return
    queries, chunks = rows("questions.jsonl"), rows("chunks.jsonl")
    qrels = defaultdict(dict)
    for r in rows("test.jsonl"):
        if r["score"] > 0:
            qrels[r["query-id"]][r["corpus-id"]] = r["score"]
    x, q, exact = [np.load(ROOT / (name + ".npy")) for name in ("documents", "questions", "exact_top50")]
    ids = [i for i, row in enumerate(queries) if (int(row["id"]) % 5 == 0) == (args.split == "dev")]
    online = args.mode == "online"
    encoder = None
    if online:
        from benchmark.encode import load_encoder
        encoder = load_encoder()
    cls = importlib.import_module("indexes." + args.index).Index
    for build_params in args.build or [{}]:
        start = time.perf_counter_ns()
        index = cls(x, **build_params)
        build_ms = (time.perf_counter_ns() - start) / 1e6
        for search_params in args.search or [{}]:
            result = measure(index, search_params, queries, chunks, q, exact, qrels, ids, online, encoder)
            print(json.dumps(dict(type="summary", mode=args.mode, index=args.index, split=args.split,
                                  build=build_params, search=search_params, build_ms=build_ms, **result)))


if __name__ == "__main__":
    main()
