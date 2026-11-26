"""Materialize each benchmark document and freeze chunk IDs, queries and file-level labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
CHECKS = {
    "corpus.jsonl": "1df8d719a1afa050abf0d92050c6240e8bf5c885f73c895e76b3c34af7964dcd",
    "queries.jsonl": "fb14f0569f9a207eb0ef1285ecc5db71ccaebd99ff9d93ffa8c13a735848c179",
    "test.jsonl": "b92509f60a297271fae89a62a8a43ea58a3a49b4e45e276caacdba2a46c5d47e",
}


def read(name):
    p = ROOT / name
    if hashlib.sha256(p.read_bytes()).hexdigest() != CHECKS[name]:
        raise ValueError("Unexpected input: " + name)
    return [json.loads(line) for line in p.read_text(encoding="utf8").splitlines()]


def write(name, rows):
    with (ROOT / name).open("w", encoding="utf8") as output:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=output)


if __name__ == "__main__":
    docs, queries, labels = (read(name) for name in CHECKS)
    folder = ROOT / "files"
    folder.mkdir(exist_ok=True)
    chunks = []
    for doc in docs:
        name = doc["_id"]
        (folder / (name + ".txt")).write_text(doc["text"], encoding="utf8")
        words = doc["text"].split()
        for start in range(0, len(words), 108):
            end = min(start + 128, len(words))
            chunks.append(dict(file=name, id=len(chunks), text=" ".join(words[start:end])))
            if end == len(words):
                break
    write("chunks.jsonl", chunks)
    write("questions.jsonl", [dict(id=q["_id"], question=q["text"]) for q in queries])
    print(len(docs), "files", len(chunks), "chunks", len(queries), "queries", len(labels), "qrels")
