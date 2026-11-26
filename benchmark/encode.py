"""Encode the fixed English benchmark on CPU using the pinned MiniLM encoder."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1] / "data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def rows(filename):
    return [json.loads(line) for line in (ROOT / filename).read_text(encoding="utf8").splitlines()]


def load_encoder(model_dir=None, threads=None):
    torch.set_num_threads(int(os.environ.get("LAB_MODEL_THREADS", "4")) if threads is None else threads)
    src = model_dir or os.environ.get("LAB_ENCODER", MODEL)
    tok = AutoTokenizer.from_pretrained(src, revision=REVISION)
    model = AutoModel.from_pretrained(src, revision=REVISION).cpu().eval()

    @torch.inference_mode()
    def encode(texts):
        batches = []
        for i in range(0, len(texts), 32):
            inputs = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokens = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            mean = (tokens * mask).sum(1) / mask.sum(1)
            batches.append(torch.nn.functional.normalize(mean, dim=1).numpy())
        return np.concatenate(batches).astype("float32")

    return encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir")
    p.add_argument("--threads", type=int, default=int(os.environ.get("LAB_MODEL_THREADS", "4")))
    args = p.parse_args()
    encode = load_encoder(args.model_dir, args.threads)
    docs = rows("chunks.jsonl")
    encode(["Warm up the encoder."])
    texts = [r["text"] for r in docs]
    start = perf_counter()
    encoded = encode(texts)
    seconds = perf_counter() - start
    print(json.dumps(dict(stage="document_embedding", chunks=len(encoded), dimensions=encoded.shape[1],
                          batch_size=32, threads=args.threads, doc_embed_s=seconds)))


if __name__ == "__main__":
    main()
