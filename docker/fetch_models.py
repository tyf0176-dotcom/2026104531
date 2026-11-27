"""Fetch redistributable model snapshots when the image is built."""
from pathlib import Path
from hashlib import sha256
from huggingface_hub import hf_hub_download, snapshot_download
import shutil

ROOT = Path("/models")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                  revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                  "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"],
                  local_dir=ROOT / "encoder")
name = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
for filename in (name, "LICENSE", "README.md"):
    hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", filename,
                    revision="9217f5db79a29953eb74d5343926648285ec7e67", local_dir=ROOT / "generator")
with (ROOT / "generator" / name).open("rb") as f:
    digest = sha256()
    for block in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(block)
assert digest.hexdigest() == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
for folder in (ROOT / "encoder", ROOT / "generator"):
    shutil.rmtree(folder / ".cache", ignore_errors=True)
