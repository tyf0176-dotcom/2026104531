"""Implement one graph (for example HNSW) index. You may change all parameter names."""


class Index:
    def __init__(self, vectors, M=16, efConstruction=100):
        """Build once from N x 384 normalized float32 vectors."""
        raise NotImplementedError("Implement your offline index construction")

    def search(self, query, k, efSearch=128):
        """Return at most k distinct chunk row IDs ranked from best to worst."""
        raise NotImplementedError("Implement your online ANN search")
