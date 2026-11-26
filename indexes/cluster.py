"""Implement one cluster (for example IVF-PQ) index. You may change all parameter names."""


class Index:
    def __init__(self, vectors, nlist=64, m=16, nbits=4):
        """Build once from N x 384 normalized float32 vectors."""
        raise NotImplementedError("Implement your offline index construction")

    def search(self, query, k, nprobe=4):
        """Return at most k distinct chunk row IDs ranked from best to worst."""
        raise NotImplementedError("Implement your online ANN search")
