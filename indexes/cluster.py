"""IVF cluster index over the frozen 384-d chunk vectors.

NumPy only. Coarse k-means builds the inverted lists. Each residual is then
either stored as is (``nbits=0`` -> IVF-Flat) or split into ``m`` sub-vectors
and encoded with its own ``2**nbits``-entry codebook (``nbits>0`` -> IVF-PQ).
Search scans the ``nprobe`` closest lists and ranks their members with either
an exact inner product or a code look-up table.

Construction parameters
    nlist   number of coarse clusters (inverted lists)
    nbits   bits per sub-vector code; 0 keeps the raw residuals (IVF-Flat)
    m       number of sub-vectors per residual (only used when nbits > 0)
    seed    seed for k-means (fixed => reproducible build)
    iters   k-means iterations for the coarse and the PQ stage
Query parameter
    nprobe  number of closest clusters scanned per query
"""

import numpy as np


def _l2_sq(x, c):
    """Squared L2 distance matrix between rows of x and rows of c."""
    return (
        np.sum(x * x, axis=1)[:, None]
        + np.sum(c * c, axis=1)[None, :]
        - 2.0 * (x @ c.T)
    )


def _kmeans(x, k, iters, rng):
    """Plain Lloyd k-means with k-means++ seeding on the given data."""
    n = x.shape[0]
    k = min(k, n)
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[int(rng.integers(n))]
    closest = np.maximum(_l2_sq(x, centers[:1])[:, 0], 0.0)
    for j in range(1, k):
        total = float(closest.sum())
        pick = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=closest / total))
        centers[j] = x[pick]
        np.minimum(closest, _l2_sq(x, centers[j : j + 1])[:, 0], out=closest)
        np.maximum(closest, 0.0, out=closest)

    for _ in range(int(iters)):
        labels = np.argmin(_l2_sq(x, centers), axis=1)
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, x)
        empty = counts == 0
        counts[empty] = 1
        centers = sums / counts[:, None]
        if empty.any():  # reseed empty cells on random data points
            centers[empty] = x[rng.integers(0, n, size=int(empty.sum()))]
    return np.ascontiguousarray(centers, dtype=np.float32)


class Index:
    def __init__(self, vectors, nlist=64, nbits=0, m=16, seed=42, iters=15, train=0):
        """Build once from N x 384 normalized float32 vectors."""
        x = np.ascontiguousarray(vectors, dtype=np.float32)
        n, dim = x.shape
        self.dim = dim
        self.nbits = int(nbits)
        self.flat = self.nbits <= 0
        self.m = int(m) if not self.flat else 1
        if dim % self.m:
            raise ValueError("m must divide the vector dimension")
        self.ds = dim // self.m
        self.ks = 1 << self.nbits if not self.flat else 0
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        # 1) coarse k-means -> centroids, then assignment to inverted lists
        fit = x
        if train and 0 < int(train) < n:
            fit = x[rng.choice(n, size=int(train), replace=False)]
        self.nlist = min(int(nlist), n)
        self.centroids = _kmeans(fit, self.nlist, iters, rng)

        assign = np.argmin(_l2_sq(x, self.centroids), axis=1).astype(np.int32)
        self.assign = assign
        order = np.argsort(assign, kind="stable").astype(np.int32)
        counts = np.bincount(assign, minlength=self.nlist)
        self.offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.order = order

        residual = x - self.centroids[assign]
        if self.flat:
            # residuals kept verbatim, stored list-wise for contiguous slicing
            self.residuals = np.ascontiguousarray(residual[order])
            self.codebooks = None
            self.codes = None
            return

        # 2) residual product quantisation, one codebook per sub-vector
        self.codebooks = np.empty((self.m, self.ks, self.ds), dtype=np.float32)
        self.codes = np.empty((n, self.m), dtype=np.uint8 if self.ks <= 256 else np.uint16)
        for s in range(self.m):
            block = residual[:, s * self.ds : (s + 1) * self.ds]
            book = _kmeans(block, self.ks, iters, rng)
            self.codebooks[s] = book
            self.codes[:, s] = np.argmin(_l2_sq(block, book), axis=1)

    # ----------------------------------------------------------------- search
    def search(self, query, k, nprobe=4):
        """Return at most k distinct chunk row ids ranked from best to worst."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if self.offsets[-1] == 0 or k <= 0:
            return []

        coarse = self.centroids @ q
        probe = min(int(nprobe), self.nlist)
        if probe < self.nlist:
            picked = np.argpartition(-coarse, probe - 1)[:probe]
        else:
            picked = np.arange(self.nlist)
        picked = picked[np.argsort(-coarse[picked])]

        if self.flat:
            scores, ids = [], []
            for c in picked:
                lo, hi = int(self.offsets[c]), int(self.offsets[c + 1])
                if hi <= lo:
                    continue
                scores.append(self.residuals[lo:hi] @ q + coarse[c])
                ids.append(self.order[lo:hi])
        else:
            # look-up table of the query against every sub-vector codebook
            lut = np.einsum("md,mkd->mk", q.reshape(self.m, self.ds), self.codebooks)
            slots = np.arange(self.m)[None, :]
            scores, ids = [], []
            for c in picked:
                lo, hi = int(self.offsets[c]), int(self.offsets[c + 1])
                if hi <= lo:
                    continue
                rows = self.order[lo:hi]
                scores.append(lut[slots, self.codes[rows]].sum(axis=1) + coarse[c])
                ids.append(rows)
        if not scores:
            return []

        score = np.concatenate(scores)
        row = np.concatenate(ids)
        take = min(int(k), score.shape[0])
        top = np.argpartition(-score, take - 1)[:take]
        top = top[np.argsort(-score[top], kind="stable")]
        return [int(i) for i in row[top]]

    # ------------------------------------------------------------- accounting
    def size_bytes(self):
        tables = self.centroids.nbytes + self.order.nbytes + self.offsets.nbytes
        if self.flat:
            return int(tables + self.residuals.nbytes)
        return int(tables + self.codes.nbytes + self.codebooks.nbytes)
