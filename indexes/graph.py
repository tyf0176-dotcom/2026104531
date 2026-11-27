"""HNSW graph index over the frozen 384-d chunk vectors.

Only NumPy is used for the arithmetic; the layered graph, insertion,
neighbour selection and beam search are implemented below.

Construction parameters
    M               neighbours kept per node above layer 0 (layer 0 keeps 2*M)
    efConstruction  candidate width while inserting
    seed            seed for the level assignment (fixed => reproducible build)
    mL              level scale; 0 means 1/ln(M)
Query parameter
    efSearch        candidate width of the bottom-layer beam search
"""

import heapq

import numpy as np


class Index:
    def __init__(self, vectors, M=16, efConstruction=100, seed=42, mL=0):
        """Build once from N x 384 normalized float32 vectors."""
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.dim = self.vectors.shape
        self.M = max(int(M), 2)
        self.M0 = 2 * self.M
        self.efConstruction = max(int(efConstruction), self.M)
        self.mL = float(mL) if mL else 1.0 / np.log(self.M)
        rng = np.random.default_rng(int(seed))

        # Exponentially distributed levels, level = floor(-ln(U) * mL).
        u = 1.0 - rng.random(self.n)
        self.levels = np.minimum((-np.log(u) * self.mL).astype(np.int32), 24)
        self.max_level = int(self.levels.max()) if self.n else 0

        # links[level][node] -> list of neighbour row ids
        self.links = [[[] for _ in range(self.n)] for _ in range(self.max_level + 1)]
        self.entry = -1
        self.top_level = -1
        for node in range(self.n):
            self._insert(node)

    # ------------------------------------------------------------------ build
    def _insert(self, node):
        q = self.vectors[node]
        level = int(self.levels[node])
        if self.entry < 0:
            self.entry, self.top_level = node, level
            return

        ep = self.entry
        for lc in range(self.top_level, level, -1):
            ep = self._greedy(q, ep, lc)

        for lc in range(min(level, self.top_level), -1, -1):
            limit = self.M0 if lc == 0 else self.M
            candidates = self._search_layer(q, [ep], self.efConstruction, lc)
            chosen = self._select(node, candidates, self.M)
            self.links[lc][node] = chosen
            for other in chosen:
                bucket = self.links[lc][other]
                bucket.append(node)
                if len(bucket) > limit:
                    self.links[lc][other] = self._select(other, self._pairs(other, bucket), limit)
            if candidates:
                ep = candidates[0][1]

        if level > self.top_level:
            self.entry, self.top_level = node, level

    def _pairs(self, node, neighbours):
        """(similarity, id) pairs of a node against the given ids."""
        sims = self.vectors[np.asarray(neighbours, dtype=np.int64)] @ self.vectors[node]
        return [(float(s), int(j)) for s, j in zip(sims, neighbours)]

    def _select(self, node, candidates, keep):
        """Neighbour heuristic: keep a candidate only if it is closer to the
        query than to any already kept neighbour, then fill with nearest."""
        if not candidates:
            return []
        order = sorted(candidates, reverse=True)
        if len(order) <= keep:
            return [int(j) for _, j in order]

        ids = np.fromiter((j for _, j in order), dtype=np.int64, count=len(order))
        to_query = np.fromiter((s for s, _ in order), dtype=np.float32, count=len(order))
        vecs = self.vectors[ids]
        kept = []
        for pos in range(len(ids)):
            if kept:
                if np.any(vecs[kept] @ vecs[pos] >= to_query[pos]):
                    continue
            kept.append(pos)
            if len(kept) == keep:
                break
        if len(kept) < keep:
            taken = set(kept)
            for pos in range(len(ids)):
                if pos not in taken:
                    kept.append(pos)
                    if len(kept) == keep:
                        break
        return [int(ids[p]) for p in kept]

    # ----------------------------------------------------------------- search
    def _greedy(self, q, ep, level):
        """Hill-climb on one upper layer."""
        cur = ep
        cur_sim = float(self.vectors[cur] @ q)
        while True:
            neighbours = self.links[level][cur]
            if not neighbours:
                return cur
            sims = self.vectors[neighbours] @ q
            best = int(np.argmax(sims))
            if float(sims[best]) <= cur_sim:
                return cur
            cur, cur_sim = neighbours[best], float(sims[best])

    def _search_layer(self, q, entries, ef, level):
        """Beam search on one layer; returns (similarity, id) sorted descending."""
        visited = set(entries)
        candidates = []
        result = []
        for e in entries:
            s = float(self.vectors[e] @ q)
            heapq.heappush(candidates, (-s, e))
            heapq.heappush(result, (s, e))
        while len(result) > ef:
            heapq.heappop(result)

        while candidates:
            neg_s, cur = heapq.heappop(candidates)
            s = -neg_s
            if len(result) >= ef and s < result[0][0]:
                break
            neighbours = self.links[level][cur]
            if not neighbours:
                continue
            sims = self.vectors[neighbours] @ q
            worst = result[0][0] if len(result) >= ef else -np.inf
            for pos, nb in enumerate(neighbours):
                if nb in visited:
                    continue
                visited.add(nb)
                sn = float(sims[pos])
                if len(result) < ef or sn > worst:
                    heapq.heappush(candidates, (-sn, nb))
                    heapq.heappush(result, (sn, nb))
                    if len(result) > ef:
                        heapq.heappop(result)
                    worst = result[0][0]
        return sorted(((s, i) for s, i in result), reverse=True)

    def search(self, query, k, efSearch=128):
        """Return at most k distinct chunk row ids ranked from best to worst."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if self.n == 0 or k <= 0:
            return []
        ef = max(int(efSearch), int(k))
        ep = self.entry
        for lc in range(self.top_level, 0, -1):
            ep = self._greedy(q, ep, lc)
        found = self._search_layer(q, [ep], ef, 0)
        return [int(i) for _, i in found[: int(k)]]

    # ------------------------------------------------------------- accounting
    def size_bytes(self):
        """Adjacency storage counted as 4-byte ids per directed edge."""
        edges = sum(len(bucket) for level in self.links for bucket in level)
        return edges * 4 + int(self.levels.nbytes)
