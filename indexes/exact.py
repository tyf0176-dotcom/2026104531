"""Provided exhaustive baseline over the frozen corpus vectors."""
import numpy as np


class Index:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, query, k):
        scores = self.vectors @ query
        return np.argsort(-scores, kind="stable")[:k]
