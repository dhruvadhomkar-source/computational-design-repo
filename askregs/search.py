"""Retrieval: find the chunks most likely to answer a question.

Three modes, so they can be compared on the test set:
  bm25   - keyword match (askregs/bm25.py)
  dense  - embedding (meaning) match: chunks and question become 384-number vectors,
           and we rank chunks by cosine similarity to the question
  hybrid - both, merged with Reciprocal Rank Fusion (RRF): a chunk scores 1/(60 + rank)
           in each list, so one ranked high by either method rises to the top
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .ingest import Chunk, load_chunks

MODEL = "BAAI/bge-small-en-v1.5"  # small, free, runs on a laptop CPU
RRF_K = 60                        # standard RRF constant; dampens the gap between ranks 1 and 2


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks_file: Path, embeddings_file: Path):
        self.chunks = load_chunks(chunks_file)
        self.bm25 = BM25([c.text for c in self.chunks])
        self._embeddings_file = embeddings_file
        self._model = None
        self._vectors = None

    # --- dense search -------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: only dense search needs it
            self._model = TextEmbedding(MODEL)
        return self._model

    @property
    def vectors(self) -> np.ndarray:
        """Chunk vectors, computed once and cached to disk (normalised to length 1)."""
        if self._vectors is None:
            if self._embeddings_file.exists():
                self._vectors = np.load(self._embeddings_file)
            if self._vectors is None or len(self._vectors) != len(self.chunks):
                # Small batches: long chunks in big batches use gigabytes of memory and run slower.
                texts = [c.text for c in self.chunks]
                vectors = np.array(list(self.model.passage_embed(texts, batch_size=16)))
                self._vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                np.save(self._embeddings_file, self._vectors)
        return self._vectors

    def dense_scores(self, query: str) -> np.ndarray:
        q = np.array(next(iter(self.model.query_embed([query]))))
        return self.vectors @ (q / np.linalg.norm(q))  # cosine similarity = dot product of unit vectors

    # --- the three modes ----------------------------------------------------------
    def ranking(self, query: str, mode: str) -> list[tuple[int, float]]:
        """All chunk indices, best first, with their score in this mode."""
        if mode == "bm25":
            scores = np.array(self.bm25.scores(query))
        elif mode == "dense":
            scores = self.dense_scores(query)
        elif mode == "hybrid":
            scores = np.zeros(len(self.chunks))
            for sub in ("bm25", "dense"):
                for rank, (i, _) in enumerate(self.ranking(query, sub), start=1):
                    scores[i] += 1 / (RRF_K + rank)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, k: int = 5, mode: str = "dense") -> list[Hit]:
        return [Hit(self.chunks[i], s) for i, s in self.ranking(query, mode)[:k]]
