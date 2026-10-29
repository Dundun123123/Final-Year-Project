"""Local BM25 + BGE-M3 + RRF retrieval for the course knowledge base.

The runtime intentionally keeps its index in process memory. It is rebuilt when
the chunk corpus changes, which makes newly uploaded PDFs and saved notes
searchable without a separate vector-database service during this project stage.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9_+-]+")
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "can",
    "could",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "this",
    "that",
    "these",
    "those",
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    file_id: str
    subject: str
    page: int
    content: str
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0


class HybridRetriever:
    """Provides a local hybrid retriever with a graceful keyword-only fallback."""

    def __init__(
        self, model_cache_dir: Path, model_name: str = "BAAI/bge-m3", rrf_k: int = 60
    ):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_name = model_name
        self.rrf_k = rrf_k
        self._lock = threading.RLock()
        self._model = None
        self._index = None
        self._indexed_chunks: list[RetrievedChunk] = []
        self._fingerprint: str | None = None
        self._last_error: str | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [
            token
            for token in TOKEN_PATTERN.findall((text or "").lower())
            if "\u4e00" <= token[0] <= "\u9fff"
            or (len(token) > 1 and token not in ENGLISH_STOPWORDS)
        ]

    @staticmethod
    def _to_chunk(row: Any) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(row["chunk_id"]),
            file_id=str(row["file_id"]),
            subject=str(row["subject"]),
            page=int(row["page"]),
            content=str(row["content"]),
        )

    @staticmethod
    def _fingerprint_for(chunks: Iterable[RetrievedChunk]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(f"{chunk.chunk_id}\0{chunk.content}\0".encode())
        return digest.hexdigest()

    def invalidate(self) -> None:
        """Discard the in-memory vector index after upload or note updates."""
        with self._lock:
            self._index = None
            self._indexed_chunks = []
            self._fingerprint = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "index_ready": self._index is not None,
            "indexed_chunk_count": len(self._indexed_chunks),
            "last_error": self._last_error,
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=str(self.model_cache_dir),
            local_files_only=True,
        )
        return self._model

    def _ensure_vector_index(self, chunks: list[RetrievedChunk]) -> None:
        fingerprint = self._fingerprint_for(chunks)
        if self._index is not None and self._fingerprint == fingerprint:
            return

        import faiss

        model = self._load_model()
        texts = [f"{item.subject}\nPage {item.page}\n{item.content}" for item in chunks]
        vectors = model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index
        self._indexed_chunks = chunks
        self._fingerprint = fingerprint

    def _bm25_results(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[tuple[int, float]]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        try:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [self.tokenize(item.content) for item in chunks]
            scores = BM25Okapi(tokenized_corpus).get_scores(query_tokens)
        except ModuleNotFoundError:
            # Preserve a safe local baseline if the optional RAG dependencies
            # have not yet been installed in the active Python environment.
            scores = [
                sum(
                    item.content.lower().count(token)
                    for token in query_tokens
                    if len(token) > 1
                )
                for item in chunks
            ]
        positions = sorted(
            range(len(chunks)), key=lambda index: float(scores[index]), reverse=True
        )
        return [
            (position, float(scores[position]))
            for position in positions[:top_k]
            if scores[position] > 0
        ]

    def _vector_results(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[tuple[int, float]]:
        self._ensure_vector_index(chunks)
        model = self._load_model()
        query_vector = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        scores, positions = self._index.search(query_vector, min(top_k, len(chunks)))
        return [
            (int(position), float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0
        ]

    def _rrf(
        self,
        chunks: list[RetrievedChunk],
        bm25_results: list[tuple[int, float]],
        vector_results: list[tuple[int, float]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        merged: dict[int, dict[str, float | int | None]] = {}
        for source, results in (("bm25", bm25_results), ("vector", vector_results)):
            for rank, (position, _) in enumerate(results, start=1):
                record = merged.setdefault(
                    position, {"score": 0.0, "bm25_rank": None, "vector_rank": None}
                )
                record["score"] = float(record["score"]) + 1 / (self.rrf_k + rank)
                record[f"{source}_rank"] = rank

        ordered = sorted(
            merged.items(), key=lambda pair: float(pair[1]["score"]), reverse=True
        )[:top_k]
        return [
            RetrievedChunk(
                **{
                    **chunks[position].__dict__,
                    "bm25_rank": data["bm25_rank"],
                    "vector_rank": data["vector_rank"],
                    "rrf_score": round(float(data["score"]), 8),
                }
            )
            for position, data in ordered
        ]

    def retrieve(
        self,
        query: str,
        rows: Iterable[Any],
        candidate_k: int = 20,
        evidence_k: int = 5,
    ) -> tuple[list[RetrievedChunk], str]:
        """Return fused evidence and the active retrieval mode.

        The fallback is intentional: the application can still answer from the
        original PDF chunks when local BGE-M3 dependencies or cached weights are
        unavailable, while exposing the mode to the client for transparency.
        """
        chunks: list[RetrievedChunk] = []
        seen: set[tuple[str, int, str]] = set()
        for row in rows:
            item = self._to_chunk(row)
            normalised_content = " ".join(item.content.split())
            # PDF extraction can create micro-fragments such as “is”. They are
            # not meaningful evidence, but retain short chunks containing a
            # number because they may be a compact formula or parameter.
            if len(normalised_content) < 16 and not any(
                char.isdigit() for char in normalised_content
            ):
                continue
            duplicate_key = (item.subject, item.page, normalised_content)
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)
            chunks.append(item)
        if not chunks or not query.strip():
            return [], "no-evidence"
        with self._lock:
            try:
                bm25 = self._bm25_results(query, chunks, candidate_k)
                vector = self._vector_results(query, chunks, candidate_k)
                self._last_error = None
                return self._rrf(chunks, bm25, vector, evidence_k), "bm25+bge-m3+rrf"
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                fallback = self._bm25_results(query, chunks, evidence_k)
                return self._rrf(chunks, fallback, [], evidence_k), "bm25-fallback"
