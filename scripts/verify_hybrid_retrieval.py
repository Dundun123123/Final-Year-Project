"""Smoke-test local BM25 + BGE-M3 + RRF without calling a chat API."""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.hybrid_retrieval import HybridRetriever  # noqa: E402 - ROOT added above


def main() -> None:
    connection = sqlite3.connect(ROOT / "course_kb.db")
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT chunk_id, file_id, subject, page, content FROM chunks WHERE trim(content) <> '' LIMIT 24"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError(
            "No course chunks found. Upload a PDF before running this check."
        )

    retriever = HybridRetriever(ROOT / "storage" / "models")
    results, mode = retriever.retrieve(
        "cloud computing system", rows, candidate_k=10, evidence_k=3
    )
    print(f"mode={mode}")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item.subject} page={item.page} rrf={item.rrf_score} | {item.content[:90]}"
        )


if __name__ == "__main__":
    main()
