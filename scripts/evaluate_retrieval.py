"""Evaluate a local embedding model against manually labelled courseware questions.

The script intentionally evaluates retrieval only. It does not call any chat API,
so Recall@5, MRR@10 and page accuracy remain attributable to the local model.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "course_kb.db"
DEFAULT_QUESTIONS = ROOT / "evaluation" / "questions.json"
DEFAULT_RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_CACHE_DIR = ROOT / "storage" / "models"


@dataclass
class Chunk:
    chunk_id: str
    file_id: str
    file_name: str
    subject: str
    page: int
    content: str


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", []) if isinstance(payload, dict) else payload
    if not questions:
        raise ValueError(
            f"{path} 还没有人工标注题目。请先参考 evaluation/questions.template.json 填写至少一题。"
        )
    required = {"id", "question", "expected_pages"}
    for item in questions:
        missing = required - set(item)
        if missing:
            raise ValueError(f"题目 {item.get('id', '<unknown>')} 缺少字段：{', '.join(sorted(missing))}")
    return questions


def load_chunks(database_path: Path) -> list[Chunk]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT c.chunk_id, c.file_id, f.file_name, c.subject, c.page, c.content
            FROM chunks AS c
            JOIN files AS f ON f.file_id = c.file_id
            WHERE c.content IS NOT NULL AND trim(c.content) <> ''
            ORDER BY c.subject, c.file_id, c.page, c.chunk_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [Chunk(**dict(row)) for row in rows]


def normalise_scope(question: dict[str, Any]) -> str | None:
    scope = str(question.get("scope", "global")).strip()
    return None if not scope or scope.lower() == "global" else scope


def is_relevant(question: dict[str, Any], chunk: Chunk) -> bool:
    acceptable_chunk_ids = set(question.get("acceptable_chunk_ids") or [])
    if chunk.chunk_id in acceptable_chunk_ids:
        return True

    expected_pages = {int(page) for page in question.get("expected_pages", [])}
    expected_file_ids = set(question.get("expected_file_ids") or [])
    expected_file_names = set(question.get("expected_file_names") or [])
    file_matches = not expected_file_ids and not expected_file_names
    file_matches = file_matches or chunk.file_id in expected_file_ids or chunk.file_name in expected_file_names
    return file_matches and chunk.page in expected_pages


def group_chunks(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    scopes: dict[str, list[Chunk]] = {"global": chunks}
    for chunk in chunks:
        scopes.setdefault(chunk.subject, []).append(chunk)
    return scopes


def keyword_score(query: str, content: str) -> float:
    """Match the current application's keyword-count baseline."""
    tokens = set(re.findall(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9_]+", query.lower()))
    return float(sum(content.lower().count(token) for token in tokens if len(token) > 1))


def build_indexes(
    model: SentenceTransformer,
    chunks: list[Chunk],
    batch_size: int,
) -> dict[str, tuple[faiss.IndexFlatIP, list[Chunk]]]:
    indexes: dict[str, tuple[faiss.IndexFlatIP, list[Chunk]]] = {}
    for scope, scope_chunks in group_chunks(chunks).items():
        texts = [f"{chunk.subject}\nPage {chunk.page}\n{chunk.content}" for chunk in scope_chunks]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        indexes[scope] = (index, scope_chunks)
        print(f"indexed scope={scope} chunks={len(scope_chunks)}")
    return indexes


def evaluate(
    model: SentenceTransformer | None,
    indexes: dict[str, tuple[faiss.IndexFlatIP, list[Chunk]]] | None,
    scopes: dict[str, list[Chunk]],
    questions: list[dict[str, Any]],
    top_k: int,
    retriever: str,
) -> dict[str, Any]:
    per_question: list[dict[str, Any]] = []
    recall_hits = 0
    page_hits = 0
    reciprocal_rank_sum = 0.0
    total_latency_ms = 0.0

    for question in questions:
        scope = normalise_scope(question) or "global"
        if scope not in scopes:
            raise ValueError(f"题目 {question['id']} 的 scope={scope} 不存在于当前知识库。")
        scope_chunks = scopes[scope]
        started_at = time.perf_counter()
        limit = min(max(top_k, 10), len(scope_chunks))
        if retriever == "keyword":
            candidate_pairs = sorted(
                ((position, keyword_score(question["question"], chunk.content)) for position, chunk in enumerate(scope_chunks)),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        else:
            if model is None or indexes is None:
                raise RuntimeError("dense 检索需要已加载的模型和向量索引。")
            index, _ = indexes[scope]
            query_vector = model.encode(
                [question["question"]],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")
            scores, positions = index.search(query_vector, limit)
            candidate_pairs = [(int(position), float(score)) for position, score in zip(positions[0], scores[0], strict=True)]
        latency_ms = (time.perf_counter() - started_at) * 1000
        total_latency_ms += latency_ms

        ranked = []
        first_relevant_rank: int | None = None
        for position, score in candidate_pairs:
            chunk = scope_chunks[position]
            relevant = is_relevant(question, chunk)
            rank = len(ranked) + 1
            if relevant and first_relevant_rank is None:
                first_relevant_rank = rank
            ranked.append(
                {
                    "rank": rank,
                    "score": round(float(score), 6),
                    "relevant": relevant,
                    "chunk_id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "file_name": chunk.file_name,
                    "subject": chunk.subject,
                    "page": chunk.page,
                    "snippet": chunk.content[:240],
                }
            )

        recall_hits += int(first_relevant_rank is not None and first_relevant_rank <= 5)
        page_hits += int(any(item["relevant"] for item in ranked[:3]))
        reciprocal_rank_sum += 1 / first_relevant_rank if first_relevant_rank and first_relevant_rank <= 10 else 0.0
        per_question.append(
            {
                "id": question["id"],
                "question": question["question"],
                "scope": scope,
                "expected_pages": question["expected_pages"],
                "first_relevant_rank": first_relevant_rank,
                "latency_ms": round(latency_ms, 2),
                "top_results": ranked,
            }
        )

    total = len(questions)
    return {
        "question_count": total,
        "recall_at_5": round(recall_hits / total, 4),
        "mrr_at_10": round(reciprocal_rank_sum / total, 4),
        "page_accuracy_at_3": round(page_hits / total, 4),
        "average_query_latency_ms": round(total_latency_ms / total, 2),
        "per_question": per_question,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local embedding model on courseware retrieval.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Hugging Face model name or local fine-tuned model directory")
    parser.add_argument("--name", required=True, help="Result file name, e.g. bge-m3-base")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--retriever", choices=["dense", "keyword"], default="dense")
    parser.add_argument("--local-only", action="store_true", help="Only load already cached local model files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database.exists():
        raise FileNotFoundError(f"找不到数据库：{args.database}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请使用 --device cpu 或检查 NVIDIA 环境。")

    questions = load_questions(args.questions)
    chunks = load_chunks(args.database)
    if not chunks:
        raise ValueError("知识库中没有可评测 chunk，请先上传并解析至少一份 PDF。")
    scopes = group_chunks(chunks)
    model = None
    indexes = None
    if args.retriever == "dense":
        print(f"loading model={args.model} device={device}")
        model = SentenceTransformer(
            args.model,
            device=device,
            cache_folder=str(args.cache_dir),
            local_files_only=args.local_only,
        )
        indexes = build_indexes(model, chunks, args.batch_size)
    metrics = evaluate(model, indexes, scopes, questions, args.top_k, args.retriever)
    output = {
        "run_name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "retriever": args.retriever,
        "device": device,
        "database": str(args.database),
        "metrics": metrics,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / f"{args.name}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Retrieval evaluation ===")
    print(f"Recall@5: {metrics['recall_at_5']:.2%}")
    print(f"MRR@10: {metrics['mrr_at_10']:.4f}")
    print(f"Page Accuracy@3: {metrics['page_accuracy_at_3']:.2%}")
    print(f"Average query latency: {metrics['average_query_latency_ms']:.2f} ms")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
