"""Evidence-constrained LangChain RAG flow and traceable Document adapter."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from prompts.rag_prompts import rag_answer_system_prompt, rag_answer_user_prompt
from services.hybrid_retrieval import RetrievedChunk

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", rag_answer_system_prompt()),
        ("human", rag_answer_user_prompt("{question}", "{evidence}")),
    ]
)


def chunks_to_documents(chunks: Iterable[RetrievedChunk]) -> list[Document]:
    """Keep PDF and personal-note provenance attached to each retrieved chunk."""
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "subject": chunk.subject,
                "page": chunk.page,
                "source_type": "note" if chunk.chunk_id.startswith("note_") else "pdf",
                "bm25_rank": chunk.bm25_rank,
                "vector_rank": chunk.vector_rank,
                "rrf_score": chunk.rrf_score,
            },
        )
        for chunk in chunks
    ]


def citations_from_documents(documents: Iterable[Document]) -> list[dict]:
    return [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "subject": doc.metadata["subject"],
            "page": doc.metadata["page"],
            "snippet": doc.page_content[:160],
            "bm25_rank": doc.metadata["bm25_rank"],
            "vector_rank": doc.metadata["vector_rank"],
            "rrf_score": doc.metadata["rrf_score"],
        }
        for doc in documents
    ]


def format_evidence(documents: Iterable[Document]) -> str:
    return "\n\n".join(
        f"【证据 {index}｜科目：{doc.metadata['subject']}｜页码：{doc.metadata['page']}】\n{doc.page_content}"
        for index, doc in enumerate(documents, start=1)
    )


def answer_from_documents(
    question: str, documents: list[Document], model: Runnable
) -> str:
    """Compose the prompt, model and output parser as a LangChain chain."""
    chain = RAG_PROMPT | model | StrOutputParser()
    return chain.invoke(
        {"question": question, "evidence": format_evidence(documents)}
    ).strip()
