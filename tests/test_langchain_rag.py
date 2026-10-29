"""Offline contract tests for the LangChain integration."""

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from services.hybrid_retrieval import RetrievedChunk
from services.llm_gateway import call_deepseek
from services.rag_chain import answer_from_documents, chunks_to_documents, citations_from_documents


class LangChainRagTests(unittest.TestCase):
    def test_pdf_and_note_documents_keep_traceability(self):
        chunks = [
            RetrievedChunk("pdf_1", "file-1", "CST435W", 4, "Grid computing", 1, 2, 0.03),
            RetrievedChunk("note_2", "file-1", "CST435W", 4, "我的课堂笔记", 2, 1, 0.02),
        ]
        docs = chunks_to_documents(chunks)
        self.assertEqual([doc.metadata["source_type"] for doc in docs], ["pdf", "note"])
        self.assertEqual(docs[0].metadata["file_id"], "file-1")
        self.assertEqual(citations_from_documents(docs)[1]["page"], 4)

    def test_answer_chain_includes_question_and_page_evidence(self):
        docs = chunks_to_documents([RetrievedChunk("c1", "f1", "CST435W", 4, "Grid computing uses distributed resources.")])

        def fake_model(prompt):
            messages = prompt.to_messages()
            self.assertIn("Grid computing uses distributed resources.", messages[1].content)
            self.assertIn("页码：4", messages[1].content)
            self.assertIn("有什么区别？", messages[1].content)
            return AIMessage(content="基于第 4 页的回答。")

        answer = answer_from_documents("有什么区别？", docs, RunnableLambda(fake_model))
        self.assertEqual(answer, "基于第 4 页的回答。")

    def test_gateway_converts_existing_message_format(self):
        received = []

        class FakeModel:
            def invoke(self, messages):
                received.extend(messages)
                return AIMessage(content="  翻译结果  ")

        with patch("services.llm_gateway.make_deepseek_model", return_value=FakeModel()):
            result = call_deepseek(
                [{"role": "system", "content": "翻译"}, {"role": "user", "content": "knowledge"}],
                "deepseek-flash",
            )
        self.assertEqual(result, "翻译结果")
        self.assertIsInstance(received[0], SystemMessage)
        self.assertIsInstance(received[1], HumanMessage)


if __name__ == "__main__":
    unittest.main()
