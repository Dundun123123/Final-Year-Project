# RAG 检索评测集

本目录保存人工审核的测试问题和每轮模型评测结果。它用于比较：

```text
关键词基线 → 预训练 Embedding → 微调 Embedding → 混合检索 → Reranker
```

## 1. 标注流程

1. 打开原始 PDF，自己阅读并提出一个真实的学习问题。
2. 记录正确课程、PDF 文件名与页码。
3. 从正确页复制 1 至 3 句原文到 `expected_evidence`。
4. 如有需要，补充一至两句 `reference_answer`；它不参与检索指标计算。
5. 将确认后的对象写入 `questions.json` 的 `questions` 数组。

不要让模型自动生成正确页码或标准证据。评测集必须人工确认，否则无法作为毕业设计实验依据。

## 2. 每题最小字段

```json
{
  "id": "q001",
  "question": "Grid Computing 和 Cloud Computing 有什么区别？",
  "scope": "CST435W",
  "expected_file_names": ["Cloud Computing 2026 Part I.pdf"],
  "expected_pages": [4],
  "expected_evidence": ["原始 PDF 中核对过的关键句"],
  "type": "semantic_paraphrase"
}
```

`scope` 填课程代码时只在该课程检索；填 `global` 时会在全部课程中检索。

## 3. 推荐题目比例

- 10 题：精确术语或英文缩写。
- 15 题：同义改写或自然语言解释。
- 8 题：跨课程定位。
- 8 题：个人笔记召回。
- 5 题：资料中不存在答案的拒答测试。
- 其余：图表、公式或章节定位问题。

第一轮至少完成 30 题；最终报告建议达到 50 题以上。

## 4. 运行预训练 BGE-M3 基线

在项目根目录运行：

```cmd
.rag-venv\Scripts\python.exe scripts\evaluate_retrieval.py --model BAAI/bge-m3 --name bge-m3-base --local-only
```

结果会写入 `evaluation/results/bge-m3-base.json`。

## 5. 运行当前关键词基线

```cmd
.rag-venv\Scripts\python.exe scripts\evaluate_retrieval.py --retriever keyword --name keyword-baseline
```

结果会写入 `evaluation/results/keyword-baseline.json`。该模式复用当前系统的关键词计数逻辑，不加载任何模型。

## 6. 运行微调后的模型

假设你将微调模型保存在 `D:\models\course-bge-m3`：

```cmd
.rag-venv\Scripts\python.exe scripts\evaluate_retrieval.py --model D:\models\course-bge-m3 --name bge-m3-finetuned --local-only
```

比较 `bge-m3-base.json` 与 `bge-m3-finetuned.json` 中的：

- `recall_at_5`：正确证据是否进入前 5；
- `mrr_at_10`：第一个正确证据的平均排名质量；
- `page_accuracy_at_3`：前 3 个结果是否包含正确 PDF 页；
- `average_query_latency_ms`：单题平均检索耗时。

只有在独立评测题集上指标提高时，才能说明微调带来了有效改进。
