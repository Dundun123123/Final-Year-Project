# 课程学习知识库平台：本地混合 RAG 工程实施规范

> 版本：v1.0
> 适用项目：课程学习知识库平台（CAT405）
> 技术目标：将当前“关键词召回 + 外部大模型生成”的原型升级为“本地检索模型主导、生成模型可替换”的可评测 RAG 系统。

---

## 1. 工程定位

本项目的核心智能贡献不是“调用某个聊天 API”，而是实现一条可解释、可运行、可比较的本地检索链路：

```text
课程 PDF / 个人笔记
  → 页码与章节结构化
  → 本地向量编码 + BM25 关键词索引
  → RRF 融合排序
  → 本地 Reranker 重排序
  → 带来源与页码的证据集
  → 可选生成模型组织答案
```

其中，DeepSeek 或本地小语言模型只承担“基于证据组织自然语言答案”的角色。即使生成模型暂时不可用，系统仍能返回已排序的原文证据、文件名和页码；因此，检索效果可以被独立评测。

### 1.1 本期必须完成的智能计算模块

1. 结构化文本分块与页码追溯。
2. 本地 Embedding 向量检索。
3. BM25 与向量检索的混合召回。
4. RRF（Reciprocal Rank Fusion）融合排序。
5. 本地 Cross-Encoder Reranker 重排序。
6. 基于人工标注问题集的检索评测。

### 1.2 本期不作为核心目标的事项

- 从零训练聊天大语言模型。
- 将全部课件上传给外部模型后直接生成答案。
- 仅以模型回复是否“看起来流畅”作为效果判断。
- 承诺处理所有图片公式；公式 OCR 作为后续扩展能力单列实现。

---

## 2. 总体架构

### 2.1 分层职责

```text
展示层（static/）
  - PDF 阅读、标注、问答、检索调试结果与引用跳转

接口层（routers/，后续拆分）
  - 上传、问答、笔记、索引重建、评测 API

业务层（services/）
  - 解析、分块、向量化、BM25、融合、重排序、答案组织

数据层（SQLite + 向量库 + 文件存储）
  - 文件、页码、chunk 元数据、笔记、评测集、向量索引
```

### 2.2 数据存储边界

```text
storage/pdfs/                 原始 PDF
storage/annotated/            含用户高亮标注的 PDF 副本
storage/first_markdown/       课件首次阅读与按页结构化正文
storage/second_markdown/      用户笔记整理结果
storage/vector_db/            本地向量库持久化目录
evaluation/questions.json     人工标注的评测问题集
evaluation/results/           每轮检索实验的结果快照
course_kb.db                  SQLite 元数据与业务数据
```

SQLite 是文件、页码、chunk、笔记和评测元数据的唯一事实来源；向量库只保存向量、`chunk_id` 和检索过滤字段，避免两边同时保存大段正文而造成不一致。

---

## 3. 模型与工具选型

### 3.1 推荐首选组合

- Embedding：`BAAI/bge-m3`。
  - 原因：适合中英混合课件与笔记，可用于语义检索；模型版本必须固定写入配置。
- Reranker：`BAAI/bge-reranker-v2-m3`。
  - 原因：对“问题—候选片段”直接给出相关性排序，适合在 Top 20 候选中选出最能回答问题的证据。
- 向量库：开发阶段优先 FAISS；如需元数据过滤、跨课程 collection 和可视化管理，再迁移至 Chroma 或 Qdrant。
- BM25：使用 `rank-bm25`，每门课程维护一个内存索引，并在上传、删除、笔记更新后重建。
- 生成模型：保留 DeepSeek 作为可选生成端；后续可改为本地小语言模型，接口保持一致。

### 3.2 选择原则

- 检索模型、重排序模型与生成模型职责分离。
- Embedding 与 Reranker 必须可离线运行，确保答辩时不依赖网络。
- 首版先使用预训练权重；只有在积累了人工校验的数据集后才进入微调。
- 所有模型名、版本、设备、批处理大小与向量维度必须记录在 `.env` 或配置文件中。

---

## 4. 数据规范

### 4.1 Chunk 统一结构

每个可检索片段必须具备以下字段：

```json
{
  "chunk_id": "file_uuid_p004_c002",
  "file_id": "file_uuid",
  "subject": "CST435W",
  "page": 4,
  "section_path": "Chapter 2 / Grid Computing",
  "content": "...",
  "source_type": "courseware",
  "note_weight": 1.0,
  "bbox_json": "[...]"
}
```

`source_type` 仅允许使用：

- `courseware`：来自 first_markdown 或 PDF 原文的课件知识；
- `personal_note`：来自 PDF 标注与 second_markdown 的个人笔记；
- `formula_ocr`：后续图片公式识别结果；
- `image_caption`：后续图像描述结果。

### 4.2 笔记索引规则

笔记保存成功后必须在同一个业务流程中完成：

```text
写入 notes 表
  → 更新批注 PDF
  → 更新 second_markdown
  → 生成或更新 personal_note chunk
  → 更新 BM25 索引
  → 更新向量索引
```

若向量索引更新失败，系统必须记录任务状态并允许“重建当前文件索引”；不得静默显示为“已完成”。

### 4.3 结构化分块规则

1. 先按 PDF 页码分隔，页码不得跨越 chunk。
2. 在单页内按 Markdown 标题和段落聚合。
3. 标题必须与紧随其后的正文进入同一个 chunk。
4. 内容超过 600 个中文字符时再切分，保留 80 至 120 字符重叠。
5. 内容低于 120 字符的相邻段落可合并，但不可跨页。
6. 每个 chunk 保留 `section_path`、页码及来源类型。

---

## 5. 检索管线设计

### 5.1 输入与路由

输入为用户问题、当前文件、当前页与当前科目。路由优先级如下：

```text
当前 PDF（用户正在阅读）
  → 当前科目
  → 全部科目
```

若用户明确出现“哪门课”“所有资料”“之前学过”等跨课程语义，才进入全局检索。若用户提到“我的笔记”“我标注过”，则优先增加 `personal_note` 来源的候选配额。

### 5.2 BM25 召回

- 每个检索范围返回 Top 20 候选。
- 用于精确术语、缩写、课程代码、英文专有词和公式名称。
- 文本由 `section_path + content` 组成，使标题也参与匹配。

### 5.3 向量召回

- 使用本地 Embedding 对问题编码。
- 在与 BM25 相同的过滤范围中返回 Top 20 候选。
- 向量库 payload 至少包含：`chunk_id`、`file_id`、`subject`、`page`、`source_type`。

### 5.4 RRF 融合

对 BM25 与向量的结果采用 RRF：

```text
RRF(chunk) = 1 / (k + rank_bm25) + 1 / (k + rank_vector)
```

- 初始参数使用 `k = 60`。
- 某一检索器未召回该 chunk 时，不贡献该项分数。
- `personal_note` 可在识别到笔记意图时乘以 `1.15` 至 `1.25` 的权重。
- 融合后去重并取 Top 20 进入重排序。

### 5.5 本地重排序

```text
输入：问题 q，与融合后每个候选 chunk c 组成 (q, c)
输出：relevance_score(q, c)
结果：按分数降序取 Top 5
```

重排序日志必须记录：`chunk_id`、BM25 名次、向量名次、RRF 分数、rerank 分数、最终名次和耗时。该日志将用于答辩展示“算法如何决定引用来源”。

### 5.6 答案生成与降级

- 若 Top 1 rerank 分数低于阈值，系统返回“当前资料未找到充分依据”，不要求生成模型编造答案。
- 若生成模型可用，向其发送 Top 3 至 Top 5 证据片段，并强制回答附带文件名、页码和来源类型。
- 若生成模型不可用，前端直接展示经过排序的证据卡片，系统仍具备可用检索价值。

---

## 6. 模块拆分与接口约定

### 6.1 建议目录

```text
services/
  chunk_service.py          结构化分块与 chunk 元数据
  embedding_service.py      本地模型加载、单条/批量编码
  vector_store.py           FAISS/Chroma 的写入、查询、删除、持久化
  bm25_service.py           BM25 构建、缓存、召回
  retrieval_service.py      路由、混合召回、RRF 与结果合并
  rerank_service.py         本地 Cross-Encoder 打分
  answer_service.py         证据上下文组织与可选生成模型调用
  evaluation_service.py     指标计算与实验结果写入
```

所有服务只接收明确的数据对象，不直接操作 DOM；API 路由只负责请求验证与调用服务。

### 6.2 新增 API

```text
POST /api/rag/search
  请求：question、scope、current_file_id、current_page、debug
  返回：ranked_chunks、retrieval_trace、latency_ms

POST /api/rag/rebuild-index
  请求：file_id 或 subject
  返回：任务状态、chunk_count、indexed_count

POST /api/evaluation/run
  请求：pipeline_name、question_set
  返回：Recall@5、MRR@10、PageAccuracy、耗时

GET /api/evaluation/results
  返回：历史实验结果摘要
```

`/api/chat` 可保留，但内部必须调用 `retrieval_service`，而不是继续直接按关键词计数排序。

---

## 7. 配置规范

新增配置必须置于 `.env`，并在 `.env.example` 中只提供占位符：

```dotenv
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_DEVICE=cpu
RAG_VECTOR_BACKEND=faiss
RAG_VECTOR_DIR=storage/vector_db
RAG_BM25_TOP_K=20
RAG_VECTOR_TOP_K=20
RAG_RERANK_TOP_K=5
RAG_RRF_K=60
RAG_RELEVANCE_THRESHOLD=0.0
RAG_NOTE_WEIGHT=1.2
```

不得把 API Key、真实绝对用户路径、课程原文或测试问题答案写入公开版本库。

---

## 8. 五阶段实施计划

### P0：评测基线

**目标**：记录当前关键词检索的真实表现。

- 创建 `evaluation/questions.json`，至少 50 道人工审核问题。
- 每题包含：问题、检索范围、正确文件、正确页码、可接受 chunk、问题类型。
- 问题类型至少覆盖术语、同义改写、跨课程、个人笔记、无答案、图表/公式定位。
- 运行现有关键词检索，输出 `baseline_keyword.json`。

**验收**：可以计算 Recall@5、MRR@10、Page Accuracy 与平均检索延迟。

### P1：本地 Embedding 与向量库

**目标**：完成离线语义召回。

- 新增 `embedding_service.py` 与 `vector_store.py`。
- 课件上传和笔记保存后异步建立向量。
- 提供索引重建能力。
- 使用 P0 问题集单独评测向量检索。

**验收**：断网后可重启并完成本地语义检索；每条结果能回到文件和页码。

### P2：BM25 + 向量 + RRF

**目标**：保留术语精确性，同时提升同义问题召回。

- 新增 `bm25_service.py` 与 `retrieval_service.py`。
- 输出 BM25、向量和融合后的候选清单。
- 在界面中仅向普通用户显示引用；调试信息只在开发模式显示。

**验收**：混合检索在 P0 中的 Recall@5 不低于任一单独检索器。

### P3：本地 Reranker

**目标**：提高 Top 3 证据质量。

- 新增 `rerank_service.py`。
- 输入融合 Top 20，输出最终 Top 5。
- 对相邻页、相同章节候选进行上下文合并，避免引用重复。

**验收**：MRR@10 与 Page Accuracy 相对 P2 提升；单次检索耗时被记录。

### P4：数据集与微调预研

**目标**：准备可选的领域适配训练，而非盲目训练聊天模型。

- 从 P0 问题集扩展为 200 至 500 组人工审核的 `query / positive_chunk / hard_negative_chunk` 三元组。
- 划分训练、验证、测试集，禁止同一问题或同一高度相似页面同时出现在训练与测试中。
- 优先微调 Embedding 或 Reranker；生成模型只在具备充足显存、时间和明确目标时再考虑 LoRA。
- 微调结果必须与预训练版本用同一独立测试集比较。

**验收**：微调版本在独立测试集的 Recall@5、MRR@10 或 Page Accuracy 至少有一项可重复提升；若无提升，保留预训练版本并如实记录原因。

---

## 9. 评测规范

### 9.1 问题集格式

```json
{
  "id": "q001",
  "question": "Grid Computing 与 Cloud Computing 的区别是什么？",
  "scope": "CST435W",
  "expected_file_id": "...",
  "expected_pages": [4],
  "acceptable_chunk_ids": ["..."],
  "type": "semantic_paraphrase",
  "verified_by": "student"
}
```

### 9.2 指标定义

- `Recall@5`：正确 chunk 是否出现在前 5 个结果中。
- `MRR@10`：首个正确 chunk 的倒数排名，未在前 10 出现则记为 0。
- `Page Accuracy`：前 3 个结果中是否包含正确页码。
- `Note Recall@5`：笔记类问题是否召回正确个人笔记。
- `No-answer Precision`：资料无答案时，系统正确拒答的比例。
- `Latency`：检索、重排序、生成各阶段耗时与总耗时。

### 9.3 对照实验

必须对比以下四组：

```text
E0：关键词基线
E1：仅本地向量检索
E2：BM25 + 向量 + RRF
E3：E2 + 本地 Reranker + 笔记加权（最终方案）
```

报告中只陈述实际跑出的数据；没有完成的实验明确写为“未来工作”，不得虚构提升幅度。

---

## 10. 答辩演示脚本

1. 上传一份课程 PDF，展示按页结构化、chunk 与本地索引构建。
2. 使用一个专业术语问题，展示 BM25 命中。
3. 使用一个同义改写问题，展示向量检索命中。
4. 打开调试视图，展示 RRF 与 Reranker 如何将正确页提升至前列。
5. 在 PDF 上保存笔记，再提问“我之前如何理解这个概念”，展示 second_markdown 笔记被召回。
6. 点击引用，跳转至 PDF 原始页并展示高亮或原文片段。
7. 切断生成模型或关闭网络，展示本地检索仍能输出证据，证明系统核心能力不依赖外部聊天 API。

---

## 11. 风险与回退方案

- **没有独立显卡**：先使用 CPU 跑小批量 Embedding 与 Reranker；评测集优先保证正确性，性能优化后置。
- **模型下载困难**：模型目录可预下载到本地缓存；系统保留关键词基线作为回退。
- **向量库损坏或版本变化**：提供“从 SQLite chunks 重建索引”功能。
- **大模型 API 不可用**：展示证据检索卡片，不将检索结果错误描述为生成式答案。
- **图片公式未被解析**：标记为 `formula_ocr` 待处理，并以 PDF 页码回跳作为最低可用追溯方案。
- **评测样本过少或泄漏**：优先保证人工审核、训练测试隔离和问题多样性，而非追求样本数量。

---

## 12. 当前执行顺序

当前只开始 **P0：评测基线**。在建立并冻结评测集前，不下载大模型、不微调、不替换现有问答逻辑。

P0 通过后，按 P1、P2、P3 顺序实现；P4 是有时间和硬件资源时的加分项，而不是阻塞系统完成的前置条件。
