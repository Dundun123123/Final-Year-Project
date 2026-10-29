# Knowledge Island（知屿）

面向课程 PDF 的本地学习知识库原型：阅读原始 PDF、按页标注、记录笔记，并通过课程片段进行可追溯问答。当前检索流程为 BM25 + 本地 BGE-M3 + RRF；若本地向量模型未就绪，会回退到关键词检索。

## 当前实现

- PDF 上传、阅读、翻页、缩放与页码定位。
- `storage/first_markdown` 保存首次整理结果，`storage/second_markdown` 保存个人笔记整理结果；同一文件和提示词可复用首次阅读缓存。
- BM25、BGE-M3 向量召回与 RRF 融合；`/api/retrieval/preview` 可在本机查看证据片段及页码。
- 课件问答、划词翻译与截图问答可调用用户配置的 DeepSeek API。
- PDF 笔记写入本地数据库和批注文件，并更新个人笔记层。
- 中英文界面和本地账号原型。

Reranker、本地回答模型、课程群共享与积分系统仍在规划或开发中，当前仓库不把它们标为已完成。

## Windows 本地运行

需要 Python 3.11 或更高版本。项目默认监听 `127.0.0.1:8000`。

```cmd
py -m venv .rag-venv
.rag-venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-rag.txt
copy .env.example .env
start.bat
```

在本机 `.env` 中填写自己的 `DEEPSEEK_API_KEY`。如需语义检索，先将 `BAAI/bge-m3` 下载到 `storage/models`；运行时按本地缓存加载。无缓存或依赖时，聊天仍可使用关键词回退。首次准备 CUDA 版 PyTorch 时，请按本机显卡环境选择相应安装包。

浏览器入口：<http://127.0.0.1:8000/main>。进入学习工作台：<http://127.0.0.1:8000/app>。

## 本地检查

```cmd
.rag-venv\Scripts\python.exe scripts\verify_hybrid_retrieval.py
.rag-venv\Scripts\python.exe scripts\evaluate_retrieval.py --retriever keyword --name keyword-baseline
```

检索评测需要先上传课程 PDF，并在 `evaluation/questions.json` 填写人工核对过的问题；格式见 `evaluation/README.md`。检索预览接口不会调用云端模型。普通问答和翻译在配置 API 后会向 DeepSeek 发送必要文本。

## 代码格式与检查

开发环境额外安装 `requirements-dev.txt`。Black 负责代码格式，isort 负责导入顺序；Flake8 检查基础规范，Ruff 检查常见错误与过时写法。四项检查也会在 GitHub Actions 中运行。

```cmd
.rag-venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.rag-venv\Scripts\python.exe -m isort --check-only main.py services prompts scripts tests
.rag-venv\Scripts\python.exe -m black --check main.py services prompts scripts tests
.rag-venv\Scripts\python.exe -m flake8 main.py services prompts scripts tests
.rag-venv\Scripts\python.exe -m ruff check main.py services prompts scripts tests
.rag-venv\Scripts\python.exe -m unittest discover -s tests -v
```

需要自动整理时，先运行 `python -m isort`，再运行 `python -m black`；Ruff 不再重复执行格式化。

## 数据与使用范围

这是**单机原型**。当前登录接口没有服务器端会话和按用户隔离的访问控制，因此不要直接开放到公网，也不要把它当作已经具备多人共享权限的服务。`storage/`、SQLite 数据库、模型文件、`.env` 和本地生成物均被 Git 忽略；提交前仍应运行 `git status` 检查。

项目结构：`main.py` 为 FastAPI 入口；`services/` 为 PDF 和检索逻辑；`prompts/` 为提示词；`static/` 为前端；`evaluation/` 为人工标注与评测说明。
