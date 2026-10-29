from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.hybrid_retrieval import HybridRetriever
from services.llm_gateway import call_deepseek, make_deepseek_model
from services.markdown_pipeline import (
    create_first_markdown,
    first_markdown_prompt_fingerprint,
    local_first_markdown,
    update_second_markdown,
)
from services.pdf_extractor import extract_pdf
from services.rag_chain import (
    answer_from_documents,
    chunks_to_documents,
    citations_from_documents,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
STORAGE = ROOT / "storage"
PDFS = STORAGE / "pdfs"
LEGACY_MARKDOWN = STORAGE / "markdown"
FIRST_MARKDOWN = STORAGE / "first_markdown"
SECOND_MARKDOWN = STORAGE / "second_markdown"
ANNOTATED = STORAGE / "annotated"
MODEL_CACHE = STORAGE / "models"
DB_PATH = ROOT / "course_kb.db"
for p in (
    PDFS,
    STORAGE / "screenshots",
    LEGACY_MARKDOWN,
    FIRST_MARKDOWN,
    SECOND_MARKDOWN,
    ANNOTATED,
):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="课程学习知识库平台", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
retriever = HybridRetriever(MODEL_CACHE)


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


load_env()


def deepseek(messages, max_tokens=800):
    """Compatibility entry point for first/second Markdown and translation."""
    return call_deepseek(messages, current_model_name(), max_tokens)


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS files (file_id TEXT PRIMARY KEY, file_name TEXT, subject TEXT, file_type TEXT, pdf_path TEXT, upload_time TEXT);
        CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, file_id TEXT, page INTEGER, content TEXT, bbox_json TEXT, subject TEXT);
        CREATE TABLE IF NOT EXISTS pages (file_id TEXT, page INTEGER, content TEXT, words_json TEXT, width REAL, height REAL, PRIMARY KEY(file_id, page));
        CREATE TABLE IF NOT EXISTS ner_entities (entity_id TEXT PRIMARY KEY, file_id TEXT, page INTEGER, entity_text TEXT, entity_type TEXT, bbox_json TEXT);
        CREATE TABLE IF NOT EXISTS conversations (conversation_id TEXT PRIMARY KEY, agent_type TEXT, subject TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS messages (message_id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT, citations_json TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS translations (source TEXT PRIMARY KEY, result TEXT);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS notes (note_id TEXT PRIMARY KEY, file_id TEXT, page INTEGER, selected_text TEXT, note_content TEXT, x0 REAL, y0 REAL, x1 REAL, y1 REAL, created_at TEXT);
        CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS first_markdown_cache (source_hash TEXT, prompt_hash TEXT, model_name TEXT, subject TEXT, markdown_content TEXT, created_at TEXT, PRIMARY KEY(source_hash, prompt_hash, model_name, subject));
        """)
        columns = {row[1] for row in c.execute("PRAGMA table_info(pages)")}
        for name, ddl in {
            "words_json": "TEXT",
            "width": "REAL",
            "height": "REAL",
        }.items():
            if name not in columns:
                c.execute(f"ALTER TABLE pages ADD COLUMN {name} {ddl}")


@app.on_event("startup")
def startup():
    init_db()
    # Earlier note versions created visible PDF text-annotation icons. Keep highlights only.
    try:
        import fitz

        for annotated_path in ANNOTATED.glob("*.pdf"):
            document = fitz.open(annotated_path)
            changed = False
            for pdf_page in document:
                annotation = pdf_page.first_annot
                while annotation:
                    next_annotation = annotation.next
                    if annotation.type[1] == "Text":
                        pdf_page.delete_annot(annotation)
                        changed = True
                    annotation = next_annotation
            if changed:
                document.saveIncr()
            document.close()
    except Exception:
        pass


def score(query: str, content: str):
    tokens = set(re.findall(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9_]+", query.lower()))
    return sum(content.lower().count(t) for t in tokens if len(t) > 1)


class ChatIn(BaseModel):
    question: str
    subject: str | None = None
    conversation_id: str | None = None


class RetrievalPreviewIn(BaseModel):
    question: str
    subject: str | None = None
    global_mode: bool = False


class TranslateIn(BaseModel):
    text: str


@app.get("/")
def home():
    return RedirectResponse("/main", status_code=307)


@app.get("/main")
def product_page():
    return FileResponse(ROOT / "static" / "landing.html")


@app.get("/login")
def login_page():
    return FileResponse(ROOT / "static" / "login.html")


@app.get("/app")
def app_page():
    return FileResponse(ROOT / "static" / "index.html")


class AuthIn(BaseModel):
    username: str
    password: str


def password_digest(password: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), b"course-kb-local-v1", 120000
    ).hex()


def current_model_name() -> str:
    with db() as c:
        saved = c.execute("SELECT value FROM settings WHERE key='model'").fetchone()
    return saved["value"] if saved else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")


@app.post("/api/auth/register")
def register(data: AuthIn):
    username = data.username.strip()
    if len(username) < 2 or len(data.password) < 6:
        raise HTTPException(400, "用户名至少 2 个字符，密码至少 6 个字符")
    try:
        with db() as c:
            c.execute(
                "INSERT INTO users VALUES (?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    username,
                    password_digest(data.password),
                    datetime.now().isoformat(),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "该用户名已存在，请直接登录") from exc
    return {"username": username, "message": "本地账号已创建"}


@app.post("/api/auth/login")
def login(data: AuthIn):
    with db() as c:
        user = c.execute(
            "SELECT username, password_hash FROM users WHERE username=?",
            (data.username.strip(),),
        ).fetchone()
    if not user or user["password_hash"] != password_digest(data.password):
        raise HTTPException(401, "用户名或密码不正确")
    return {"username": user["username"], "message": "登录成功"}


@app.post("/api/upload")
async def upload(
    file: Annotated[UploadFile, File(...)], subject: Annotated[str, Form(...)]
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "请上传 PDF 文件")
    file_id = str(uuid.uuid4())
    dest = PDFS / f"{file_id}.pdf"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chunks, entities, pages = extract_pdf(dest, file_id, subject.strip())
    with db() as c:
        c.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?)",
            (
                file_id,
                file.filename,
                subject.strip(),
                "text_pdf",
                str(dest),
                datetime.now().isoformat(),
            ),
        )
        c.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
        c.executemany("INSERT INTO ner_entities VALUES (?,?,?,?,?,?)", entities)
        c.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?)", pages)
    # Create the visible first layer immediately; the model then replaces it
    # with the enriched version once its page-by-page first reading completes.
    first_path = FIRST_MARKDOWN / f"{file_id}.md"
    first_path.write_text(
        local_first_markdown(file.filename, subject.strip(), pages), encoding="utf-8"
    )
    source_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
    prompt_hash, model_name = first_markdown_prompt_fingerprint(), current_model_name()
    with db() as c:
        cached = c.execute(
            "SELECT markdown_content FROM first_markdown_cache WHERE source_hash=? AND prompt_hash=? AND model_name=? AND subject=?",
            (source_hash, prompt_hash, model_name, subject.strip()),
        ).fetchone()
    if cached:
        first_markdown, first_status = (
            cached["markdown_content"],
            "命中首次阅读缓存（未调用模型）",
        )
    else:
        first_markdown, first_status = create_first_markdown(
            file.filename, subject.strip(), pages, deepseek
        )
        if first_status == "大模型首次阅读":
            with db() as c:
                c.execute(
                    "INSERT OR REPLACE INTO first_markdown_cache VALUES (?,?,?,?,?,?)",
                    (
                        source_hash,
                        prompt_hash,
                        model_name,
                        subject.strip(),
                        first_markdown,
                        datetime.now().isoformat(),
                    ),
                )
    first_path.write_text(first_markdown, encoding="utf-8")
    retriever.invalidate()
    return {
        "file_id": file_id,
        "chunks": len(chunks),
        "entities": len(entities),
        "first_markdown_status": first_status,
        "message": "已完成首次阅读并建立本地知识库",
    }


@app.get("/api/files")
def files():
    with db() as c:
        rows = c.execute(
            "SELECT f.*, COUNT(ch.chunk_id) chunk_count FROM files f LEFT JOIN chunks ch ON f.file_id=ch.file_id GROUP BY f.file_id ORDER BY upload_time DESC"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["first_markdown_ready"] = (
            FIRST_MARKDOWN / f"{item['file_id']}.md"
        ).exists()
        item["second_markdown_ready"] = (
            SECOND_MARKDOWN / f"{item['file_id']}.md"
        ).exists()
        result.append(item)
    return result


@app.get("/api/files/{file_id}/pdf")
def pdf(file_id: str):
    with db() as c:
        row = c.execute(
            "SELECT pdf_path FROM files WHERE file_id=?", (file_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "文件不存在")
    return FileResponse(row["pdf_path"], media_type="application/pdf")


@app.get("/api/files/{file_id}/markdown/{layer}")
def markdown_file(file_id: str, layer: str):
    folders = {"first": FIRST_MARKDOWN, "second": SECOND_MARKDOWN}
    folder = folders.get(layer)
    if not folder:
        raise HTTPException(404, "未知的 Markdown 层级")
    path = folder / f"{file_id}.md"
    if not path.exists():
        raise HTTPException(404, "该层 Markdown 尚未生成")
    return FileResponse(
        path, media_type="text/markdown; charset=utf-8", filename=path.name
    )


@app.get("/api/files/{file_id}/pages")
def file_pages(file_id: str):
    with db() as c:
        rows = c.execute(
            "SELECT page, content, words_json, width, height FROM pages WHERE file_id=? ORDER BY page",
            (file_id,),
        ).fetchall()
        file_row = c.execute(
            "SELECT pdf_path FROM files WHERE file_id=?", (file_id,)
        ).fetchone()
    # Courseware imported before reader support has no page index. Build it lazily.
    if not rows and file_row:
        try:
            import fitz

            document = fitz.open(file_row["pdf_path"])
            indexed = []
            for index, pdf_page in enumerate(document):
                words = pdf_page.get_text("words")
                indexed.append(
                    (
                        file_id,
                        index + 1,
                        " ".join(word[4] for word in words),
                        json.dumps(
                            [
                                {
                                    "x0": word[0],
                                    "y0": word[1],
                                    "x1": word[2],
                                    "y1": word[3],
                                    "text": word[4],
                                }
                                for word in words
                            ],
                            ensure_ascii=False,
                        ),
                        pdf_page.rect.width,
                        pdf_page.rect.height,
                    )
                )
            document.close()
            with db() as c:
                c.executemany(
                    "INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?)", indexed
                )
                rows = c.execute(
                    "SELECT page, content, words_json, width, height FROM pages WHERE file_id=? ORDER BY page",
                    (file_id,),
                ).fetchall()
        except Exception:
            rows = []
    if not rows:
        raise HTTPException(404, "该课件暂无可阅读文本，请重新上传后解析")
    with db() as c:
        notes = c.execute(
            "SELECT note_id, page, selected_text, note_content, created_at FROM notes WHERE file_id=? ORDER BY created_at",
            (file_id,),
        ).fetchall()
    notes_by_page = {}
    for note in notes:
        notes_by_page.setdefault(note["page"], []).append(dict(note))
    result = []
    for row in rows:
        data = dict(row)
        data["notes"] = notes_by_page.get(data["page"], [])
        result.append(data)
    return result


@app.get("/api/files/{file_id}/render")
def render_pdf_page(file_id: str, page: int):
    try:
        import fitz
    except ImportError as exc:
        raise HTTPException(500, "缺少 PDF 渲染组件") from exc
    with db() as c:
        row = c.execute(
            "SELECT pdf_path FROM files WHERE file_id=?", (file_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "文件不存在")
    annotated_path = ANNOTATED / f"{file_id}.pdf"
    doc = fitz.open(annotated_path if annotated_path.exists() else row["pdf_path"])
    if page < 1 or page > len(doc):
        doc.close()
        raise HTTPException(404, "页码不存在")
    pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    png = pix.tobytes("png")
    doc.close()
    return Response(png, media_type="image/png")


class NoteIn(BaseModel):
    file_id: str
    page: int
    selected_text: str
    note_content: str
    x0: float
    y0: float
    x1: float
    y1: float


@app.post("/api/notes")
def create_note(data: NoteIn):
    content = data.note_content.strip()
    if not content:
        raise HTTPException(400, "笔记内容不能为空")
    with db() as c:
        file_row = c.execute(
            "SELECT file_name, pdf_path, subject FROM files WHERE file_id=?",
            (data.file_id,),
        ).fetchone()
    if not file_row:
        raise HTTPException(404, "课件不存在")
    note_id, now = str(uuid.uuid4()), datetime.now().isoformat()
    with db() as c:
        c.execute(
            "INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                note_id,
                data.file_id,
                data.page,
                data.selected_text,
                content,
                data.x0,
                data.y0,
                data.x1,
                data.y1,
                now,
            ),
        )
        c.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
            (
                f"note_{note_id}",
                data.file_id,
                data.page,
                f"【我的笔记】原文：{data.selected_text}\n笔记：{content}",
                None,
                file_row["subject"],
            ),
        )
    try:
        import fitz

        annotated_path = ANNOTATED / f"{data.file_id}.pdf"
        if not annotated_path.exists():
            shutil.copy2(file_row["pdf_path"], annotated_path)
        document = fitz.open(annotated_path)
        target = document[data.page - 1]
        rect = fitz.Rect(data.x0, data.y0, data.x1, data.y1)
        highlight = target.add_highlight_annot(rect)
        highlight.set_colors(stroke=(1, 0.78, 0.2))
        highlight.update()
        document.saveIncr()
        document.close()
    except Exception as exc:
        raise HTTPException(500, f"PDF 批注保存失败：{exc}") from exc
    second_status = update_second_markdown(
        data.file_id,
        file_row["file_name"],
        file_row["subject"],
        db,
        deepseek,
        SECOND_MARKDOWN,
    )
    retriever.invalidate()
    return {
        "note_id": note_id,
        "second_markdown_status": second_status,
        "message": "笔记已同步到 PDF，并更新 second_markdown",
    }


def answer(data: ChatIn, global_mode=False):
    with db() as c:
        sql = (
            "SELECT * FROM chunks"
            if global_mode or not data.subject
            else "SELECT * FROM chunks WHERE subject=?"
        )
        rows = c.execute(
            sql, () if global_mode or not data.subject else (data.subject,)
        ).fetchall()
    ranked, retrieval_mode = retriever.retrieve(
        data.question, rows, candidate_k=20, evidence_k=5
    )
    documents = chunks_to_documents(ranked)
    citations = citations_from_documents(documents)
    if documents:
        summary = None
        try:
            model = make_deepseek_model(current_model_name(), max_tokens=800)
            if model is not None:
                summary = answer_from_documents(data.question, documents, model)
        except Exception as exc:
            logger.warning(
                "RAG generation unavailable (%s); using evidence fallback",
                type(exc).__name__,
            )
        if not summary:
            summary = "根据已检索到的课程资料，" + " ".join(
                f"《{doc.metadata['subject']}》第{doc.metadata['page']}页提到：{doc.page_content[:120]}"
                for doc in documents[:2]
            )
    else:
        summary = "DeepSeek 已连接，但在已导入的课件中没有检索到与该问题相关的内容。为避免脱离课件编造答案，系统不会直接生成回答；请换一个课件中的知识点，或上传相关资料。"
    with db() as c:
        conv = data.conversation_id or str(uuid.uuid4())
        if not data.conversation_id:
            c.execute(
                "INSERT INTO conversations VALUES (?,?,?,?)",
                (
                    conv,
                    "master_agent" if global_mode else "subject_agent",
                    None if global_mode else data.subject,
                    datetime.now().isoformat(),
                ),
            )
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), conv, "user", data.question, "[]", now),
        )
        c.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                conv,
                "assistant",
                summary,
                json.dumps(citations, ensure_ascii=False),
                now,
            ),
        )
    return {
        "answer": summary,
        "citations": citations,
        "conversation_id": conv,
        "agent": (
            "总控Agent（跨科目）"
            if global_mode
            else f"{data.subject or '当前'}科目Agent"
        ),
        "retrieval_mode": retrieval_mode,
    }


@app.post("/api/chat")
def chat(data: ChatIn):
    return answer(data)


@app.post("/api/chat/global")
def global_chat(data: ChatIn):
    return answer(data, True)


@app.post("/api/retrieval/preview")
def retrieval_preview(data: RetrievalPreviewIn):
    """Run local retrieval only; this endpoint never invokes an external LLM."""
    with db() as c:
        sql = (
            "SELECT * FROM chunks"
            if data.global_mode or not data.subject
            else "SELECT * FROM chunks WHERE subject=?"
        )
        rows = c.execute(
            sql, () if data.global_mode or not data.subject else (data.subject,)
        ).fetchall()
    evidence, mode = retriever.retrieve(
        data.question, rows, candidate_k=20, evidence_k=5
    )
    return {
        "retrieval_mode": mode,
        "evidence": [
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "subject": item.subject,
                "page": item.page,
                "snippet": item.content[:360],
                "bm25_rank": item.bm25_rank,
                "vector_rank": item.vector_rank,
                "rrf_score": item.rrf_score,
            }
            for rank, item in enumerate(evidence, start=1)
        ],
    }


@app.get("/api/retrieval/diagnostics")
def retrieval_diagnostics():
    """Expose local-index state without sending any course data to a cloud model."""
    return retriever.diagnostics()


@app.get("/api/get_ner_highlights")
def highlights(file_id: str, page: int):
    with db() as c:
        rows = c.execute(
            "SELECT * FROM ner_entities WHERE file_id=? AND page=?", (file_id, page)
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/translate")
def translate(data: TranslateIn):
    source = data.text.strip()
    with db() as c:
        old = c.execute(
            "SELECT result FROM translations WHERE source=?", (source,)
        ).fetchone()
        if old:
            return {"translation": old["result"], "cached": True}
        result = deepseek(
            [
                {
                    "role": "system",
                    "content": "你是学习翻译助手。英文翻译成中文，中文翻译成英文。只输出自然准确的译文，不要解释。",
                },
                {"role": "user", "content": source},
            ],
            300,
        ) or {
            "knowledge base": "知识库",
            "machine learning": "机器学习",
            "neural network": "神经网络",
            "signal": "信号",
            "system": "系统",
        }.get(
            source.lower(), "翻译服务暂时不可用，请检查网络或模型设置。"
        )
        c.execute("INSERT INTO translations VALUES (?,?)", (source, result))
    return {"translation": result, "cached": False}


@app.post("/api/screenshot_qa")
async def screenshot_qa(
    image: Annotated[UploadFile, File(...)],
    question: Annotated[str, Form()] = "请解释图片内容",
):
    suffix = Path(image.filename or "capture.png").suffix or ".png"
    path = STORAGE / "screenshots" / f"{uuid.uuid4()}{suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    mime = image.content_type or "image/png"
    image_data = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )
    response = deepseek(
        [
            {
                "role": "system",
                "content": "你是课程学习助手。分析图片中的题目、公式或课件内容，给出清晰的学习解释。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
        ],
        800,
    )
    return {
        "answer": response or "截图已保存，但模型暂未返回结果。请检查模型设置与网络。",
        "image_path": str(path.name),
    }


class SettingsIn(BaseModel):
    model: str


@app.get("/api/settings")
def get_settings():
    with db() as c:
        row = c.execute("SELECT value FROM settings WHERE key='model'").fetchone()
    return {
        "model": row["value"] if row else os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "models": ["deepseek-flash", "deepseek-v4-pro"],
        "api_connected": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


@app.put("/api/settings")
def update_settings(data: SettingsIn):
    if data.model not in {"deepseek-flash", "deepseek-v4-pro"}:
        raise HTTPException(400, "不支持的模型")
    with db() as c:
        c.execute("INSERT OR REPLACE INTO settings VALUES ('model',?)", (data.model,))
    return {"model": data.model, "message": "模型设置已保存"}


@app.get("/api/conversations")
def conversations():
    with db() as c:
        rows = c.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/chunks")
def chunk(chunk_id: str):
    with db() as c:
        row = c.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    if not row:
        raise HTTPException(404, "引用不存在")
    return dict(row)
