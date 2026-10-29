"""Two-stage Markdown generation for course PDFs and personal notes."""

import hashlib
import inspect
from pathlib import Path

from prompts.markdown_prompts import (
    first_markdown_system_prompt,
    first_markdown_user_prompt,
    second_markdown_system_prompt,
    second_markdown_user_prompt,
)


def _clean_model_markdown(value: str) -> str:
    return value.strip().replace("```markdown", "").replace("```", "").strip()


def first_markdown_prompt_fingerprint() -> str:
    """Changes whenever either first-pass prompt function is edited."""
    source = inspect.getsource(first_markdown_system_prompt) + inspect.getsource(
        first_markdown_user_prompt
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def local_first_markdown(file_name: str, subject: str, pages) -> str:
    header = f"# {file_name}\n\n> 科目：{subject}\n> 处理层：首次阅读（本地文本回退）\n"
    body = "\n\n".join(
        f"## 第 {page_no} 页\n\n{content}" for _, page_no, content, *_ in pages
    )
    return f"{header}\n{body}\n"


def create_first_markdown(
    file_name: str, subject: str, pages, llm_call
) -> tuple[str, str]:
    """Build first_markdown by asking the selected model to read each PDF page."""
    system_prompt = first_markdown_system_prompt()
    processed_pages, model_used = [], False
    for _, page_no, content, *_ in pages:
        if not content.strip():
            processed_pages.append(f"## 第 {page_no} 页\n\n*本页未提取到可读文本。*")
            continue
        user_prompt = first_markdown_user_prompt(file_name, subject, page_no, content)
        result = llm_call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1400,
        )
        if result:
            model_used = True
            processed_pages.append(
                f"## 第 {page_no} 页\n\n{_clean_model_markdown(result)}"
            )
        else:
            processed_pages.append(f"## 第 {page_no} 页\n\n{content}")
    status = "大模型首次阅读" if model_used else "本地文本回退（模型不可用）"
    markdown = (
        f"# {file_name}\n\n> 科目：{subject}\n> 处理层：{status}\n\n"
        + "\n\n".join(processed_pages)
        + "\n"
    )
    return markdown, status


def local_second_markdown(file_name: str, subject: str, notes) -> str:
    grouped = {}
    for note in notes:
        grouped.setdefault(note["page"], []).append(note)
    sections = []
    for page, page_notes in grouped.items():
        items = "\n\n".join(
            f"### 原文锚点\n> {note['selected_text']}\n\n### 我的理解\n{note['note_content']}\n\n### 待复习点\n- 回到第 {page} 页复习这段原文。"
            for note in page_notes
        )
        sections.append(f"## 第 {page} 页\n\n{items}")
    return (
        f"# {file_name} · 个人学习笔记\n\n> 科目：{subject}\n> 来源：second_markdown（本地整理回退）\n\n"
        + "\n\n".join(sections)
        + "\n"
    )


def update_second_markdown(
    file_id: str,
    file_name: str,
    subject: str,
    db_connect,
    llm_call,
    second_markdown_dir: Path,
) -> str:
    """Regenerate second_markdown after every saved PDF annotation."""
    with db_connect() as connection:
        notes = [
            dict(row)
            for row in connection.execute(
                "SELECT page, selected_text, note_content, created_at FROM notes WHERE file_id=? ORDER BY page, created_at",
                (file_id,),
            ).fetchall()
        ]
    raw_notes = "\n\n".join(
        f"[第 {note['page']} 页]\n原文锚点：{note['selected_text']}\n我的笔记：{note['note_content']}"
        for note in notes
    )
    result = (
        llm_call(
            [
                {"role": "system", "content": second_markdown_system_prompt()},
                {
                    "role": "user",
                    "content": second_markdown_user_prompt(
                        file_name, subject, raw_notes
                    ),
                },
            ],
            max_tokens=2000,
        )
        if raw_notes
        else None
    )
    if result:
        content = f"# {file_name} · 个人学习笔记\n\n> 科目：{subject}\n> 来源：second_markdown（大模型整理）\n\n{_clean_model_markdown(result)}\n"
        status = "大模型整理"
    else:
        content, status = (
            local_second_markdown(file_name, subject, notes),
            "本地整理回退",
        )
    (second_markdown_dir / f"{file_id}.md").write_text(content, encoding="utf-8")
    return status
