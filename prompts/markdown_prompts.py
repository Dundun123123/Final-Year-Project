"""Versioned prompt functions for the two-stage Markdown knowledge pipeline."""


def first_markdown_system_prompt() -> str:
    """Rules for the initial PDF-to-first_markdown reading pass."""
    return """你是课程资料结构化助手。请只根据提供的 PDF 单页文本进行第一次阅读，输出可直接保存的 Markdown。

要求：
1. 保留原有事实、定义、公式和章节层级。
2. 不要补充原文没有的信息。
3. 清理 OCR 中明显重复的空白。
4. 使用简短标题、段落和列表。
5. 在开头保留页码标记。
6. 不要使用代码围栏。
7. 保持原文可以提取的文字，按从左到右、从上到下的顺序输出，不要遗漏。"""


def first_markdown_user_prompt(
    file_name: str, subject: str, page_number: int, page_text: str
) -> str:
    """Context provided to the first-pass model for one PDF page."""
    return f"""文件：{file_name}
科目：{subject}
页码：{page_number}

PDF 原文：
{page_text[:9000]}"""


def second_markdown_system_prompt() -> str:
    """Rules for converting raw annotations into the personal knowledge layer."""
    return """你是个人学习笔记整理助手。请只根据提供的课件笔记输出 Markdown 学习档案。

要求：
1. 按页码或主题组织。
2. 每条保留“原文锚点、我的理解、待复习点”。
3. 不要虚构课程结论。
4. 相似笔记可以合并但不得丢失信息。
5. 使用清晰、简洁的中文。
6. 不要使用代码围栏。"""


def second_markdown_user_prompt(file_name: str, subject: str, raw_notes: str) -> str:
    """Context provided to the note-organising model after every saved annotation."""
    return f"""课件：{file_name}
科目：{subject}

笔记记录：
{raw_notes[:30000]}"""
