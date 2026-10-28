"""PDF text extraction, page indexing and initial retrieval chunk generation."""
import json
import re
import uuid
from pathlib import Path


def split_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be smaller than size")
    text = re.sub(r"\s+", " ", text).strip()
    parts, position = [], 0
    while position < len(text):
        end = min(len(text), position + size)
        if end < len(text):
            # A boundary near the start can move the next overlapping window
            # only one character forward. Search the latter half of the window.
            minimum_cut = position + size // 2
            boundaries = [
                text.rfind("。", minimum_cut, end),
                text.rfind("！", minimum_cut, end),
                text.rfind("？", minimum_cut, end),
                text.rfind(". ", minimum_cut, end),
                text.rfind("; ", minimum_cut, end),
            ]
            cut = max(boundaries)
            if cut >= minimum_cut:
                end = cut + 1
        part = text[position:end].strip()
        if len(part) < 8 and parts:
            parts[-1] = f"{parts[-1]} {part}".strip()
        elif part:
            parts.append(part)
        if end >= len(text):
            break
        position = max(end - overlap, position + 1)
    return parts


def detect_terms(text: str) -> list[tuple[str, str]]:
    keywords = ["卷积神经网络", "神经网络", "傅里叶变换", "拉普拉斯变换", "采样定理", "向量", "矩阵", "算法", "模型", "特征", "损失函数", "概率", "信号", "系统"]
    return [(term, "COURSE_TERM" if len(term) <= 4 else "COURSE_CONCEPT") for term in keywords if term in text]


def extract_pdf(path: Path, file_id: str, subject: str):
    import fitz
    chunks, entities, pages = [], [], []
    document = fitz.open(path)
    for index, page in enumerate(document):
        words = page.get_text("words")
        text = " ".join(word[4] for word in words)
        page_words = [{"x0": word[0], "y0": word[1], "x1": word[2], "y1": word[3], "text": word[4]} for word in words]
        page_number = index + 1
        pages.append((file_id, page_number, text, json.dumps(page_words, ensure_ascii=False), page.rect.width, page.rect.height))
        bboxes = [[round(word[0], 1), round(word[1], 1), round(word[2], 1), round(word[3], 1)] for word in words]
        for chunk_index, chunk_content in enumerate(split_text(text)):
            chunks.append((f"{file_id}_p{page_number}_{chunk_index}", file_id, page_number, chunk_content, json.dumps(bboxes, ensure_ascii=False), subject))
        for term, term_type in detect_terms(text):
            rects = page.search_for(term)
            entities.append((str(uuid.uuid4()), file_id, page_number, term, term_type, json.dumps([[rect.x0, rect.y0, rect.x1, rect.y1] for rect in rects])))
    document.close()
    return chunks, entities, pages
