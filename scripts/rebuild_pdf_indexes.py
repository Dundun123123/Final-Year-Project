"""Rebuild derived PDF chunks, page text, and NER entries after extractor changes.

Original PDFs and user notes remain untouched. Note chunks are retained because
they are authored by the user rather than regenerated from a source PDF.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.pdf_extractor import extract_pdf


def main() -> None:
    database_path = ROOT / "course_kb.db"
    connection = sqlite3.connect(database_path)
    try:
        files = connection.execute("SELECT file_id, pdf_path, subject FROM files ORDER BY upload_time").fetchall()
        for file_id, pdf_path, subject in files:
            path = Path(pdf_path)
            if not path.exists():
                print(f"skip missing file_id={file_id}")
                continue
            chunks, entities, pages = extract_pdf(path, file_id, subject)
            with connection:
                connection.execute("DELETE FROM chunks WHERE file_id=? AND chunk_id NOT LIKE 'note_%'", (file_id,))
                connection.execute("DELETE FROM ner_entities WHERE file_id=?", (file_id,))
                connection.execute("DELETE FROM pages WHERE file_id=?", (file_id,))
                connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
                connection.executemany("INSERT INTO ner_entities VALUES (?,?,?,?,?,?)", entities)
                connection.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?)", pages)
            print(f"rebuilt file_id={file_id} chunks={len(chunks)} pages={len(pages)}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
