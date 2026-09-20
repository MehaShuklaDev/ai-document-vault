"""Text extraction with page tracking.

Output is a list of ``Page`` objects; page numbers survive into chunks so chat
answers can cite "doc.pdf p.4". DOCX has no pages, so we synthesise one "page"
per top-level heading section, which gives citations a meaningful anchor.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_TXT = "text/plain"
MIME_MD = "text/markdown"


class ExtractionError(Exception):
    pass


@dataclass
class Page:
    number: int
    text: str
    section: str | None = None


def sniff_mime(data: bytes, filename: str, declared: str | None) -> str:
    """Trust magic bytes over the client-declared content type."""
    head = data[:8]
    if head.startswith(b"%PDF"):
        return MIME_PDF
    if head.startswith(b"PK\x03\x04") and filename.lower().endswith(".docx"):
        return MIME_DOCX
    lower = filename.lower()
    if lower.endswith((".md", ".markdown")):
        return MIME_MD
    if lower.endswith(".txt") or declared in (MIME_TXT, MIME_MD):
        return MIME_TXT
    # last resort: if it decodes as UTF-8 and has no NUL bytes, call it text
    if b"\x00" not in data[:4096]:
        try:
            data[:4096].decode("utf-8")
            return MIME_TXT
        except UnicodeDecodeError:
            pass
    return declared or "application/octet-stream"


_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL.sub("\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes, max_pages: int) -> list[Page]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:  # pragma: no cover
                raise ExtractionError("PDF is password protected") from e
        n = len(reader.pages)
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Could not open PDF: {e}") from e
    if n > max_pages:
        raise ExtractionError(f"PDF has {n} pages; limit is {max_pages}")
    pages: list[Page] = []
    for i, p in enumerate(reader.pages, start=1):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        txt = clean_text(txt)
        if txt:
            pages.append(Page(number=i, text=txt))
    if not pages:
        raise ExtractionError("PDF contains no extractable text (scanned image? OCR not enabled)")
    return pages


def _extract_docx(data: bytes) -> list[Page]:
    import docx  # python-docx

    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ExtractionError(f"Could not open DOCX: {e}") from e

    pages: list[Page] = []
    current: list[str] = []
    section: str | None = None
    num = 1

    def flush():
        nonlocal current, num
        txt = clean_text("\n".join(current))
        if txt:
            pages.append(Page(number=num, text=txt, section=section))
            num += 1
        current = []

    for para in d.paragraphs:
        style = (para.style.name or "").lower() if para.style is not None else ""
        t = para.text.strip()
        if not t:
            continue
        if style.startswith("heading") or style == "title":
            flush()
            section = t
            current.append(f"# {t}")
        else:
            current.append(t)
    # tables → pipe-delimited rows so numbers stay attached to their labels
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                current.append(" | ".join(cells))
    flush()
    if not pages:
        raise ExtractionError("DOCX contains no text")
    return pages


def _extract_text(data: bytes) -> list[Page]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = clean_text(text)
    if not text:
        raise ExtractionError("File is empty")
    # synthesise pages every ~3000 chars so citations stay useful for long text files
    pages, size = [], 3000
    for i in range(0, len(text), size):
        pages.append(Page(number=i // size + 1, text=text[i : i + size]))
    return pages


def extract_pages(data: bytes, mime_type: str, *, max_pages: int = 500) -> list[Page]:
    if mime_type == MIME_PDF:
        return _extract_pdf(data, max_pages)
    if mime_type == MIME_DOCX:
        return _extract_docx(data)
    if mime_type in (MIME_TXT, MIME_MD):
        return _extract_text(data)
    raise ExtractionError(f"Unsupported content type: {mime_type}")
