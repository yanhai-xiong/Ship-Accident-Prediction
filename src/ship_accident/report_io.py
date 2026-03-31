"""Extract plain text from maritime accident report files (PDF, DOC, DOCX, WPS)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_pdf(path: Path, *, max_pages: int | None = None) -> str:
    """Extract text using pypdf (fast). If max_pages is set, only read first N pages."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    n = len(reader.pages)
    if max_pages is not None:
        n = min(n, max_pages)
    parts: list[str] = []
    for i in range(n):
        try:
            t = reader.pages[i].extract_text()
        except Exception as e:  # noqa: BLE001
            logger.warning("PDF page %s extract failed %s: %s", i, path.name, e)
            t = ""
        if t:
            parts.append(t)
    return "\n".join(parts)


def extract_text_doc_like(path: Path, *, max_chars: int = 2_000_000) -> str:
    """macOS: use ``textutil`` for legacy .doc / .wps."""
    r = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if r.returncode != 0:
        logger.warning("textutil failed %s: %s", path.name, r.stderr[:200])
    s = r.stdout or ""
    if len(s) > max_chars:
        logger.warning("Truncating %s from %s to %s chars", path.name, len(s), max_chars)
        return s[:max_chars]
    return s


def extract_text_docx(path: Path) -> str:
    import docx

    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def extract_report_text(path: Path, *, pdf_max_pages: int | None = None) -> str:
    """Dispatch by extension. ``pdf_max_pages`` limits PDF work (e.g. first pages for headers)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return extract_text_pdf(p, max_pages=pdf_max_pages)
    if suf == ".docx":
        return extract_text_docx(p)
    if suf in {".doc", ".wps"}:
        return extract_text_doc_like(p)
    raise ValueError(f"Unsupported extension: {p}")
