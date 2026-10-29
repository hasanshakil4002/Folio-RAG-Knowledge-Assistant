"""PDF text extraction (plain or OCR) and chunking."""
import re
from pathlib import Path

import pymupdf # PyMuPDF

from . import config as cfg

# Sentence boundaries: . ! ? plus the Devanagari/Bengali danda marks, or a blank line.
_SPLIT = re.compile(r"(?<=[.!?\u0964\u0965])\s+|\n{2,}")


def _tesseract():
    import pytesseract
    if cfg.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD
    return pytesseract


def ocr_ready() -> tuple[bool, str]:
    """Check that Tesseract is installed and usable."""
    try:
        _tesseract().get_tesseract_version()
        return True, ""
    except Exception:
        return False, ("Tesseract OCR is not installed. Install it (see README) "
                       "or choose 'OCR not needed'.")


def _ocr_lang() -> str:
    """Use the requested OCR languages that are installed; fall back to what exists."""
    try:
        have = set(_tesseract().get_languages(config=""))
    except Exception:
        return "eng"
    want = [x for x in cfg.OCR_LANG.split("+") if x in have]
    return "+".join(want) if want else ("eng" if "eng" in have else "+".join(sorted(have)))


def _normalize(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # join hard line wraps
    return text.strip()


def iter_pages(path: Path, use_ocr: bool):
    """Yield (page_number, total_pages, text) for every page."""
    from PIL import Image

    doc = pymupdf.open(path)
    try:
        total = len(doc)
        lang = _ocr_lang() if use_ocr else None
        tess = _tesseract() if use_ocr else None
        for i, page in enumerate(doc, 1):
            if use_ocr:
                pix = page.get_pixmap(dpi=cfg.OCR_DPI, colorspace=pymupdf.csRGB, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = tess.image_to_string(img, lang=lang)
            else:
                text = page.get_text("text")
            yield i, total, _normalize(text)
    finally:
        doc.close()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for u in _SPLIT.split(text):
        u = (u or "").strip()
        while len(u) > size:
            units.append(u[:size])
            u = u[size:]
        if u:
            units.append(u)

    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for u in units:
        if cur and n + len(u) + 1 > size:
            chunks.append(" ".join(cur))
            keep, k = [], 0
            for s in reversed(cur):
                if k + len(s) > overlap:
                    break
                keep.insert(0, s)
                k += len(s) + 1
            cur, n = keep, k
        cur.append(u)
        n += len(u) + 1
    if cur:
        last = " ".join(cur)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return [c for c in chunks if len(c.strip()) >= 20]
