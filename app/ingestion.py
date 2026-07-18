"""
Ingestion pipelines: extraction -> validation -> chunking -> validation,
for both PDF and Python source files.

Design note: PDF and code are fundamentally different content, so they get
separate pipeline functions (process_pdf / process_code), but both live in
this one file because the four steps within each pipeline are tightly
coupled and should be read top-to-bottom as a single story, not scattered
across extract_*.py / chunk_*.py / validate_*.py files.

Every chunk this module produces is a dict with a common shape:
{
    "text": str,                 # the chunk content that gets embedded
    "chunk_type": str,           # "text" | "table" | "function" | "class"
    "page_num": int | None,
    "function_name": str | None,
    "start_line": int | None,
    "end_line": int | None,
    "token_count": int,
}
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

import fitz  # pymupdf
import pdfplumber

import ast
import re
from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np
import pdfplumber
import pytesseract

from PIL import Image
pytesseract.pytesseract.tesseract_cmd = (
    r"C:/Program Files/Tesseract-OCR/tesseract.exe"
)
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 - offline/restricted network fallback
    _ENCODER = None


def _count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Approximation used only when tiktoken's vocab file can't be fetched
    # (e.g. restricted network). ~4 chars/token is a standard rough estimate
    # for English text; good enough for chunk-size bounding, not for billing.
    return max(1, len(text) // 4)


@dataclass
class ValidationReport:
    stage: str
    checks: dict = field(default_factory=dict)
    passed: bool = True
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, value, ok: bool = True, warn: str | None = None):
        self.checks[name] = value
        if not ok:
            self.passed = False
        if warn:
            self.warnings.append(warn)

    def as_dict(self) -> dict:
        return {"stage": self.stage, "checks": self.checks, "passed": self.passed, "warnings": self.warnings}


# ======================================================================
# PDF PIPELINE
# ======================================================================

def _extract_pdf_tables_by_page(path: str) -> dict[int, list[str]]:
    """Return {page_num: [markdown_table_str, ...]} using pdfplumber, kept
    separate from body text so a table is never flattened into a paragraph."""
    tables_by_page: dict[int, list[str]] = {}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue
            md_tables = []
            for table in tables:
                if not table or not any(any(cell for cell in row) for row in table):
                    continue
                rows = ["| " + " | ".join((cell or "").strip() for cell in row) + " |" for row in table]
                header_sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
                md_tables.append("\n".join([rows[0], header_sep, *rows[1:]]))
            if md_tables:
                tables_by_page[i] = md_tables
    return tables_by_page

def _ocr_page(page: fitz.Page) -> str:
    """
    Convert a PDF page into an image and extract text using Tesseract OCR.
    """

    # Render page at high resolution
    pix = page.get_pixmap(dpi=300)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    img = np.array(img)

    # Convert RGB → Gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Noise removal
    gray = cv2.medianBlur(gray, 3)

    # Adaptive threshold
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config="--oem 3 --psm 6"
    )

    return text

def _extract_pdf_text_by_page(path: str) -> dict[int, str]:
    text_by_page = {}

    doc = fitz.open(path)

    for page_num, page in enumerate(doc, start=1):

        # First try native extraction
        text = page.get_text("text")

        # If very little text was found, use OCR
        if len(text.strip()) < 20:
            text = _ocr_page(page)

        text_by_page[page_num] = text

    doc.close()

    return text_by_page


def _extract_pdf_images_meta(path: str) -> list[dict]:
    """Return image references (page + index) without embedding pixel data -
    we store a pointer, not the raw image, per the spec's stated scope."""
    images = []
    doc = fitz.open(path)
    for page_index, page in enumerate(doc, start=1):
        for img_index, _img in enumerate(page.get_images(full=True)):
            images.append({"page_num": page_index, "image_index": img_index})
    doc.close()
    return images


def _strip_table_text_from_page(page_text: str, tables: list[str]) -> str:
    """Best-effort removal of table-like lines already captured as
    structured tables, so body-text chunking doesn't duplicate them.
    This is intentionally conservative: it only strips lines that look like
    dense whitespace-separated columns, since pdfplumber's text and table
    extraction don't share exact string boundaries."""
    lines = page_text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        looks_tabular = bool(re.match(r"^(\S+\s{2,}){2,}\S+$", stripped))
        if looks_tabular and len(stripped) > 0:
            continue
        kept.append(line)
    return "\n".join(kept)


def extract_pdf(path: str) -> tuple[dict, ValidationReport]:
    """Extract text, tables, and image metadata from a PDF, keyed by page.
    Returns (extraction_result, validation_report)."""
    report = ValidationReport(stage="pdf_extraction")

    text_by_page = _extract_pdf_text_by_page(path)
    tables_by_page = _extract_pdf_tables_by_page(path)
    images = _extract_pdf_images_meta(path)

    doc = fitz.open(path)
    actual_page_count = doc.page_count
    doc.close()

    extracted_page_count = len(text_by_page)
    report.add("actual_page_count", actual_page_count)
    report.add("extracted_page_count", extracted_page_count,
               ok=(extracted_page_count == actual_page_count),
               warn=None if extracted_page_count == actual_page_count
               else "Extracted page count does not match actual PDF page count")

    low_text_pages = [p for p, t in text_by_page.items() if len(t.strip()) < 20]
    report.add("low_text_pages", low_text_pages,
               ok=True,
               warn=(f"Pages with <20 chars of extracted text (likely image-only or blank): {low_text_pages}"
                     if low_text_pages else None))

    report.add("pages_with_tables", sorted(tables_by_page.keys()))
    report.add("total_tables_found", sum(len(v) for v in tables_by_page.values()))
    report.add("images_found", len(images))

    # Remove table-like lines from body text to reduce duplication between
    # a table chunk and a text chunk covering the same page.
    cleaned_text_by_page = {
        p: _strip_table_text_from_page(t, tables_by_page.get(p, []))
        for p, t in text_by_page.items()
    }

    return {
        "text_by_page": cleaned_text_by_page,
        "tables_by_page": tables_by_page,
        "images": images,
    }, report


def chunk_pdf(extraction: dict, token_size: int, overlap_tokens: int) -> tuple[list[dict], ValidationReport]:
    """Recursive token-based chunking for body text (paragraph boundaries
    first, then hard token splits), one chunk per table, kept atomic."""
    report = ValidationReport(stage="pdf_chunking")
    chunks: list[dict] = []

    for page_num, text in extraction["text_by_page"].items():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer = ""
        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if _count_tokens(candidate) <= token_size:
                buffer = candidate
                continue
            if buffer:
                chunks.append(_make_text_chunk(buffer, page_num))
            # paragraph itself may exceed token_size -> hard split with overlap
            if _count_tokens(para) > token_size:
                chunks.extend(_hard_split(para, page_num, token_size, overlap_tokens))
                buffer = ""
            else:
                buffer = para
        if buffer:
            chunks.append(_make_text_chunk(buffer, page_num))

    for page_num, tables in extraction["tables_by_page"].items():
        for table_md in tables:
            chunks.append({
                "text": table_md,
                "chunk_type": "table",
                "page_num": page_num,
                "function_name": None,
                "start_line": None,
                "end_line": None,
                "token_count": _count_tokens(table_md),
            })

    empty_chunks = [c for c in chunks if not c["text"].strip()]
    oversized = [c for c in chunks if c["token_count"] > token_size * 1.5 and c["chunk_type"] == "text"]
    table_chunks_missing_pipe = [
        c for c in chunks if c["chunk_type"] == "table" and "|" not in c["text"]
    ]

    report.add("total_chunks", len(chunks))
    report.add("empty_chunks", len(empty_chunks), ok=(len(empty_chunks) == 0),
               warn=None if not empty_chunks else "Found empty chunks after chunking")
    report.add("oversized_text_chunks", len(oversized), ok=(len(oversized) == 0),
               warn=None if not oversized else "Some text chunks exceed 1.5x token limit")
    report.add("malformed_table_chunks", len(table_chunks_missing_pipe),
               ok=(len(table_chunks_missing_pipe) == 0),
               warn=None if not table_chunks_missing_pipe else "Table chunk missing expected '|' delimiter structure")

    return chunks, report


def _make_text_chunk(text: str, page_num: int) -> dict:
    return {
        "text": text,
        "chunk_type": "text",
        "page_num": page_num,
        "function_name": None,
        "start_line": None,
        "end_line": None,
        "token_count": _count_tokens(text),
    }


def _hard_split(text: str, page_num: int, token_size: int, overlap_tokens: int) -> list[dict]:
    if _ENCODER is not None:
        tokens = _ENCODER.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + token_size, len(tokens))
            piece = _ENCODER.decode(tokens[start:end])
            chunks.append(_make_text_chunk(piece, page_num))
            if end == len(tokens):
                break
            start = end - overlap_tokens
        return chunks

    # Fallback: split on whitespace-delimited words, approximating token
    # count the same way _count_tokens does, so bounds stay consistent.
    words = text.split()
    approx_words_per_chunk = max(1, token_size * 4 // 5)  # ~4 chars/token, ~5 chars/word incl. space
    overlap_words = max(0, overlap_tokens * 4 // 5)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + approx_words_per_chunk, len(words))
        piece = " ".join(words[start:end])
        chunks.append(_make_text_chunk(piece, page_num))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


# ======================================================================
# CODE PIPELINE
# ======================================================================

def extract_code(path: str) -> tuple[dict, ValidationReport]:
    """Parse a .py file with ast and pull out top-level functions/classes
    as structural units, never splitting a function body mid-way."""
    report = ValidationReport(stage="code_extraction")

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        report.add("syntax_error", str(e), ok=False, warn=f"File failed to parse: {e}")
        return {"source": source, "units": [], "lines": lines}, report

    units = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            segment = "\n".join(lines[start - 1:end])
            units.append({
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "start_line": start,
                "end_line": end,
                "text": segment,
                "docstring": ast.get_docstring(node) or "",
            })

    independent_count = len(re.findall(r"^(async def |def |class )", source, flags=re.MULTILINE))
    report.add("actual_top_level_defs_regex", independent_count)
    report.add("extracted_units", len(units),
               ok=(len(units) == independent_count),
               warn=None if len(units) == independent_count
               else "AST-extracted unit count does not match regex-based def/class count "
                    "(expected if defs are nested inside classes/functions - verify manually)")
    report.add("total_lines", len(lines))

    return {"source": source, "units": units, "lines": lines}, report


def chunk_code(extraction: dict, max_tokens: int) -> tuple[list[dict], ValidationReport]:
    """One chunk per function/class. If a class is large, split by method
    but prepend the class name + docstring to every sub-chunk so a chunk
    is never returned headless."""
    report = ValidationReport(stage="code_chunking")
    chunks: list[dict] = []
    unparseable = 0

    for unit in extraction["units"]:
        if unit["kind"] == "class" and _count_tokens(unit["text"]) > max_tokens:
            class_header = f"class {unit['name']}:\n    \"\"\"{unit['docstring']}\"\"\"\n" if unit["docstring"] else f"class {unit['name']}:\n"
            try:
                class_tree = ast.parse(unit["text"])
                class_node = class_tree.body[0]
                lines = unit["text"].splitlines()
                for sub in class_node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        s, e = sub.lineno, getattr(sub, "end_lineno", sub.lineno)
                        method_text = "\n".join(lines[s - 1:e])
                        full_text = class_header + method_text
                        chunks.append({
                            "text": full_text,
                            "chunk_type": "class",
                            "page_num": None,
                            "function_name": f"{unit['name']}.{sub.name}",
                            "start_line": unit["start_line"] + s - 1,
                            "end_line": unit["start_line"] + e - 1,
                            "token_count": _count_tokens(full_text),
                        })
            except SyntaxError:
                unparseable += 1
                chunks.append(_code_chunk_from_unit(unit))
        else:
            chunks.append(_code_chunk_from_unit(unit))

    # validate each chunk parses on its own (methods won't, since they're
    # indented fragments prefixed with a synthetic class header - only
    # validate full function/class-level chunks)
    unparseable_full_units = 0
    for c in chunks:
        if c["chunk_type"] in ("function", "class") and "." not in (c["function_name"] or ""):
            try:
                ast.parse(c["text"])
            except SyntaxError:
                unparseable_full_units += 1

    empty_chunks = [c for c in chunks if not c["text"].strip()]

    report.add("total_chunks", len(chunks))
    report.add("empty_chunks", len(empty_chunks), ok=(len(empty_chunks) == 0),
               warn=None if not empty_chunks else "Found empty code chunks")
    report.add("unparseable_full_units", unparseable_full_units,
               ok=(unparseable_full_units == 0),
               warn=None if unparseable_full_units == 0 else "Some full function/class chunks failed ast.parse")

    return chunks, report


def _code_chunk_from_unit(unit: dict) -> dict:
    return {
        "text": unit["text"],
        "chunk_type": unit["kind"],
        "page_num": None,
        "function_name": unit["name"],
        "start_line": unit["start_line"],
        "end_line": unit["end_line"],
        "token_count": _count_tokens(unit["text"]),
    }


# ======================================================================
# PUBLIC ENTRYPOINTS
# ======================================================================

def process_pdf(path: str, token_size: int, overlap_tokens: int) -> tuple[list[dict], list[ValidationReport]]:
    extraction, extract_report = extract_pdf(path)
    chunks, chunk_report = chunk_pdf(extraction, token_size, overlap_tokens)
    return chunks, [extract_report, chunk_report]


def process_code(path: str, max_tokens: int) -> tuple[list[dict], list[ValidationReport]]:
    extraction, extract_report = extract_code(path)
    chunks, chunk_report = chunk_code(extraction, max_tokens)
    return chunks, [extract_report, chunk_report]
