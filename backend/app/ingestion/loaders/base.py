"""Shared document loader types and content-type based loader selection."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.extractors import (
    MIN_NATIVE_TEXT_CHARS_FOR_OCR_SKIP,
    IngestionError,
    TextSegment,
    _extract_docx,
    _extract_image,
    _extract_pdf,
    _extract_txt,
    _has_significant_images,
)


@dataclass(frozen=True)
class Block:
    """One ordered piece of extracted document content with source metadata."""

    type: str
    text: str
    heading_level: int | None = None
    page_number: int | None = None
    font_size: float | None = None
    heading: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Common parsed document representation returned by every loader."""

    blocks: list[Block]
    parser_used: str
    page_count: int | None = None
    ocr_pages: tuple[int, ...] | None = None

    def to_segments(self) -> list[TextSegment]:
        """Convert blocks to the legacy ingestion representation."""
        segments: list[TextSegment] = []
        active_heading: str | None = None
        for block in self.blocks:
            if block.type == "heading":
                active_heading = block.text
            segments.append(
                TextSegment(
                    block.text,
                    block.page_number,
                    block.heading or active_heading,
                )
            )
        return segments


class DocumentLoader(ABC):
    """Interface implemented by document format parsers."""

    @abstractmethod
    def load(self, filename: str, content: bytes) -> ParsedDocument:
        """Parse the supplied file bytes and retain source structure."""


class SegmentLoader(DocumentLoader):
    """Adapter for existing PDF, DOCX, TXT, and image extraction routines."""

    def load(self, filename: str, content: bytes) -> ParsedDocument:
        """Adapt the current production extractors to the loader interface."""
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            segments = _extract_pdf(content)
        elif suffix == ".docx":
            segments = _extract_docx(content)
        elif suffix == ".txt":
            segments = _extract_txt(content)
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            segments = _extract_image(content, suffix[1:])
        else:
            raise IngestionError("Unsupported document type")
        blocks = [
            Block(
                type="heading"
                if segment.heading and segment.text == segment.heading
                else "paragraph",
                text=segment.text,
                page_number=segment.page_number,
                heading=segment.heading,
            )
            for segment in segments
        ]
        page_count: int | None = None
        ocr_pages: tuple[int, ...] | None = None
        if suffix == ".pdf":
            import fitz

            with fitz.open(stream=content, filetype="pdf") as pdf:
                page_count = pdf.page_count
                ocr_pages = tuple(
                    index + 1
                    for index, page in enumerate(pdf)
                    if not page.get_text("text").strip()
                    or (
                        len(page.get_text("text").strip()) < MIN_NATIVE_TEXT_CHARS_FOR_OCR_SKIP
                        and _has_significant_images(page)
                    )
                )
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            import fitz

            with fitz.open(stream=content, filetype=suffix[1:]) as image:
                page_count = image.page_count
                ocr_pages = tuple(range(1, image.page_count + 1))
        return ParsedDocument(
            blocks,
            parser_used={
                ".pdf": "pymupdf",
                ".docx": "python-docx",
                ".txt": "utf-8",
                ".png": "tesseract",
                ".jpg": "tesseract",
                ".jpeg": "tesseract",
                ".tif": "tesseract",
                ".tiff": "tesseract",
                ".bmp": "tesseract",
            }.get(suffix, "unknown"),
            page_count=page_count,
            ocr_pages=ocr_pages,
        )


class HTMLLoader(DocumentLoader):
    """Extract headings, paragraphs, lists, tables, and code from HTML."""

    def load(self, filename: str, content: bytes) -> ParsedDocument:
        """Parse HTML while discarding navigation and executable elements."""
        from html.parser import HTMLParser

        class Parser(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.blocks: list[Block] = []
                self.stack: list[tuple[str, int]] = []
                self.buffer: list[str] = []
                self.skip_depth = 0

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag in {"script", "style", "nav", "footer", "header", "noscript"}:
                    self.skip_depth += 1
                    return
                if self.skip_depth:
                    return
                if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre"}:
                    self._flush()
                    self.stack.append((tag, len(self.blocks)))
                if tag == "br" and self.buffer:
                    self.buffer.append(" ")

            def handle_endtag(self, tag: str) -> None:
                if (
                    tag in {"script", "style", "nav", "footer", "header", "noscript"}
                    and self.skip_depth
                ):
                    self.skip_depth -= 1
                    return
                if self.skip_depth:
                    return
                if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre"}:
                    self._flush()
                    if self.stack and self.stack[-1][0] == tag:
                        self.stack.pop()

            def handle_data(self, data: str) -> None:
                if not self.skip_depth:
                    self.buffer.append(data)

            def _flush(self) -> None:
                value = " ".join(" ".join(self.buffer).split())
                self.buffer.clear()
                if not value:
                    return
                tag = self.stack[-1][0] if self.stack else "p"
                kind = (
                    "heading"
                    if tag.startswith("h") and tag[1:].isdigit()
                    else "list_item"
                    if tag == "li"
                    else "table"
                    if tag == "tr"
                    else "code"
                    if tag == "pre"
                    else "paragraph"
                )
                level = int(tag[1]) if kind == "heading" else None
                self.blocks.append(Block(kind, value, level))

        try:
            parser = Parser()
            parser.feed(content.decode("utf-8", errors="replace"))
            parser.close()
            parser._flush()
        except Exception as exc:
            raise IngestionError("The HTML document could not be read") from exc
        if not parser.blocks:
            raise IngestionError("No extractable text was found in the HTML document")
        return ParsedDocument(parser.blocks, "html.parser")


class MarkdownLoader(DocumentLoader):
    """Parse Markdown headings, lists, code blocks, tables, and paragraphs."""

    def load(self, filename: str, content: bytes) -> ParsedDocument:
        """Convert Markdown source lines to ordered structured blocks."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError("Markdown files must use UTF-8 encoding") from exc
        blocks: list[Block] = []
        code: list[str] = []
        in_code = False
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("```") or line.startswith("~~~"):
                if in_code:
                    blocks.append(Block("code", "\n".join(code)))
                    code.clear()
                in_code = not in_code
                continue
            if in_code:
                code.append(raw)
                continue
            if not line:
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading:
                blocks.append(Block("heading", heading.group(2).strip(), len(heading.group(1))))
            elif re.match(r"^(?:[-*+] |\d+[.)] )", line):
                blocks.append(Block("list_item", line))
            elif "|" in line:
                blocks.append(Block("table", line))
            else:
                blocks.append(Block("paragraph", line))
        if code:
            blocks.append(Block("code", "\n".join(code)))
        if not blocks:
            raise IngestionError("No extractable text was found in the Markdown document")
        return ParsedDocument(blocks, "markdown")


def sniff_document_type(filename: str, content: bytes) -> str:
    """Return a supported extension using file signatures and conservative sniffing."""
    suffix = Path(filename).suffix.lower()
    if content.startswith(b"%PDF-"):
        return ".pdf"
    if content.startswith(b"PK\x03\x04"):
        return ".docx" if suffix == ".docx" else ""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"BM"):
        return ".bmp"
    if content.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    prefix = content[:2048].lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return ".html"
    if suffix in {".html", ".htm"}:
        raise IngestionError("The file content does not appear to be HTML")
    if suffix in {".md", ".markdown", ".txt"}:
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError("Text and Markdown files must use UTF-8 encoding") from exc
        return ".md" if suffix == ".markdown" else suffix
    aliases = {".jpeg": ".jpg", ".tiff": ".tif", ".htm": ".html", ".markdown": ".md"}
    if suffix in {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return aliases.get(suffix, suffix)
    raise IngestionError(
        "Unsupported document type. Upload a PDF, DOCX, TXT, Markdown, HTML, or image file."
    )


def get_loader(filename: str, content: bytes) -> DocumentLoader:
    """Select a loader by validated content type and extension."""
    extension = sniff_document_type(filename, content)
    aliases = {".jpeg": ".jpg", ".tiff": ".tif", ".htm": ".html", ".markdown": ".md"}
    if aliases.get(Path(filename).suffix.lower(), Path(filename).suffix.lower()) != extension:
        raise IngestionError("The file extension does not match its content type")
    if extension in {".html", ".htm"}:
        return HTMLLoader()
    if extension in {".md", ".markdown"}:
        return MarkdownLoader()
    return SegmentLoader()
