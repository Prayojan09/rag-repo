"""Structure-aware chunking with section context and deterministic overlap."""

import re
from dataclasses import dataclass

from app.ingestion.extractors import TextSegment
from app.ingestion.metadata import count_tokens


@dataclass(frozen=True)
class ChunkData:
    text: str
    page_number: int | None
    heading: str | None
    embed_text: str | None = None
    section_path: tuple[str, ...] = ()
    token_count: int = 0
    tokenizer: str = "unknown"


def chunk_segments(
    segments: list[TextSegment],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ChunkData]:
    from app.core.config import get_settings

    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size_chars
    overlap = overlap if overlap is not None else settings.chunk_overlap_chars
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")
    chunks: list[ChunkData] = []
    buffer = ""
    pending_question = ""
    active_question = ""
    current_page: int | None = None
    current_heading: str | None = None

    def heading_prefix(text: str) -> str:
        heading = (current_heading or "").strip()
        if not heading or text.lstrip().casefold().startswith(heading.casefold()):
            return ""
        if len(heading) + 1 >= chunk_size:
            return heading[: max(1, chunk_size // 4)]
        return heading

    def body_limit() -> int:
        prefix = heading_prefix(buffer)
        return max(1, chunk_size - len(prefix) - (1 if prefix else 0))

    def store(text: str) -> None:
        body = text.strip()
        if not body:
            return
        context = [f"Section: {heading_prefix(body)}" if heading_prefix(body) else ""]
        if active_question and active_question.casefold() not in body.casefold():
            context.append(f"Related question: {active_question}")
        context = [item for item in context if item]
        embed_text = "\n".join([*context, body]) if context else body
        section_path = (current_heading,) if current_heading else ()
        token_count, tokenizer = count_tokens(body)
        chunks.append(
            ChunkData(
                body,
                current_page,
                current_heading,
                embed_text,
                section_path,
                token_count,
                tokenizer,
            )
        )

    def flush() -> None:
        nonlocal buffer
        text = buffer.strip()
        if text:
            store(text)
        buffer = ""

    for segment in segments:
        text = _normalize_list_markers(segment.text).strip()
        if not text:
            continue
        if buffer and (segment.page_number != current_page or segment.heading != current_heading):
            if segment.page_number != current_page:
                same_section = (
                    not segment.heading
                    or segment.heading.casefold() == (current_heading or "").casefold()
                )
                pending_question = _trailing_question(buffer) if same_section else ""
            flush()
        current_page = segment.page_number
        current_heading = segment.heading
        heading_question = _trailing_question(segment.heading or "")
        if heading_question:
            active_question = heading_question
        if pending_question:
            text = f"{pending_question}\n{text}"
            pending_question = ""
        if (
            buffer
            and len(buffer) + len(text) + 2 > body_limit()
            and not _ends_with_question(buffer)
        ):
            previous = buffer
            flush()
            tail = previous[-overlap:].lstrip() if overlap and not active_question else ""
            buffer = f"{tail}\n{text}" if tail else text
        else:
            buffer = f"{buffer}\n{text}" if buffer else text

        prefix = heading_prefix(buffer)
        heading_cost = len(prefix) + (1 if prefix else 0)
        while len(buffer) > body_limit() and (
            not _ends_with_question(buffer) or len(buffer) + heading_cost > chunk_size
        ):
            limit = body_limit()
            split_at = _preferred_split(buffer, limit)
            piece = buffer[:split_at].strip()
            if piece:
                store(piece)
            remainder = buffer[split_at:].strip()
            tail = (
                buffer[max(0, split_at - overlap) : split_at].strip()
                if overlap and not active_question
                else ""
            )
            buffer = f"{tail} {remainder}".strip()
        question = _trailing_question(text)
        if question:
            active_question = question
    flush()
    return chunks


def _ends_with_question(text: str) -> bool:
    return bool(re.search(r"\?\s*['\"\u2019\u201d]?\s*$", text))


def _trailing_question(text: str) -> str:
    match = re.search(r"([^.!?]*\?)\s*['\"\u2019\u201d]?\s*$", text)
    return match.group(1).strip() if match else ""


def _normalize_list_markers(text: str) -> str:
    """Turn common PDF/DOCX bullet glyphs into clean line-separated list items."""
    return re.sub(
        r"[ \t]*[\uF0B7\uF0A7\u2022\u25AA\u25E6\u25CF][ \t]*",
        "\n" + chr(0x2022) + " ",
        text,
    )


def _preferred_split(text: str, limit: int) -> int:
    """Prefer paragraph/sentence boundaries and keep a question with its answer."""
    question_boundaries = [
        match.start() for match in re.finditer(r"(?<=\?)\s+", text) if match.start() < limit
    ]
    sentence_boundaries = [
        match.start()
        for match in re.finditer(r"(?<=[.!;:])\s+", text)
        if limit // 2 <= match.start() < limit
    ]
    if sentence_boundaries:
        return sentence_boundaries[-1]
    if question_boundaries:
        after_question = next(
            (
                match.start()
                for match in re.finditer(r"(?<=[.!?])\s+", text)
                if match.start() > question_boundaries[-1]
            ),
            None,
        )
        if after_question is not None and after_question <= limit + max(200, limit // 4):
            return after_question
    paragraph_boundary = text.rfind("\n", 0, limit + 1)
    if paragraph_boundary >= limit // 2:
        return paragraph_boundary
    split_at = text.rfind(" ", 0, limit + 1)
    return split_at if split_at >= limit // 2 else limit
