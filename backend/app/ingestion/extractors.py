"""Text extraction for supported document formats, retaining source locations."""

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from zipfile import BadZipFile, ZipFile

import fitz
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)
_handwriting_model_lock = Lock()
MAX_HANDWRITING_LINES_PER_PAGE = 100


class IngestionError(ValueError):
    """The uploaded file is unsupported, invalid, or cannot be safely extracted."""


@dataclass(frozen=True)
class TextSegment:
    text: str
    page_number: int | None = None
    heading: str | None = None


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}
MAX_EXTRACTED_CHARS = 5_000_000
MAX_PDF_PAGES = 1_000
MAX_OCR_PAGES = 500
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
OCR_IMAGE_COVERAGE_THRESHOLD = 0.2
MIN_NATIVE_TEXT_CHARS_FOR_OCR_SKIP = 200


def validate_file(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            "Supported formats are PDF, DOCX, TXT, Markdown, HTML, PNG, JPG, TIFF, and BMP"
        )
    if not content:
        raise IngestionError("The uploaded file is empty")
    from app.ingestion.loaders.base import sniff_document_type

    detected_type = sniff_document_type(filename, content)
    aliases = {".jpeg": ".jpg", ".tiff": ".tif", ".htm": ".html", ".markdown": ".md"}
    normalized_extension = aliases.get(extension, extension)
    if normalized_extension != detected_type:
        raise IngestionError("The file extension does not match its content type")
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise IngestionError("The file does not have a valid PDF signature")
    if extension == ".docx":
        try:
            with ZipFile(BytesIO(content)) as archive:
                entries = archive.infolist()
                if (
                    len(entries) > 10_000
                    or sum(item.file_size for item in entries) > MAX_DOCX_UNCOMPRESSED_BYTES
                ):
                    raise IngestionError("The DOCX archive exceeds the processing limit")
                if "word/document.xml" not in archive.namelist():
                    raise IngestionError("The DOCX file is missing its document content")
        except BadZipFile as exc:
            raise IngestionError("The DOCX file is not a valid Office document") from exc
    if extension == ".txt":
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError("TXT files must use UTF-8 encoding") from exc
    if extension in {".md", ".markdown", ".html", ".htm"}:
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError("HTML and Markdown files must use UTF-8 encoding") from exc
    if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            with fitz.open(stream=content, filetype=extension[1:]) as image:
                if not image.page_count or image.page_count > MAX_OCR_PAGES:
                    raise IngestionError("The image has an unsupported number of pages")
                page = image[0]
                if page.rect.width * page.rect.height > 100_000_000:
                    raise IngestionError("Image dimensions exceed the processing limit")
        except (fitz.FileDataError, RuntimeError) as exc:
            raise IngestionError("The image file could not be opened") from exc
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }[extension]


def extract_text(filename: str, content: bytes) -> list[TextSegment]:
    extension = Path(filename).suffix.lower()
    if extension in {".html", ".htm", ".md", ".markdown"}:
        from app.ingestion.loaders.base import get_loader

        parsed = get_loader(filename, content).load(filename, content)
        from app.ingestion.normalize import normalize_segments

        return normalize_segments(parsed.to_segments())
    if extension == ".pdf":
        segments = _extract_pdf(content)
    elif extension == ".docx":
        segments = _extract_docx(content)
    elif extension == ".txt":
        segments = _extract_txt(content)
    elif extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        segments = _extract_image(content, extension[1:])
    else:
        raise IngestionError("Unsupported document type")
    from app.ingestion.normalize import normalize_segments

    return normalize_segments(segments)


def _extract_pdf(content: bytes) -> list[TextSegment]:
    segments: list[TextSegment] = []
    total_chars = 0
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            if pdf.page_count > MAX_PDF_PAGES:
                raise IngestionError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit")
            ocr_pages = 0
            heading: str | None = None
            for page_index, page in enumerate(pdf):
                page_text = page.get_text("text", sort=True).strip()
                # Large embedded images are common in otherwise text-based PDFs
                # (page backgrounds, screenshots, charts). OCR is expensive, so
                # only run it on image-heavy pages when their native text layer is
                # missing or too sparse to be useful.
                needs_ocr = not page_text or (
                    len(page_text) < MIN_NATIVE_TEXT_CHARS_FOR_OCR_SKIP
                    and _has_significant_images(page)
                )
                if needs_ocr:
                    ocr_pages += 1
                    if ocr_pages > MAX_OCR_PAGES:
                        raise IngestionError(f"PDF exceeds the {MAX_OCR_PAGES}-page OCR limit")
                    ocr_started = time.perf_counter()
                    page_text = _ocr_page(page, page_number=page_index + 1)
                    logger.info(
                        "PDF OCR page completed (page=%d/%d ocr_pages=%d elapsed_seconds=%.2f)",
                        page_index + 1,
                        pdf.page_count,
                        ocr_pages,
                        time.perf_counter() - ocr_started,
                    )
                    if page_text:
                        segments.append(TextSegment(page_text, page_index + 1, heading))
                        total_chars += len(page_text)
                        if total_chars > MAX_EXTRACTED_CHARS:
                            raise IngestionError("Extracted text exceeds the processing limit")
                        continue
                page_dict = page.get_text("dict", sort=True)
                if page_text and not any(
                    span["text"].strip()
                    for block in page_dict["blocks"]
                    if "lines" in block
                    for line in block["lines"]
                    for span in line["spans"]
                ):
                    segments.append(TextSegment(page_text, page_index + 1))
                    total_chars += len(page_text)
                    if total_chars > MAX_EXTRACTED_CHARS:
                        raise IngestionError("Extracted text exceeds the processing limit")
                    continue
                spans = [
                    span
                    for block in page_dict["blocks"]
                    if "lines" in block
                    for line in block["lines"]
                    for span in line["spans"]
                    if span["text"].strip()
                ]
                font_sizes = sorted(span["size"] for span in spans)
                median_size = font_sizes[len(font_sizes) // 2] if font_sizes else 0
                for block in page_dict["blocks"]:
                    if "lines" not in block:
                        continue
                    block_text = " ".join(
                        span["text"].strip()
                        for line in block["lines"]
                        for span in line["spans"]
                        if span["text"].strip()
                    ).strip()
                    if not block_text:
                        continue
                    block_spans = [
                        span
                        for line in block["lines"]
                        for span in line["spans"]
                        if span["text"].strip()
                    ]
                    sizes = [span["size"] for span in block_spans]
                    has_bold_text = any(span["flags"] & 16 for span in block_spans)
                    is_heading = (
                        len(block_text) <= 180
                        and bool(sizes)
                        and (
                            (median_size > 0 and max(sizes) >= median_size * 1.2)
                            or _looks_like_structural_heading(block_text)
                            or (has_bold_text and len(block_text) <= 100)
                        )
                    )
                    if is_heading:
                        heading = block_text
                    else:
                        segments.append(TextSegment(block_text, page_index + 1, heading))
                    total_chars += len(block_text)
                    if total_chars > MAX_EXTRACTED_CHARS:
                        raise IngestionError("Extracted text exceeds the processing limit")
    except IngestionError:
        raise
    except Exception as exc:
        logger.warning("PyMuPDF PDF extraction failed; trying pdfplumber fallback")
        try:
            import pdfplumber

            with pdfplumber.open(BytesIO(content)) as pdf:
                fallback_segments = [
                    TextSegment(page.extract_text() or "", page_number=index + 1)
                    for index, page in enumerate(pdf.pages)
                    if (page.extract_text() or "").strip()
                ]
            if fallback_segments:
                return fallback_segments
        except Exception as fallback_error:
            logger.warning(
                "PDF fallback extraction failed (error_type=%s)",
                type(fallback_error).__name__,
            )
        if isinstance(exc, fitz.FileDataError):
            raise IngestionError("The PDF could not be opened") from exc
        raise IngestionError("The PDF text could not be extracted") from exc
    if not any(segment.text.strip() for segment in segments):
        raise IngestionError("No extractable text was found in the PDF")
    return segments


def _has_significant_images(page: fitz.Page) -> bool:
    """OCR image heavy pages while skipping small logos on text based pages."""
    page_area = page.rect.get_area()
    if page_area <= 0:
        return False
    image_area = 0.0
    for image in page.get_images(full=True):
        for image_rect in page.get_image_rects(image[0]):
            image_area += (image_rect & page.rect).get_area()
    return image_area / page_area >= OCR_IMAGE_COVERAGE_THRESHOLD


def _looks_like_structural_heading(text: str) -> bool:
    """Recognize common document hierarchy labels when font sizes are uniform."""
    from app.core.config import get_settings

    value = text.strip().lstrip("#* ").strip()
    if not value or len(value) > get_settings().ingestion_heading_max_chars:
        return False
    numbered = re.match(r"^\d+(?:\.\d+)*[.)]?\s+(.+)$", value)
    if numbered:
        title = numbered.group(1)
        if ("." in value and re.match(r"^\d+(?:\.\d+)+", value)) or (
            len(title.split()) <= 7 and not re.search(r"[,;.!?]", title)
        ):
            return True
    if value.endswith(":") and len(value.split()) <= 12:
        return True
    letters = [character for character in value if character.isalpha()]
    if (
        len(value.split()) <= 12
        and len(letters) >= 4
        and all(character.isupper() for character in letters)
    ):
        return True
    words = value.split()
    title_case_words = [word for word in words if word[:1].isalpha()]
    return (
        1 < len(words) <= 8
        and not re.search(r"[.!?]$", value)
        and bool(title_case_words)
        and sum(word[0].isupper() for word in title_case_words) / len(title_case_words) >= 0.75
    )


def _ocr_page(
    page: fitz.Page,
    *,
    page_number: int | None = None,
    correct_perspective: bool = False,
) -> str:
    from app.core.config import get_settings

    settings = get_settings()
    _configure_tesseract(settings.tesseract_cmd)
    if not shutil.which("tesseract"):
        raise IngestionError(
            "Tesseract OCR was not found. Add tesseract.exe to PATH or set TESSERACT_CMD, "
            "then restart the backend."
        )
    rendered_pixels = (page.rect.width * settings.ocr_dpi / 72) * (
        page.rect.height * settings.ocr_dpi / 72
    )
    if rendered_pixels > 100_000_000:
        raise IngestionError("Page dimensions exceed the OCR processing limit")
    try:
        import cv2
        import numpy as np
        import pytesseract
        from pytesseract import Output

        scale = settings.ocr_dpi / 72
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False
        )
        image = (
            np.frombuffer(pixmap.samples, dtype=np.uint8)
            .reshape(pixmap.height, pixmap.width)
            .copy()
        )
        enhanced, binary = _prepare_ocr_images(
            image, cv2, np, correct_perspective=correct_perspective
        )
        primary_text, primary_confidence = _recognize_image(
            enhanced, settings.ocr_language, pytesseract, Output
        )
        if (
            primary_confidence >= settings.handwriting_ocr_min_confidence
            and len(primary_text) >= 20
        ):
            return primary_text

        # Adaptive thresholding often recovers faint handwriting and unevenly lit
        # phone photos. Run it before the slower handwriting model and keep whichever
        # Tesseract pass reports higher confidence.
        best_text, best_confidence = primary_text, primary_confidence
        if primary_confidence < settings.handwriting_ocr_min_confidence:
            fallback_text, fallback_confidence = _recognize_image(
                binary,
                settings.ocr_language,
                pytesseract,
                Output,
                page_segmentation_mode=settings.ocr_fallback_page_segmentation_mode,
            )
            if fallback_text and (
                fallback_confidence > best_confidence or not best_text
            ):
                best_text, best_confidence = fallback_text, fallback_confidence

        if (
            len(best_text) >= 20
            and best_confidence >= settings.handwriting_ocr_min_confidence
        ):
            return best_text

        # TrOCR is useful for handwriting but can misread printed text. Only use it
        # after both conventional OCR passes were weak, and never let an empty model
        # result erase readable Tesseract output.
        if settings.handwriting_ocr_model:
            try:
                handwriting_text = _recognize_handwriting(
                    enhanced,
                    cv2,
                    np,
                    settings.handwriting_ocr_model,
                    settings.handwriting_ocr_batch_size,
                )
                if handwriting_text and len(handwriting_text) >= len(best_text) * 0.7:
                    return handwriting_text
            except Exception as exc:
                logger.error(
                    "Handwriting recognition failed (error_type=%s detail=%s)",
                    type(exc).__name__,
                    str(exc)[:300],
                    extra={"error_type": type(exc).__name__},
                )
                if not best_text:
                    raise IngestionError(
                        "Handwriting OCR could not start. Install the backend dependencies with "
                        "'python -m pip install -r requirements.txt', restart the backend, and "
                        "ensure Hugging Face is reachable for the model's first download."
                    ) from exc
        return best_text
    except Exception as exc:
        if isinstance(exc, IngestionError):
            raise
        if isinstance(exc, ModuleNotFoundError):
            raise IngestionError(
                "Image enhancement dependencies are missing. Install backend/requirements.txt "
                "and restart the backend."
            ) from exc
        message = str(exc).lower()
        location = f" on PDF page {page_number}" if page_number else ""
        if "language" in message or "traineddata" in message:
            raise IngestionError(
                f"Tesseract language data '{settings.ocr_language}' is unavailable{location}. "
                "Install that language pack or set OCR_LANGUAGE to an installed language."
            ) from exc
        raise IngestionError(
            f"Tesseract OCR failed{location} ({type(exc).__name__}). "
            "Check the Tesseract installation and available memory, then retry."
        ) from exc


def _prepare_ocr_images(image, cv2, np, *, correct_perspective: bool):
    """Create contrast-enhanced and locally thresholded page images for OCR."""
    if correct_perspective:
        image = _correct_document_perspective(image, cv2, np)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.medianBlur(enhanced, 3)
    enhanced = _deskew_image(enhanced, cv2, np)

    minimum_dimension = min(enhanced.shape[:2])
    block_size = min(31, minimum_dimension if minimum_dimension % 2 else minimum_dimension - 1)
    if block_size < 3:
        return enhanced, enhanced
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        11,
    )
    return enhanced, binary


def _correct_document_perspective(image, cv2, np):
    """Flatten a clearly visible four-corner paper boundary in a photo upload."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or cv2.contourArea(polygon) < image_area * 0.3:
            continue

        corners = polygon.reshape(4, 2).astype("float32")
        sums = corners.sum(axis=1)
        differences = np.diff(corners, axis=1).reshape(4)
        ordered = np.array(
            [
                corners[np.argmin(sums)],
                corners[np.argmax(differences)],
                corners[np.argmax(sums)],
                corners[np.argmin(differences)],
            ],
            dtype="float32",
        )
        top_left, top_right, bottom_right, bottom_left = ordered
        width = int(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
        height = int(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
        if width < 200 or height < 200 or width * height > 40_000_000:
            continue
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        return cv2.warpPerspective(
            image, transform, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
    return image


def _deskew_image(image, cv2, np):
    """Correct slight camera or scanner rotation without rotating page orientation."""
    height, width = image.shape[:2]
    scale = min(1.0, 1400 / max(height, width))
    sample = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )
    _, foreground = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    points = cv2.findNonZero(foreground)
    if points is None or len(points) < 50:
        return image
    angle = cv2.minAreaRect(points)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.15 or abs(angle) > 8:
        return image
    center = (width / 2, height / 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _recognize_image(
    image,
    language: str,
    pytesseract,
    output,
    *,
    page_segmentation_mode: int = 3,
) -> tuple[str, float]:
    """Recognize a page and return its text with Tesseract's mean word confidence."""
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=f"--oem 1 --psm {page_segmentation_mode}",
        output_type=output.DICT,
    )
    lines: dict[tuple[str, str, str, str], list[str]] = {}
    confidences: list[float] = []
    for index, raw_word in enumerate(data["text"]):
        word = raw_word.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1
        if confidence >= 0:
            confidences.append(confidence)
        key = tuple(
            str(data[field][index]) for field in ("block_num", "par_num", "line_num", "page_num")
        )
        lines.setdefault(key, []).append(word)
    text = "\n".join(" ".join(words) for words in lines.values()).strip()
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence


def _recognize_handwriting(image, cv2, np, model_name: str, batch_size: int) -> str:
    """Use a line-level handwriting model when general OCR confidence is low."""
    import torch
    from PIL import Image

    line_images = _handwriting_line_crops(image, cv2, np)
    if not line_images or len(line_images) > MAX_HANDWRITING_LINES_PER_PAGE:
        return ""

    with _handwriting_model_lock:
        processor, model = _load_handwriting_model(model_name)
        output_lines: list[str] = []
        for start in range(0, len(line_images), batch_size):
            batch = [
                Image.fromarray(np.repeat(crop[:, :, None], 3, axis=2))
                for crop in line_images[start : start + batch_size]
            ]
            pixel_values = processor(images=batch, return_tensors="pt").pixel_values
            with torch.inference_mode():
                generated = model.generate(
                    pixel_values,
                    max_new_tokens=96,
                    num_beams=1,
                )
            output_lines.extend(processor.batch_decode(generated, skip_special_tokens=True))
    return "\n".join(line.strip() for line in output_lines if line.strip())


@lru_cache(maxsize=1)
def _load_handwriting_model(model_name: str):
    from importlib.metadata import PackageNotFoundError, version

    import torch

    try:
        version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError("The transformers package is not installed") from exc

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    from app.core.config import get_settings

    torch.set_num_threads(get_settings().handwriting_ocr_cpu_threads)
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return processor, model


def _handwriting_line_crops(image, cv2, np):
    """Remove notebook rules, find text rows, and return padded line images."""
    from app.core.config import get_settings

    settings = get_settings()
    image = cv2.medianBlur(image, 5)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height, width = image.shape[:2]
    ink = 255 - binary
    rule_width = max(40, width // 10)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rule_width, 1))
    horizontal_rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horizontal_kernel)
    text_ink = cv2.subtract(ink, horizontal_rules)

    row_ink = np.count_nonzero(text_ink, axis=1)
    threshold = max(
        settings.handwriting_line_min_ink_pixels,
        int(width * settings.handwriting_line_ink_fraction),
    )
    active_rows = np.flatnonzero(row_ink >= threshold)
    if not len(active_rows):
        return []

    merge_gap = max(2, int(height * settings.handwriting_line_merge_gap_fraction))
    groups: list[tuple[int, int]] = []
    group_start = previous = int(active_rows[0])
    for row in active_rows[1:]:
        row = int(row)
        if row - previous > merge_gap:
            groups.append((group_start, previous + 1))
            group_start = row
        previous = row
    groups.append((group_start, previous + 1))

    crops = []
    vertical_padding = max(8, height // 150)
    horizontal_padding = max(12, width // 100)
    for top, bottom in groups:
        if bottom - top < 10:
            continue
        top = max(0, top - vertical_padding)
        bottom = min(height, bottom + vertical_padding)
        line_ink = text_ink[top:bottom]
        columns = np.flatnonzero(np.count_nonzero(line_ink, axis=0) >= 2)
        if len(columns) < 15:
            continue
        left = max(0, int(columns[0]) - horizontal_padding)
        right = min(width, int(columns[-1]) + horizontal_padding + 1)
        crop = 255 - text_ink[top:bottom, left:right]
        if crop.shape[1] >= crop.shape[0] * 1.3:
            crops.append(crop)
    return crops


def _configure_tesseract(configured_path: str | None) -> None:
    """Make a Windows Tesseract install discoverable by PyMuPDF's OCR helper."""
    if shutil.which("tesseract"):
        return
    candidates = [Path(configured_path)] if configured_path else []
    if os.name == "nt":
        candidates.extend(
            [
                Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR",
                Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
                / "Tesseract-OCR",
            ]
        )
    for candidate in candidates:
        executable = candidate if candidate.is_file() else candidate / "tesseract.exe"
        if executable.is_file():
            directory = str(executable.parent)
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
            return


def _extract_image(content: bytes, filetype: str) -> list[TextSegment]:
    try:
        with fitz.open(stream=content, filetype=filetype) as image:
            segments = []
            for index, page in enumerate(image):
                text = _ocr_page(page, page_number=index + 1, correct_perspective=True)
                if text:
                    segments.append(TextSegment(text, page_number=index + 1))
                if sum(len(segment.text) for segment in segments) > MAX_EXTRACTED_CHARS:
                    raise IngestionError("Extracted text exceeds the processing limit")
    except Exception as exc:
        if isinstance(exc, IngestionError):
            raise
        raise IngestionError("The image could not be processed") from exc
    if not segments:
        raise IngestionError("No text could be read from the image")
    return segments


def _extract_docx(content: bytes) -> list[TextSegment]:
    segments: list[TextSegment] = []
    heading: str | None = None
    total_chars = 0
    try:
        document = DocxDocument(BytesIO(content))
    except Exception as exc:
        raise IngestionError("The DOCX document could not be read") from exc
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = paragraph.style.name.lower() if paragraph.style else ""
            is_bold_title = len(text) <= 140 and any(
                run.bold and run.text.strip() for run in paragraph.runs
            )
            if (
                style_name.startswith("heading")
                or style_name == "title"
                or is_bold_title
                or _looks_like_structural_heading(text)
            ):
                heading = text
            else:
                if style_name.startswith("list bullet"):
                    text = f"{chr(0x2022)} {text}"
                elif style_name.startswith("list number"):
                    text = f"1. {text}"
                segments.append(TextSegment(text, heading=heading))
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = [
                " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
                for row in table.rows
            ]
            text = "\n".join(row for row in rows if row.strip(" |"))
            if text:
                segments.append(TextSegment(text, heading=heading))
        else:
            continue
        total_chars += len(text)
        if total_chars > MAX_EXTRACTED_CHARS:
            raise IngestionError("Extracted text exceeds the processing limit")
    if not segments:
        raise IngestionError("No extractable text was found in the DOCX")
    return segments


def _extract_txt(content: bytes) -> list[TextSegment]:
    text = content.decode("utf-8-sig")
    if len(text) > MAX_EXTRACTED_CHARS:
        raise IngestionError("Extracted text exceeds the processing limit")
    segments: list[TextSegment] = []
    heading: str | None = None
    for paragraph in text.splitlines():
        line = paragraph.strip()
        if not line:
            continue
        if line.startswith("#") or _looks_like_structural_heading(line):
            heading = line.lstrip("# ").strip()[:500] or heading
        else:
            segments.append(TextSegment(line, heading=heading))
    if not segments:
        raise IngestionError("No extractable text was found in the TXT file")
    return segments
