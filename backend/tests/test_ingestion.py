from pathlib import Path

import fitz
from docx import Document as DocxDocument

from app.ingestion.chunking import chunk_segments
from app.ingestion.extractors import TextSegment, extract_text


def test_txt_extraction_retains_markdown_heading():
    segments = extract_text("guide.txt", b"# Security\n\nRotate credentials monthly.\n")
    assert segments == [TextSegment("Rotate credentials monthly.", heading="Security")]


def test_docx_extraction_retains_heading():
    document = DocxDocument()
    document.add_heading("Retention", level=1)
    document.add_paragraph("Records are retained for seven years.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Record"
    table.cell(0, 1).text = "Retention period"
    from io import BytesIO

    output = BytesIO()
    document.save(output)
    segments = extract_text("policy.docx", output.getvalue())
    assert segments[0].heading == "Retention"
    assert "seven years" in segments[0].text
    assert "Record | Retention period" in segments[1].text


def test_pdf_extraction_keeps_one_based_page_number():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "The policy begins here.")
    content = pdf.tobytes()
    pdf.close()
    segments = extract_text("policy.pdf", content)
    assert segments[0].page_number == 1
    assert "policy begins" in segments[0].text


def test_image_extraction_runs_ocr(monkeypatch):
    from app.ingestion import extractors

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
    content = pixmap.tobytes("png")
    monkeypatch.setattr(extractors, "_ocr_page", lambda _page, **_kwargs: "Text from picture")

    assert extractors.validate_file("picture.png", content) == "image/png"
    segments = extractors.extract_text("picture.png", content)
    assert [(segment.text, segment.page_number) for segment in segments] == [
        ("Text from picture", 1)
    ]


def test_scanned_pdf_page_uses_ocr(monkeypatch):
    from app.ingestion import extractors

    document = fitz.open()
    document.new_page()
    content = document.tobytes()
    document.close()
    monkeypatch.setattr(extractors, "_ocr_page", lambda _page, **_kwargs: "Scanned page text")

    segments = extractors.extract_text("scan.pdf", content)

    assert [(segment.text, segment.page_number) for segment in segments] == [
        ("Scanned page text", 1)
    ]


def test_ocr_recognizer_supports_configured_fallback_layout():
    """The alternate OCR pass should use its configured page layout mode."""
    from app.ingestion.extractors import _recognize_image

    class Output:
        DICT = object()

    class Tesseract:
        def image_to_data(self, _image, *, lang, config, output_type):
            assert lang == "eng"
            assert config == "--oem 1 --psm 6"
            assert output_type is Output.DICT
            return {
                "text": ["Readable", "text"],
                "conf": ["90", "85"],
                "block_num": ["1", "1"],
                "par_num": ["1", "1"],
                "line_num": ["1", "1"],
                "page_num": ["1", "1"],
            }

    assert _recognize_image(
        None,
        "eng",
        Tesseract(),
        Output,
        page_segmentation_mode=6,
    ) == ("Readable text", 87.5)


def test_chunking_preserves_heading_and_bounds_chunk_length():
    segments = [TextSegment("alpha " * 20, page_number=3, heading="Details")]
    chunks = chunk_segments(segments, chunk_size=40, overlap=5)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert all(chunk.page_number == 3 and chunk.heading == "Details" for chunk in chunks)
    assert all("Details" not in chunk.text for chunk in chunks)
    assert all(chunk.embed_text and "Section: Details" in chunk.embed_text for chunk in chunks)


def test_upload_persists_quality_report(client, monkeypatch, upload_dir: Path):
    """Processed documents should expose parser and chunk diagnostics."""
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    response = client.post(
        "/api/v1/documents",
        files={
            "file": ("quality.txt", b"# Access\n\nUse a secure connection for work.", "text/plain")
        },
    )
    assert response.status_code == 201
    details = client.get(f"/api/v1/documents/{response.json()['id']}").json()
    report = details["quality_report"]
    assert report["parser_used"] == "utf-8"
    assert report["chunk_count"] == len(details["chunks"])
    assert "token_size_min_median_max" in report


def test_upload_duplicate_versioning_detail_and_delete(client, monkeypatch, upload_dir: Path):
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    payload = b"# Access\n\nEmployees must use multi-factor authentication."

    first = client.post("/api/v1/documents", files={"file": ("policy.txt", payload, "text/plain")})
    assert first.status_code == 201
    first_doc = first.json()
    assert first_doc["status"] == "queued"
    assert first_doc["version"] == 1

    detail = client.get(f"/api/v1/documents/{first_doc['id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"
    assert detail.json()["chunks"][0]["heading"] == "Access"

    duplicate = client.post(
        "/api/v1/documents", files={"file": ("policy.txt", payload, "text/plain")}
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["document_id"] == first_doc["id"]

    next_version = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", b"Updated content.", "text/plain")},
    )
    assert next_version.status_code == 201
    assert next_version.json()["version"] == 2

    listing = client.get("/api/v1/documents?limit=1&offset=0")
    assert listing.status_code == 200
    assert listing.json()["total"] == 2
    assert len(listing.json()["items"]) == 1

    stored = upload_dir / f"{first_doc['id']}.txt"
    assert stored.exists()
    deleted = client.delete(f"/api/v1/documents/{first_doc['id']}")
    assert deleted.status_code == 204
    assert not stored.exists()
    assert client.get(f"/api/v1/documents/{first_doc['id']}").status_code == 404


def test_upload_rejects_unsupported_file(client):
    response = client.post(
        "/api/v1/documents", files={"file": ("script.exe", b"nope", "application/octet-stream")}
    )
    assert response.status_code == 422


def test_upload_enforces_size_limit(client, monkeypatch):
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"max_upload_bytes": 4})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    response = client.post(
        "/api/v1/documents", files={"file": ("large.txt", b"12345", "text/plain")}
    )
    assert response.status_code == 422


def test_upload_queues_ingestion_and_returns_before_processing(
    client, monkeypatch, upload_dir: Path
):
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(document_routes, "process_queued_document", lambda *_args: None)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("queued.txt", b"Processing happens after upload.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "queued"


def test_failed_ingestion_can_be_retried(client, monkeypatch, upload_dir: Path):
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings
    from app.ingestion.extractors import IngestionError
    from app.services import document_ingestion

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    original_extract = document_ingestion.extract_text

    def fail_once(_filename, _content):
        raise IngestionError("temporary parser error")

    monkeypatch.setattr(document_ingestion, "extract_text", fail_once)
    uploaded = client.post(
        "/api/v1/documents",
        files={"file": ("retry.txt", b"Retry this document.", "text/plain")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["status"] == "queued"
    failed = client.get(f"/api/v1/documents/{uploaded.json()['id']}").json()
    assert failed["status"] == "failed"
    assert failed["retry_count"] == 0

    monkeypatch.setattr(document_ingestion, "extract_text", original_extract)
    retried = client.post(f"/api/v1/documents/{uploaded.json()['id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "ready"
    assert retried.json()["retry_count"] == 1


def test_qdrant_permission_failure_has_actionable_ingestion_status(
    client, monkeypatch, upload_dir: Path
):
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings
    from app.services import document_ingestion

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)

    class QdrantForbiddenError(Exception):
        status_code = 403

    class ForbiddenVectorService:
        def index_document(self, **_kwargs):
            raise QdrantForbiddenError()

    monkeypatch.setattr(
        document_ingestion, "get_dense_vector_service", lambda: ForbiddenVectorService()
    )
    response = client.post(
        "/api/v1/documents",
        files={"file": ("permission.txt", b"Text to index.", "text/plain")},
    )

    assert response.status_code == 201
    failed = client.get(f"/api/v1/documents/{response.json()['id']}").json()
    assert failed["status"] == "failed"
    assert "Qdrant denied collection access" in failed["error_message"]
