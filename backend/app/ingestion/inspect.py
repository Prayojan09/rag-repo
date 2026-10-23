"""CLI for inspecting source extraction, normalization, structure, and chunks."""

import argparse
import json
from pathlib import Path

from app.ingestion.chunking import chunk_segments
from app.ingestion.extractors import IngestionError, validate_file
from app.ingestion.loaders import get_loader
from app.ingestion.metadata import document_sha256
from app.ingestion.normalize import normalize_segments_with_removed
from app.ingestion.report import build_quality_report, save_quality_report
from app.ingestion.structure import section_paths


def inspect_file(
    path: Path,
    *,
    show_clean: bool = False,
    show_chunks: bool = False,
    report_only: bool = False,
) -> dict[str, object]:
    """Inspect one supported file and return a JSON-serializable quality summary."""
    content = path.read_bytes()
    validate_file(path.name, content)
    parsed = get_loader(path.name, content).load(path.name, content)
    segments, removed_lines = normalize_segments_with_removed(parsed.to_segments())
    chunks = chunk_segments(segments)
    report = build_quality_report(
        parsed,
        segments,
        chunks,
        filename=path.name,
        removed_header_footer_lines=removed_lines,
    )
    save_quality_report(f"inspect-{document_sha256(content)}", report)
    if report_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    print("RAW EXTRACTED TEXT\n" + "\n\n".join(block.text for block in parsed.blocks))
    if show_clean:
        print("\nCLEANED TEXT\n" + "\n\n".join(segment.text for segment in segments))
    print(
        "\nSTRUCTURE\n"
        + json.dumps(
            [
                {"text": segment.text, "section_path": list(section)}
                for segment, section in section_paths(segments)
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    if show_chunks:
        print(
            "\nCHUNKS\n"
            + json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2)
        )
    print("\nQUALITY REPORT\n" + json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    """Parse CLI options and inspect one input document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--show-clean", action="store_true")
    parser.add_argument("--show-chunks", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print and save only the quality report, without extracted document text.",
    )
    args = parser.parse_args()
    try:
        inspect_file(
            args.file,
            show_clean=args.show_clean,
            show_chunks=args.show_chunks,
            report_only=args.report_only,
        )
    except (OSError, IngestionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
