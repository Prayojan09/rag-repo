"""Tests for generic text normalization defects found in extracted documents."""

from app.ingestion.extractors import TextSegment
from app.ingestion.normalize import (
    normalize_hyphenation,
    normalize_segments,
    normalize_segments_with_removed,
    normalize_spacing,
    normalize_unicode,
)


def test_unicode_and_control_characters_are_normalized() -> None:
    """Compatibility characters and invisible controls should be removed."""
    assert normalize_unicode("ﬁle\u200b — ‘text’\x00") == "file - 'text'"


def test_stray_spaces_before_punctuation_are_removed() -> None:
    """Whitespace before punctuation should not survive extraction cleanup."""
    assert (
        normalize_spacing("attacks . frameworks , values : yes !")
        == "attacks. frameworks, values: yes!"
    )


def test_wrapped_hyphenation_is_rejoined_without_removing_compound_hyphens() -> None:
    """Whitespace introduced around a line-wrap hyphen should be removed."""
    assert (
        normalize_hyphenation("post- exploitation and man-in- the-middle")
        == "post-exploitation and man-in-the-middle"
    )
    assert normalize_hyphenation("infor- mation") == "information"
    assert normalize_hyphenation("state-of-the-art") == "state-of-the-art"


def test_hyphenated_line_breaks_are_rejoined_within_page() -> None:
    """Words broken at a line-ending hyphen should be rejoined."""
    cleaned = normalize_segments([TextSegment("recon-", 1), TextSegment("naissance", 1)])
    assert [segment.text for segment in cleaned] == ["reconnaissance"]


def test_bullets_and_glued_question_heading_are_split() -> None:
    """Common bullet glyphs and a trailing question should become separate units."""
    result = normalize_segments(
        [TextSegment("A useful paragraph. What are the benefits?\n▪ First benefit", 1)]
    )
    assert [segment.text for segment in result] == [
        "A useful paragraph.",
        "What are the benefits?",
        "• First benefit",
    ]
    assert result[1].heading == "What are the benefits?"


def test_repeated_page_furniture_is_removed_but_unique_content_remains() -> None:
    """Repeated lines across pages and bare page numbers are treated as furniture."""
    result = normalize_segments(
        [
            TextSegment("Organization Handbook", 1),
            TextSegment("1", 1),
            TextSegment("First page text", 1),
            TextSegment("Organization Handbook", 2),
            TextSegment("2", 2),
            TextSegment("Second page text", 2),
        ]
    )
    assert [segment.text for segment in result] == ["First page text", "Second page text"]


def test_repeated_page_furniture_removal_is_reported() -> None:
    """Removed lines should be available to the document quality report."""
    cleaned, removed = normalize_segments_with_removed(
        [TextSegment("Organization Handbook", 1), TextSegment("Organization Handbook", 2)]
    )
    assert not cleaned
    assert removed == ["Organization Handbook"]


def test_configured_label_value_pairs_are_split_generically() -> None:
    """Configured metadata labels should split a clearly repeated label/value line."""
    cleaned = normalize_segments([TextSegment("Level Basic Duration 2 Months Format Offline", 1)])
    assert [segment.text for segment in cleaned] == [
        "Level: Basic",
        "Duration: 2 Months",
        "Format: Offline",
    ]
