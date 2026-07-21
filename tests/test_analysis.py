"""Tests for word search, date discovery, line capture, and date trees."""

from __future__ import annotations

from redactortron.analysis import (
    build_date_tree,
    entity_line_indices,
    find_all_dates,
    find_word_matches,
    line_entities_for,
)
from redactortron.models import (
    BoundingBox,
    DetectedEntity,
    PageResult,
    ScanResult,
    WordSpan,
)


def _statement_result() -> ScanResult:
    """One page, two lines: a transaction row and a name row."""
    words = [
        WordSpan("12/03/2022", BoundingBox(0, 0, 50, 10), 0, 0, 10, line_index=0),
        WordSpan("WALMART", BoundingBox(60, 0, 110, 10), 0, 11, 18, line_index=0),
        WordSpan("-45.00", BoundingBox(120, 0, 160, 10), 0, 19, 25, line_index=0),
        WordSpan("John", BoundingBox(0, 20, 30, 30), 0, 26, 30, line_index=1),
        WordSpan("Doe", BoundingBox(35, 20, 60, 30), 0, 31, 34, line_index=1),
    ]
    entities = [
        DetectedEntity(
            "12/03/2022", "date", 0.95, 0, BoundingBox(0, 0, 50, 10), 0, 10
        ),
        DetectedEntity(
            "WALMART", "merchant", 0.9, 0, BoundingBox(60, 0, 110, 10), 11, 18
        ),
        DetectedEntity(
            "John Doe", "person", 0.85, 0, BoundingBox(0, 20, 60, 30), 26, 34
        ),
    ]
    page = PageResult(
        page_index=0,
        width=200,
        height=100,
        full_text="12/03/2022 WALMART -45.00 John Doe",
        words=words,
        entities=entities,
    )
    result = ScanResult(source_path="doc.pdf", pages=[page])
    result.assign_entity_ids()
    return result


def test_find_word_matches_single_word() -> None:
    result = _statement_result()
    matches = find_word_matches(result, "walmart")
    assert len(matches) == 1
    assert matches[0].category == "WORD MATCH"
    assert matches[0].box.as_tuple() == (60, 0, 110, 10)


def test_find_word_matches_phrase_same_line() -> None:
    result = _statement_result()
    matches = find_word_matches(result, "john doe")
    assert len(matches) == 1
    assert matches[0].box.as_tuple() == (0, 20, 60, 30)
    assert find_word_matches(result, "") == []
    assert find_word_matches(result, "nonexistent") == []


def test_find_all_dates() -> None:
    result = _statement_result()
    found = find_all_dates(result)
    assert len(found) == 1
    assert found[0].text == "12/03/2022"
    assert found[0].category == "DATE FOUND"


def test_entity_line_indices_char_overlap() -> None:
    result = _statement_result()
    page = result.pages[0]
    date_entity = page.entities[0]
    assert entity_line_indices(page, date_entity) == [0]
    person = page.entities[2]
    assert entity_line_indices(page, person) == [1]


def test_line_entities_cover_full_row() -> None:
    result = _statement_result()
    date_entity = result.pages[0].entities[0]
    lines = line_entities_for(result, [date_entity])
    assert len(lines) == 1
    assert lines[0].category == "LINE"
    assert lines[0].box.as_tuple() == (0, 0, 160, 10)
    assert "WALMART" in lines[0].text

    # Duplicate lines are collapsed.
    merchant = result.pages[0].entities[1]
    both = line_entities_for(result, [date_entity, merchant])
    assert len(both) == 1


def test_build_date_tree_links_entities_on_line() -> None:
    result = _statement_result()
    dates = [e for e in result.all_entities if e.category == "DATE"]
    tree = build_date_tree(result, dates)
    assert "### Page 1" in tree
    assert "12/03/2022" in tree
    assert "WALMART" in tree
    # The name line is not connected to the date line.
    assert "John Doe" not in tree
