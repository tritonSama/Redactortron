"""Tests for data models."""

from __future__ import annotations

from redactortron.models import BoundingBox, DetectedEntity, PageResult, ScanResult


def test_bounding_box_clamp() -> None:
    box = BoundingBox(x_min=-10, y_min=-5, x_max=500, y_max=500)
    clamped = box.clamp(100, 80)
    assert clamped.as_tuple() == (0, 0, 100, 80)


def test_detected_entity_category_normalized() -> None:
    entity = DetectedEntity(
        text="Ada Lovelace",
        label=" person ",
        score=0.9,
        page_index=0,
        box=BoundingBox(1, 2, 3, 4),
    )
    assert entity.category == "PERSON"


def test_scan_result_categories_and_filter() -> None:
    box = BoundingBox(0, 0, 10, 10)
    page = PageResult(
        page_index=0,
        width=100,
        height=100,
        full_text="Ada works at Acme",
        entities=[
            DetectedEntity("Ada", "person", 0.9, 0, box),
            DetectedEntity("Acme", "organization", 0.8, 0, box),
            DetectedEntity("Ada", "person", 0.7, 0, box),
        ],
    )
    result = ScanResult(source_path="doc.pdf", pages=[page])
    result.assign_entity_ids()
    assert result.categories() == ["ORGANIZATION", "PERSON"]
    people = result.entities_for_categories(["person"])
    assert len(people) == 2
    assert all(e.category == "PERSON" for e in people)
    assert [e.entity_id for e in result.all_entities] == ["E0001", "E0002", "E0003"]
    by_id = result.entities_for_ids(["E0002"])
    assert len(by_id) == 1
    assert by_id[0].text == "Acme"
    label = result.all_entities[0].display_label()
    assert label.startswith("E0001")
    assert "PERSON" in label
    assert "[PI]" in label
    assert ScanResult.entity_id_from_choice(label) == "E0001"
    assert result.all_entities[0].family == "personal"


def test_add_entities_continues_ids_and_dedupes() -> None:
    box = BoundingBox(0, 0, 10, 10)
    page = PageResult(
        page_index=0,
        width=100,
        height=100,
        full_text="Ada",
        entities=[DetectedEntity("Ada", "person", 0.9, 0, box)],
    )
    result = ScanResult(source_path="doc.pdf", pages=[page])
    result.assign_entity_ids()

    extra_box = BoundingBox(20, 20, 40, 30)
    extra = DetectedEntity("Walmart", "word match", 1.0, 0, extra_box)
    added = result.add_entities([extra])
    assert [e.entity_id for e in added] == ["E0002"]

    # Same box + category again is a duplicate and must be skipped.
    dup = DetectedEntity("Walmart", "word match", 1.0, 0, extra_box)
    assert result.add_entities([dup]) == []
    assert len(result.all_entities) == 2


def test_line_box_and_text() -> None:
    from redactortron.models import WordSpan

    page = PageResult(
        page_index=0,
        width=200,
        height=100,
        full_text="12/03/2022 WALMART -45.00",
        words=[
            WordSpan("12/03/2022", BoundingBox(0, 0, 50, 10), 0, 0, 10, line_index=0),
            WordSpan("WALMART", BoundingBox(60, 0, 110, 10), 0, 11, 18, line_index=0),
            WordSpan("-45.00", BoundingBox(120, 0, 160, 10), 0, 19, 25, line_index=0),
        ],
    )
    assert page.line_text(0) == "12/03/2022 WALMART -45.00"
    box = page.line_box(0)
    assert box is not None
    assert box.as_tuple() == (0, 0, 160, 10)
    assert page.line_box(5) is None
