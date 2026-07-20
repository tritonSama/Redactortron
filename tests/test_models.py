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
    assert ScanResult.entity_id_from_choice(label) == "E0001"
