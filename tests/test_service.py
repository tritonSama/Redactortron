"""Tests for RedactortronService validation and pipeline wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from redactortron.exceptions import (
    CategorySelectionError,
    ConfigurationError,
    InputError,
)
from redactortron.models import (
    BoundingBox,
    DetectedEntity,
    PageResult,
    ScanResult,
)
from redactortron.service import RedactortronService


def _sample_scan(path: str = "doc.pdf") -> ScanResult:
    box = BoundingBox(0, 0, 20, 20)
    page = PageResult(
        page_index=0,
        width=200,
        height=200,
        full_text="Jane Doe email jane@example.com",
        entities=[
            DetectedEntity("Jane Doe", "person", 0.91, 0, box),
            DetectedEntity("jane@example.com", "email", 0.88, 0, box),
        ],
    )
    return ScanResult(source_path=path, pages=[page])


def test_validate_threshold_rejects_out_of_range() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        RedactortronService.validate_threshold(1.5)
    assert exc_info.value.stage == "config"
    assert "out of range" in exc_info.value.message.lower()


def test_validate_threshold_rejects_non_numeric() -> None:
    with pytest.raises(ConfigurationError):
        RedactortronService.validate_threshold("nope")  # type: ignore[arg-type]


def test_validate_input_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "gone.pdf"
    with pytest.raises(InputError) as exc_info:
        RedactortronService.validate_input_path(missing)
    assert exc_info.value.code == "INPUT_NOT_FOUND"
    assert exc_info.value.stage == "load"


def test_validate_input_path_unsupported(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(InputError) as exc_info:
        RedactortronService.validate_input_path(bad)
    assert exc_info.value.code == "INPUT_UNSUPPORTED_TYPE"
    assert ".txt" in exc_info.value.format()


def test_validate_categories_requires_selection() -> None:
    with pytest.raises(CategorySelectionError):
        RedactortronService.validate_categories([])


def test_validate_categories_filters_unknown() -> None:
    selected = RedactortronService.validate_categories(
        ["person", "UNKNOWN_TAG"],
        available=["PERSON", "EMAIL"],
    )
    assert selected == ["PERSON"]


def test_validate_categories_all_unknown_raises() -> None:
    with pytest.raises(CategorySelectionError) as exc_info:
        RedactortronService.validate_categories(
            ["SSN"],
            available=["PERSON", "EMAIL"],
        )
    assert "available" in exc_info.value.context


def test_summarize_scan_result() -> None:
    summary = RedactortronService.summarize(_sample_scan("/tmp/doc.pdf"))
    assert summary.page_count == 1
    assert summary.entity_count == 2
    assert summary.categories == ["EMAIL", "PERSON"]
    payload = summary.to_dict()
    assert payload["entities"][0]["page"] == 1


def test_scan_delegates_to_core(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    # Minimal valid PNG via OpenCV would pull cv2; write bytes and mock load/scan.
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    core = MagicMock()
    scan = _sample_scan(str(image))
    core.scan.return_value = scan
    core.labels = ["person", "email"]
    core._ocr = None
    core._gliner = None

    service = RedactortronService(core=core)
    # Bypass real path-type image decode by patching validate only — file exists
    # but suffix is supported; core.scan is mocked so load_pages is not called.
    summary, result = service.scan(image, threshold=0.4)

    core.scan.assert_called_once()
    assert summary.entity_count == 2
    assert result is scan


def test_redact_requires_categories(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    core = MagicMock()
    service = RedactortronService(core=core)
    with pytest.raises(CategorySelectionError):
        service.redact(image, categories=[], scan_result=_sample_scan(str(image)))


def test_redact_writes_via_core(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out = tmp_path / "out.png"

    core = MagicMock()
    core.redact.return_value = out
    service = RedactortronService(core=core)

    written = service.redact(
        image,
        categories=["PERSON"],
        output=out,
        scan_result=_sample_scan(str(image)),
    )
    assert written == out
    core.redact.assert_called_once()
    kwargs = core.redact.call_args.kwargs
    assert kwargs["categories"] == ["PERSON"]


def test_redact_by_entity_ids(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out = tmp_path / "out.png"

    scan = _sample_scan(str(image))
    scan.assign_entity_ids()

    core = MagicMock()
    core.redact.return_value = out
    service = RedactortronService(core=core)

    written = service.redact(
        image,
        entity_ids=["E0001"],
        output=out,
        scan_result=scan,
    )
    assert written == out
    kwargs = core.redact.call_args.kwargs
    assert kwargs["entities"][0].entity_id == "E0001"


def test_health_payload() -> None:
    core = MagicMock()
    core._ocr = None
    core._gliner = object()
    core.labels = ["person"]
    service = RedactortronService(core=core)
    health = service.health()
    assert health["status"] == "ok"
    assert health["gliner_loaded"] is True
    assert health["ocr_loaded"] is False
