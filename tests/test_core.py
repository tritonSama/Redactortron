"""Tests for RedactortronCore helpers that do not need heavy ML models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from redactortron.core import RedactortronCore
from redactortron.exceptions import ExportError, InputError
from redactortron.models import BoundingBox, DetectedEntity, WordSpan


def test_load_pages_missing_file(tmp_path: Path) -> None:
    core = RedactortronCore()
    with pytest.raises(InputError) as exc_info:
        core.load_pages(tmp_path / "nope.pdf")
    assert exc_info.value.code == "INPUT_NOT_FOUND"


def test_load_pages_unsupported_type(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_bytes(b"x")
    core = RedactortronCore()
    with pytest.raises(InputError) as exc_info:
        core.load_pages(path)
    assert exc_info.value.code == "INPUT_UNSUPPORTED_TYPE"


def test_load_pages_image(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "page.png"
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:] = (20, 40, 60)
    assert cv2.imwrite(str(path), img)

    core = RedactortronCore()
    frames = core.load_pages(path)
    assert len(frames) == 1
    assert frames[0].shape[0] == 40
    assert frames[0].shape[1] == 60


def test_map_entity_to_box_by_offsets() -> None:
    core = RedactortronCore()
    words = [
        WordSpan("Jane", BoundingBox(0, 0, 10, 10), 0, 0, 4),
        WordSpan("Doe", BoundingBox(12, 0, 22, 10), 0, 5, 8),
    ]
    box = core._map_entity_to_box("Jane Doe", 0, 8, words, 100, 100)
    assert box is not None
    assert box.as_tuple() == (0, 0, 22, 10)


def test_blur_entities_obscures_region() -> None:
    core = RedactortronCore(blur_kernel=15)
    # Checkerboard so a blur visibly changes local pixels.
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[0:50:2, 0:50:2] = 255
    image[1:50:2, 1:50:2] = 255
    entity = DetectedEntity(
        text="secret",
        label="ssn",
        score=0.99,
        page_index=0,
        box=BoundingBox(10, 10, 30, 30),
    )
    out = core.blur_entities(image, [entity])
    assert not np.array_equal(out[15:25, 15:25], image[15:25, 15:25])
    # Far corner stays unchanged.
    assert np.array_equal(out[0:3, 0:3], image[0:3, 0:3])


def test_redact_image_roundtrip(tmp_path: Path) -> None:
    import cv2

    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    img = np.full((80, 80, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(src), img)

    entity = DetectedEntity(
        text="X",
        label="person",
        score=1.0,
        page_index=0,
        box=BoundingBox(5, 5, 40, 40),
    )
    from redactortron.models import PageResult, ScanResult

    scan = ScanResult(
        source_path=str(src),
        pages=[
            PageResult(
                page_index=0,
                width=80,
                height=80,
                full_text="X",
                entities=[entity],
            )
        ],
    )

    core = RedactortronCore(blur_kernel=21)
    written = core.redact(
        source=src,
        categories=["PERSON"],
        output=out,
        scan_result=scan,
    )
    assert written.exists()
    loaded = cv2.imread(str(written))
    assert loaded is not None
    assert loaded.shape == img.shape


def test_export_multipage_requires_pdf(tmp_path: Path) -> None:
    import cv2

    src = tmp_path / "a.png"
    cv2.imwrite(str(src), np.zeros((10, 10, 3), dtype=np.uint8))
    core = RedactortronCore()

    # Force multi-frame path by stubbing load_pages.
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]
    with patch.object(core, "load_pages", return_value=frames):
        with patch.object(core, "scan", return_value=MagicMock(entities_for_categories=lambda _c: [])):
            with pytest.raises(ExportError) as exc_info:
                core.redact(
                    source=src,
                    categories=["PERSON"],
                    output=tmp_path / "out.png",
                    scan_result=MagicMock(entities_for_categories=lambda _c: []),
                )
    assert "pdf" in exc_info.value.hint.lower() or "pdf" in exc_info.value.message.lower()
