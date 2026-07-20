"""RedactortronCore — OCR, GLiNER entity detection, and OpenCV blur."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from redactortron.exceptions import (
    DependencyError,
    EntityDetectionError,
    ExportError,
    InputError,
    ModelLoadError,
    OCRError,
    RedactionError,
)
from redactortron.models import (
    BoundingBox,
    DetectedEntity,
    PageResult,
    ScanResult,
    WordSpan,
)
from redactortron.paths import resolve_poppler_path

logger = logging.getLogger(__name__)

# Default GLiNER labels: PII + financial / transaction types.
DEFAULT_LABELS: List[str] = [
    "person",
    "organization",
    "location",
    "email",
    "phone number",
    "date",
    "address",
    "credit card",
    "ssn",
    "account number",
    # Transaction / banking item types
    "transaction amount",
    "transaction date",
    "merchant",
    "vendor",
    "invoice number",
    "check number",
    "routing number",
    "bank name",
    "payment method",
    "transaction id",
    "balance",
    "currency amount",
    "deposit",
    "withdrawal",
]

PathLike = Union[str, Path]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class RedactortronCore:
    """Local AI-powered document redaction engine.

    Pipeline:
        1. Render PDF pages (or load images).
        2. OCR with docTR → text + word bounding boxes.
        3. Named-entity recognition with GLiNER.
        4. Blur selected entity regions with OpenCV.
    """

    def __init__(
        self,
        gliner_model: str = "urchade/gliner_multi_pii-v1",
        labels: Optional[Sequence[str]] = None,
        blur_kernel: int = 51,
        pdf_dpi: int = 200,
        device: str = "cpu",
    ) -> None:
        self.gliner_model_name = gliner_model
        self.labels = list(labels) if labels else list(DEFAULT_LABELS)
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.pdf_dpi = pdf_dpi
        self.device = device

        self._ocr = None
        self._gliner = None
        self._DocumentFile = None

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _ensure_ocr(self) -> None:
        if self._ocr is not None:
            return
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
        except ImportError as exc:
            raise DependencyError(
                "python-doctr is not installed or failed to import.",
                context={"dependency": "python-doctr"},
                cause=exc,
                hint="Install with: pip install 'python-doctr[torch]'",
            ) from exc

        try:
            self._DocumentFile = DocumentFile
            self._ocr = ocr_predictor(pretrained=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "Failed to load the docTR OCR predictor.",
                context={"component": "doctr.ocr_predictor"},
                cause=exc,
            ) from exc
        logger.info("docTR OCR predictor loaded")

    def _ensure_gliner(self) -> None:
        if self._gliner is not None:
            return
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise DependencyError(
                "gliner is not installed or failed to import.",
                context={"dependency": "gliner"},
                cause=exc,
                hint="Install with: pip install gliner",
            ) from exc

        try:
            self._gliner = GLiNER.from_pretrained(self.gliner_model_name)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load GLiNER model {self.gliner_model_name!r}.",
                context={"model": self.gliner_model_name},
                cause=exc,
            ) from exc
        logger.info("GLiNER model loaded: %s", self.gliner_model_name)

    # ------------------------------------------------------------------
    # Page loading
    # ------------------------------------------------------------------

    def load_pages(self, source: PathLike) -> List[np.ndarray]:
        """Load a PDF or image file as BGR OpenCV frames."""
        path = Path(source)
        if not path.exists():
            raise InputError(
                f"Input not found: {path}",
                code="INPUT_NOT_FOUND",
                context={"path": str(path)},
            )

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                kwargs = {"dpi": self.pdf_dpi}
                poppler = resolve_poppler_path()
                if poppler:
                    kwargs["poppler_path"] = poppler
                pil_pages = convert_from_path(str(path), **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise DependencyError(
                    "Failed to render PDF pages.",
                    stage="load",
                    code="PDF_RENDER_ERROR",
                    context={"path": str(path), "dpi": self.pdf_dpi},
                    cause=exc,
                    hint=(
                        "Install Poppler and ensure `pdftoppm` is on PATH, "
                        "or run: python scripts/install_deps.py --with-poppler"
                    ),
                ) from exc
            if not pil_pages:
                raise InputError(
                    f"PDF contained no renderable pages: {path}",
                    code="PDF_EMPTY",
                    context={"path": str(path)},
                )
            return [
                cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR)
                for p in pil_pages
            ]

        if suffix in _IMAGE_SUFFIXES:
            image = cv2.imread(str(path))
            if image is None:
                raise InputError(
                    f"Unable to read image (corrupt or unsupported encoding): {path}",
                    code="IMAGE_UNREADABLE",
                    context={"path": str(path)},
                )
            return [image]

        raise InputError(
            f"Unsupported file type: {suffix or '(none)'}",
            code="INPUT_UNSUPPORTED_TYPE",
            context={"path": str(path), "suffix": suffix},
        )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_page(self, image_bgr: np.ndarray, page_index: int) -> PageResult:
        """Run docTR on a single page and collect word boxes."""
        self._ensure_ocr()
        assert self._ocr is not None and self._DocumentFile is not None

        height, width = image_bgr.shape[:2]

        try:
            # docTR's DocumentFile.from_images accepts paths or encoded bytes —
            # not raw numpy arrays (raises TypeError otherwise).
            ok, encoded = cv2.imencode(".png", image_bgr)
            if not ok:
                raise OCRError(
                    f"Failed to encode page {page_index + 1} for OCR.",
                    context={"page_index": page_index, "width": width, "height": height},
                )
            doc = self._DocumentFile.from_images([encoded.tobytes()])
            result = self._ocr(doc)
        except OCRError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRError(
                f"OCR failed on page {page_index + 1}.",
                context={"page_index": page_index, "width": width, "height": height},
                cause=exc,
            ) from exc

        words: List[WordSpan] = []
        text_parts: List[str] = []
        cursor = 0

        page = result.pages[0]
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x_min, y_min), (x_max, y_max) = word.geometry
                    box = BoundingBox(
                        x_min=int(x_min * width),
                        y_min=int(y_min * height),
                        x_max=int(x_max * width),
                        y_max=int(y_max * height),
                    ).clamp(width, height)

                    token = word.value
                    start = cursor
                    end = cursor + len(token)
                    words.append(
                        WordSpan(
                            text=token,
                            box=box,
                            page_index=page_index,
                            start_char=start,
                            end_char=end,
                        )
                    )
                    text_parts.append(token)
                    cursor = end + 1

        full_text = " ".join(text_parts)
        return PageResult(
            page_index=page_index,
            width=width,
            height=height,
            full_text=full_text,
            words=words,
        )

    # ------------------------------------------------------------------
    # Entity detection (GLiNER)
    # ------------------------------------------------------------------

    def _map_entity_to_box(
        self,
        entity_text: str,
        start: Optional[int],
        end: Optional[int],
        words: Sequence[WordSpan],
        page_width: int,
        page_height: int,
    ) -> Optional[BoundingBox]:
        """Map a GLiNER span onto OCR word boxes (union of overlapping words)."""
        matched: List[WordSpan] = []

        if start is not None and end is not None:
            for word in words:
                if word.start_char < end and word.end_char > start:
                    matched.append(word)
        else:
            needle = entity_text.strip().lower()
            if not needle:
                return None
            for word in words:
                if needle in word.text.lower() or word.text.lower() in needle:
                    matched.append(word)

        if not matched:
            return None

        return BoundingBox(
            x_min=min(w.box.x_min for w in matched),
            y_min=min(w.box.y_min for w in matched),
            x_max=max(w.box.x_max for w in matched),
            y_max=max(w.box.y_max for w in matched),
        ).clamp(page_width, page_height)

    def _detect_entities(self, page: PageResult, threshold: float = 0.4) -> List[DetectedEntity]:
        self._ensure_gliner()
        assert self._gliner is not None

        if not page.full_text.strip():
            return []

        try:
            raw = self._gliner.predict_entities(
                page.full_text,
                self.labels,
                threshold=threshold,
            )
        except Exception as exc:  # noqa: BLE001
            raise EntityDetectionError(
                f"GLiNER failed on page {page.page_index + 1}.",
                context={
                    "page_index": page.page_index,
                    "text_length": len(page.full_text),
                    "labels": list(self.labels),
                    "threshold": threshold,
                },
                cause=exc,
            ) from exc

        entities: List[DetectedEntity] = []
        for item in raw:
            text = item.get("text", "")
            label = item.get("label", "UNKNOWN")
            score = float(item.get("score", 0.0))
            start = item.get("start")
            end = item.get("end")

            box = self._map_entity_to_box(
                text, start, end, page.words, page.width, page.height
            )
            if box is None:
                logger.debug("Skipping entity without geometry: %r (%s)", text, label)
                continue

            entities.append(
                DetectedEntity(
                    text=text,
                    label=label,
                    score=score,
                    page_index=page.page_index,
                    box=box,
                    start_char=start,
                    end_char=end,
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def scan(
        self,
        source: PathLike,
        threshold: float = 0.4,
        labels: Optional[Sequence[str]] = None,
    ) -> ScanResult:
        """OCR + entity detection for every page in *source*."""
        if labels:
            self.labels = list(labels)

        frames = self.load_pages(source)
        pages: List[PageResult] = []

        for idx, frame in enumerate(frames):
            logger.info("OCR page %d/%d", idx + 1, len(frames))
            page = self._ocr_page(frame, idx)
            logger.info("Detecting entities on page %d", idx + 1)
            page.entities = self._detect_entities(page, threshold=threshold)
            pages.append(page)

        result = ScanResult(source_path=str(Path(source).resolve()), pages=pages)
        result.assign_entity_ids()
        return result

    def blur_entities(
        self,
        image_bgr: np.ndarray,
        entities: Iterable[DetectedEntity],
        padding: int = 4,
    ) -> np.ndarray:
        """Apply a strong Gaussian blur over each entity bounding box."""
        try:
            output = image_bgr.copy()
            height, width = output.shape[:2]

            for entity in entities:
                box = entity.box.clamp(width, height)
                x0 = max(0, box.x_min - padding)
                y0 = max(0, box.y_min - padding)
                x1 = min(width, box.x_max + padding)
                y1 = min(height, box.y_max + padding)
                if x1 <= x0 or y1 <= y0:
                    continue
                roi = output[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                output[y0:y1, x0:x1] = cv2.GaussianBlur(
                    roi, (self.blur_kernel, self.blur_kernel), 0
                )
            return output
        except Exception as exc:  # noqa: BLE001
            raise RedactionError(
                "OpenCV blur failed while redacting an entity region.",
                context={"blur_kernel": self.blur_kernel},
                cause=exc,
            ) from exc

    def redact(
        self,
        source: PathLike,
        categories: Optional[Sequence[str]] = None,
        output: Optional[PathLike] = None,
        scan_result: Optional[ScanResult] = None,
        threshold: float = 0.4,
        entity_ids: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[DetectedEntity]] = None,
    ) -> Path:
        """Scan (if needed), blur selected items / categories, and write the result."""
        if output is None:
            raise ExportError(
                "Output path is required.",
                hint="Pass an --output / destination path for the redacted file.",
            )
        path = Path(source)
        out = Path(output)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExportError(
                f"Cannot create output directory: {out.parent}",
                context={"output": str(out)},
                cause=exc,
            ) from exc

        result = scan_result or self.scan(path, threshold=threshold)
        if not any(e.entity_id for e in result.all_entities):
            result.assign_entity_ids()

        if entities is not None:
            selected = list(entities)
        elif entity_ids:
            selected = result.entities_for_ids(list(entity_ids))
        else:
            selected = result.entities_for_categories(list(categories or []))

        frames = self.load_pages(path)
        redacted_frames: List[np.ndarray] = []

        for idx, frame in enumerate(frames):
            page_entities = [e for e in selected if e.page_index == idx]
            redacted_frames.append(self.blur_entities(frame, page_entities))

        suffix = out.suffix.lower()
        try:
            if suffix == ".pdf" or path.suffix.lower() == ".pdf":
                if suffix != ".pdf":
                    out = out.with_suffix(".pdf")
                pil_images = [
                    Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    for f in redacted_frames
                ]
                if not pil_images:
                    raise ExportError(
                        "No pages available to write.",
                        context={"source": str(path), "output": str(out)},
                    )
                first, rest = pil_images[0], pil_images[1:]
                first.save(
                    str(out),
                    "PDF",
                    resolution=self.pdf_dpi,
                    save_all=True,
                    append_images=rest,
                )
            else:
                if len(redacted_frames) != 1:
                    raise ExportError(
                        "Multi-page outputs must use a .pdf destination.",
                        context={
                            "page_count": len(redacted_frames),
                            "output": str(out),
                        },
                        hint="Pass an output path ending in .pdf.",
                    )
                ok = cv2.imwrite(str(out), redacted_frames[0])
                if not ok:
                    raise ExportError(
                        f"OpenCV failed to write image: {out}",
                        context={"output": str(out)},
                    )
        except ExportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExportError(
                f"Failed writing redacted document to {out}.",
                context={"output": str(out), "source": str(path)},
                cause=exc,
            ) from exc

        logger.info("Wrote redacted document to %s", out)
        return out
