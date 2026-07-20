"""Shared application service used by CLI, Web UI, and HTTP API.

All user-facing entry points should call ``RedactortronService`` so errors,
validation, and pipeline stages stay consistent.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from redactortron.core import DEFAULT_LABELS, RedactortronCore
from redactortron.exceptions import (
    CategorySelectionError,
    ConfigurationError,
    ExportError,
    InputError,
    RedactortronError,
    wrap_unexpected,
)
from redactortron.models import DetectedEntity, ScanResult

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class EntityView:
    """Lightweight entity DTO for API / UI tables."""

    page: int
    category: str
    text: str
    score: float

    @classmethod
    def from_entity(cls, entity: DetectedEntity) -> "EntityView":
        return cls(
            page=entity.page_index + 1,
            category=entity.category,
            text=entity.text,
            score=round(float(entity.score), 4),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScanSummary:
    """Serializable scan summary shared by CLI / UI / API."""

    source_path: str
    page_count: int
    entity_count: int
    categories: List[str]
    entities: List[EntityView]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "page_count": self.page_count,
            "entity_count": self.entity_count,
            "categories": list(self.categories),
            "entities": [e.to_dict() for e in self.entities],
        }


class RedactortronService:
    """Facade over ``RedactortronCore`` with validation and stage-aware errors.

    Example::

        service = RedactortronService()
        summary, result = service.scan("doc.pdf")
        path = service.redact("doc.pdf", categories=["PERSON"], scan_result=result)
    """

    def __init__(
        self,
        core: Optional[RedactortronCore] = None,
        *,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.core = core or RedactortronCore(labels=labels)
        if labels:
            self.core.labels = list(labels)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_threshold(threshold: float) -> float:
        try:
            value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Threshold must be a number between 0 and 1 (got {threshold!r}).",
                context={"threshold": threshold},
                cause=exc,
            ) from exc
        if not 0.0 < value <= 1.0:
            raise ConfigurationError(
                f"Threshold out of range: {value}. Expected (0, 1].",
                context={"threshold": value},
                hint="Use something like 0.4 (default) or 0.25 for more matches.",
            )
        return value

    @staticmethod
    def validate_input_path(source: PathLike) -> Path:
        path = Path(source)
        if not path.exists():
            raise InputError(
                f"Input file not found: {path}",
                code="INPUT_NOT_FOUND",
                context={"path": str(path.resolve()) if path.parent.exists() else str(path)},
                hint="Check the path spelling, or pass an absolute path.",
            )
        if not path.is_file():
            raise InputError(
                f"Input path is not a file: {path}",
                code="INPUT_NOT_A_FILE",
                context={"path": str(path)},
            )
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise InputError(
                f"Unsupported file type: {suffix or '(none)'}",
                code="INPUT_UNSUPPORTED_TYPE",
                context={
                    "path": str(path),
                    "suffix": suffix,
                    "supported": sorted(SUPPORTED_SUFFIXES),
                },
            )
        return path.resolve()

    @staticmethod
    def validate_categories(
        categories: Sequence[str],
        *,
        available: Optional[Sequence[str]] = None,
        require_non_empty: bool = True,
    ) -> List[str]:
        selected = [c.strip().upper() for c in categories if str(c).strip()]
        if require_non_empty and not selected:
            raise CategorySelectionError(
                "No categories selected for redaction.",
                hint="Choose at least one detected category before blurring.",
            )
        if available is not None:
            available_set = {a.strip().upper() for a in available}
            missing = [c for c in selected if c not in available_set]
            if missing and not any(c in available_set for c in selected):
                raise CategorySelectionError(
                    "None of the requested categories were found in the scan.",
                    context={
                        "requested": selected,
                        "available": sorted(available_set),
                        "missing": missing,
                    },
                    hint="Run a scan first and pick from the detected category list.",
                )
            if missing:
                logger.warning(
                    "Ignoring unknown categories %s (available: %s)",
                    missing,
                    sorted(available_set),
                )
            selected = [c for c in selected if c in available_set]
            if require_non_empty and not selected:
                raise CategorySelectionError(
                    "After filtering unknown names, no valid categories remain.",
                    context={"missing": missing, "available": sorted(available_set)},
                )
        return selected

    @staticmethod
    def default_output_path(input_path: Path) -> Path:
        return input_path.with_name(
            f"{input_path.stem}_redacted{input_path.suffix or '.pdf'}"
        )

    @staticmethod
    def summarize(result: ScanResult) -> ScanSummary:
        entities = [EntityView.from_entity(e) for e in result.all_entities]
        return ScanSummary(
            source_path=result.source_path,
            page_count=len(result.pages),
            entity_count=len(entities),
            categories=result.categories(),
            entities=entities,
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def scan(
        self,
        source: PathLike,
        *,
        threshold: float = 0.4,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[ScanSummary, ScanResult]:
        """Validate input, run OCR + NER, return summary + full result."""
        path = self.validate_input_path(source)
        thr = self.validate_threshold(threshold)
        if labels is not None:
            cleaned = [str(x).strip() for x in labels if str(x).strip()]
            if not cleaned:
                raise ConfigurationError(
                    "Label list is empty after cleaning.",
                    context={"labels": list(labels)},
                    hint=f"Omit --labels to use defaults: {', '.join(DEFAULT_LABELS)}",
                )
            self.core.labels = cleaned

        logger.info("scan start path=%s threshold=%s", path, thr)
        try:
            result = self.core.scan(path, threshold=thr, labels=labels)
        except RedactortronError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize for all frontends
            raise wrap_unexpected(
                exc,
                stage="scan",
                message=f"Scan failed for {path.name}.",
                context={"path": str(path)},
            ) from exc

        summary = self.summarize(result)
        logger.info(
            "scan done pages=%s entities=%s categories=%s",
            summary.page_count,
            summary.entity_count,
            summary.categories,
        )
        return summary, result

    def redact(
        self,
        source: PathLike,
        *,
        categories: Optional[Sequence[str]] = None,
        entity_ids: Optional[Sequence[str]] = None,
        output: Optional[PathLike] = None,
        scan_result: Optional[ScanResult] = None,
        threshold: float = 0.4,
    ) -> Path:
        """Blur selected entity ids and/or categories; write the redacted document."""
        path = self.validate_input_path(source)
        thr = self.validate_threshold(threshold)

        result = scan_result
        if result is None:
            _, result = self.scan(path, threshold=thr)
        if not any(e.entity_id for e in result.all_entities):
            result.assign_entity_ids()

        selected_entities: List[DetectedEntity]
        if entity_ids is not None:
            ids = [ScanResult.entity_id_from_choice(i) for i in entity_ids]
            ids = [i for i in ids if i]
            if not ids:
                raise CategorySelectionError(
                    "No transactions / entities selected for redaction.",
                    hint="Check individual items, or select a category to include all of that type.",
                )
            selected_entities = result.entities_for_ids(ids)
            if not selected_entities:
                raise CategorySelectionError(
                    "None of the selected item ids matched the scan results.",
                    context={"requested": ids},
                )
            selected_cats = sorted({e.category for e in selected_entities})
        else:
            selected_cats = self.validate_categories(
                list(categories or []),
                available=result.categories(),
                require_non_empty=True,
            )
            selected_entities = result.entities_for_categories(selected_cats)

        dest = Path(output) if output else self.default_output_path(path)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExportError(
                f"Cannot create output directory: {dest.parent}",
                context={"output": str(dest)},
                cause=exc,
                hint="Choose a writable --output path.",
            ) from exc

        logger.info(
            "redact start path=%s items=%s categories=%s output=%s",
            path,
            [e.entity_id for e in selected_entities],
            selected_cats,
            dest,
        )
        try:
            written = self.core.redact(
                source=path,
                categories=selected_cats,
                output=dest,
                scan_result=result,
                threshold=thr,
                entities=selected_entities,
            )
        except RedactortronError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise wrap_unexpected(
                exc,
                stage="redact",
                message=f"Redaction failed for {path.name}.",
                context={
                    "path": str(path),
                    "output": str(dest),
                    "categories": selected_cats,
                    "entity_ids": [e.entity_id for e in selected_entities],
                },
            ) from exc

        logger.info("redact done output=%s", written)
        return Path(written)

    def health(self) -> Dict[str, Any]:
        """Lightweight status payload for API health checks."""
        from redactortron import __version__

        return {
            "status": "ok",
            "version": __version__,
            "ocr_loaded": self.core._ocr is not None,
            "gliner_loaded": self.core._gliner is not None,
            "default_labels": list(self.core.labels),
        }
