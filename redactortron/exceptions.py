"""Stage-aware exceptions for clear debugging across CLI, Web UI, and API."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from redactortron.security_log import (
    emit_structured_log,
    format_serialized_error,
    new_correlation_id,
    serialize_error,
)


class RedactortronError(Exception):
    """Base error with pipeline stage, hint, and structured context.

    Attributes:
        message: Human-readable description of what failed.
        stage: Pipeline stage where the failure occurred
            (``load``, ``ocr``, ``ner``, ``blur``, ``export``, ``config``, …).
        code: Stable machine-readable error code (e.g. ``INPUT_NOT_FOUND``).
        hint: Actionable suggestion for the user.
        context: Extra key/value details (paths, page index, dependency name…).
        cause: Original exception, if any.
        correlation_id: Opaque id linking public errors to private logs.
    """

    code: str = "REDACTORTRON_ERROR"
    stage: str = "unknown"
    default_hint: str = "Check the logs with -v / --verbose for more detail."

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        code: Optional[str] = None,
        hint: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        self.message = message
        if stage is not None:
            self.stage = stage
        if code is not None:
            self.code = code
        self.hint = hint if hint is not None else self.default_hint
        self.context: Dict[str, Any] = dict(context or {})
        self.cause = cause
        self.correlation_id = correlation_id or new_correlation_id()
        if cause is not None:
            self.__cause__ = cause
        super().__init__(self.format(audience="public"))

    def to_event(
        self,
        *,
        audience: str = "public",
        include_cause: bool = False,
    ) -> Dict[str, Any]:
        """Serilog-style structured event (safe by default for public audiences)."""
        return serialize_error(
            code=self.code,
            stage=self.stage,
            message=self.message,
            hint=self.hint,
            context=self.context,
            cause=self.cause,
            correlation_id=self.correlation_id,
            audience=audience,
            include_cause=include_cause or audience != "public",
        )

    def format(
        self,
        *,
        verbose: bool = False,
        audience: Optional[str] = None,
    ) -> str:
        """Format an error report.

        Default audience is ``public`` (no absolute paths). Pass
        ``audience='debug'`` (or ``verbose=True`` in CLI) for local diagnostics.
        """
        if audience is None:
            audience = "debug" if verbose else "public"
        event = self.to_event(
            audience=audience,
            include_cause=verbose or audience != "public",
        )
        return format_serialized_error(event, verbose=verbose or audience == "debug")

    def to_dict(self, *, audience: str = "public") -> Dict[str, Any]:
        """JSON-serializable payload for REST / Gradio (public-safe by default)."""
        # Always attach a sanitized Cause; public gets type-only.
        event = self.to_event(audience=audience, include_cause=True)
        return {
            "error": event["Code"],
            "stage": event["Stage"],
            "message": event["Message"],
            "hint": event["Hint"],
            "correlation_id": event["CorrelationId"],
            "context": event["Context"],
            "cause": event.get("Cause"),
            "serilog": event,
        }

    def log(self, *, audience: str = "internal") -> Dict[str, Any]:
        """Emit a structured log line and return the event that was logged."""
        event = self.to_event(audience=audience, include_cause=True)
        emit_structured_log(event)
        return event


class ConfigurationError(RedactortronError):
    """Invalid user configuration (threshold, labels, flags, ports)."""

    code = "CONFIGURATION_ERROR"
    stage = "config"
    default_hint = "Check CLI flags / API request parameters and try again."


class InputError(RedactortronError):
    """Missing, unreadable, or unsupported input document."""

    code = "INPUT_ERROR"
    stage = "load"
    default_hint = (
        "Provide an existing PDF or image "
        "(.pdf, .png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp)."
    )


class DependencyError(RedactortronError):
    """A required native or Python dependency failed to load."""

    code = "DEPENDENCY_ERROR"
    stage = "init"
    default_hint = (
        "Install missing packages (`pip install -e .`) and ensure Poppler "
        "is on PATH for PDF rendering."
    )


class ModelLoadError(DependencyError):
    """OCR or GLiNER model failed to download / initialize."""

    code = "MODEL_LOAD_ERROR"
    stage = "init"
    default_hint = (
        "Check network access for the first model download, disk space, "
        "and that torch / gliner / doctr are installed."
    )


class OCRError(RedactortronError):
    """docTR OCR failed on a page or document."""

    code = "OCR_ERROR"
    stage = "ocr"
    default_hint = (
        "Confirm the page is a readable raster (not an empty/corrupt image) "
        "and retry with -v for the underlying traceback."
    )


class EntityDetectionError(RedactortronError):
    """GLiNER entity detection failed."""

    code = "ENTITY_DETECTION_ERROR"
    stage = "ner"
    default_hint = (
        "Try a lower --threshold, fewer custom --labels, or re-run after "
        "confirming the GLiNER model loaded successfully."
    )


class RedactionError(RedactortronError):
    """Blur / redaction application failed."""

    code = "REDACTION_ERROR"
    stage = "blur"
    default_hint = "Ensure categories were selected and the scan completed successfully."


class ExportError(RedactortronError):
    """Writing the redacted output file failed."""

    code = "EXPORT_ERROR"
    stage = "export"
    default_hint = (
        "Check that the output path is writable and, for multi-page docs, "
        "use a .pdf destination."
    )


class CategorySelectionError(RedactortronError):
    """Invalid or missing category selection for redaction."""

    code = "CATEGORY_SELECTION_ERROR"
    stage = "select"
    default_hint = (
        "Pick one or more detected categories, or pass --categories in "
        "non-interactive mode."
    )


def wrap_unexpected(
    exc: BaseException,
    *,
    stage: str,
    message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> RedactortronError:
    """Wrap an unexpected exception while preserving debugging context."""
    if isinstance(exc, RedactortronError):
        return exc
    return RedactortronError(
        message,
        stage=stage,
        code="UNEXPECTED_ERROR",
        hint="Re-run with -v / --verbose to see the full traceback.",
        context=context,
        cause=exc,
    )
