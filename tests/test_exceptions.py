"""Tests for stage-aware exceptions and secure Serilog-style serialization."""

from __future__ import annotations

from redactortron.exceptions import (
    ConfigurationError,
    InputError,
    OCRError,
    RedactortronError,
    wrap_unexpected,
)
from redactortron.security_log import sanitize_path, serialize_error


def test_format_includes_stage_code_hint_and_context() -> None:
    err = InputError(
        "Input file not found.",
        code="INPUT_NOT_FOUND",
        context={"path": "missing.pdf"},
        hint="Check the path spelling.",
    )
    text = err.format(audience="public")
    assert "[INPUT_NOT_FOUND]" in text
    assert "stage=load" in text
    assert "Problem: Input file not found" in text
    assert "<redacted>/<file>.pdf" in text
    assert "Hint:    Check the path spelling." in text
    assert err.correlation_id


def test_public_format_redacts_absolute_paths() -> None:
    err = InputError(
        "Failed to render PDF pages.",
        code="PDF_RENDER_ERROR",
        context={
            "path": r"C:\Users\NEW\AppData\Local\Temp\gradio\abc\march 8 stub.pdf",
            "dpi": 200,
        },
        cause=RuntimeError(
            r"Unable to get page count for C:\Users\NEW\secret\doc.pdf"
        ),
    )
    public = err.format(audience="public")
    assert "C:\\Users\\NEW" not in public
    assert "AppData" not in public
    assert "march 8 stub" not in public
    assert "<redacted>/<file>.pdf" in public
    assert "secret" not in public

    payload = err.to_dict(audience="public")
    assert payload["context"]["path"] == "<redacted>/<file>.pdf"
    assert payload["context"]["dpi"] == 200
    assert payload["cause"] == {"type": "RuntimeError"}
    assert "serilog" in payload
    assert payload["serilog"]["@mt"].startswith("Redactortron")


def test_internal_event_uses_fingerprint_not_full_tree() -> None:
    path = r"C:\Users\NEW\Documents\secret-paycheck.pdf"
    event = serialize_error(
        code="PDF_RENDER_ERROR",
        stage="load",
        message="Failed to render PDF pages.",
        hint="Install Poppler.",
        context={"path": path},
        audience="internal",
    )
    assert event["Context"]["path"]["name"] == "secret-paycheck.pdf"
    assert "fingerprint" in event["Context"]["path"]
    assert "Users\\NEW" not in str(event["Context"])


def test_sanitize_path_debug_keeps_original() -> None:
    raw = r"C:\Users\NEW\file.pdf"
    assert sanitize_path(raw, audience="debug") == raw


def test_to_dict_is_api_friendly() -> None:
    cause = ValueError("boom")
    err = OCRError(
        "OCR failed on page 1.",
        context={"page_index": 0},
        cause=cause,
    )
    payload = err.to_dict(audience="public")
    assert payload["error"] == "OCR_ERROR"
    assert payload["stage"] == "ocr"
    assert payload["context"]["page_index"] == 0
    assert payload["cause"] == {"type": "ValueError"}
    assert payload["correlation_id"] == err.correlation_id


def test_wrap_unexpected_preserves_redactortron_errors() -> None:
    original = ConfigurationError("bad threshold")
    wrapped = wrap_unexpected(original, stage="scan", message="ignored")
    assert wrapped is original


def test_wrap_unexpected_wraps_generic_errors() -> None:
    wrapped = wrap_unexpected(
        RuntimeError("kaboom"),
        stage="blur",
        message="Blur failed.",
        context={"page": 2},
    )
    assert isinstance(wrapped, RedactortronError)
    assert wrapped.code == "UNEXPECTED_ERROR"
    assert wrapped.stage == "blur"
    assert wrapped.context["page"] == 2
    assert isinstance(wrapped.cause, RuntimeError)


def test_verbose_debug_format_includes_cause_type() -> None:
    err = OCRError("fail", cause=RuntimeError("inner"))
    text = err.format(verbose=True, audience="debug")
    assert "Cause:" in text
    assert "RuntimeError" in text
