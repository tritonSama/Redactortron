"""CLI parser and error-reporting smoke tests."""

from __future__ import annotations

from redactortron.cli import build_parser
from redactortron.exceptions import InputError


def test_parser_scan_ui_and_serve() -> None:
    parser = build_parser()

    scan_args = parser.parse_args(["scan", "--input", "doc.pdf"])
    assert scan_args.command == "scan"
    assert str(scan_args.input).endswith("doc.pdf")

    ui_args = parser.parse_args(["ui", "--port", "9000", "--no-browser"])
    assert ui_args.command == "ui"
    assert ui_args.port == 9000
    assert ui_args.no_browser is True

    serve_args = parser.parse_args(["serve", "--port", "8001"])
    assert serve_args.command == "serve"
    assert serve_args.port == 8001


def test_top_level_input_shortcut() -> None:
    parser = build_parser()
    args = parser.parse_args(["--input", "file.pdf", "--categories", "PERSON"])
    assert args.top_input is not None
    assert args.top_categories == ["PERSON"]


def test_input_error_format_usable_in_cli() -> None:
    err = InputError(
        "Input file not found: x.pdf",
        code="INPUT_NOT_FOUND",
        context={"path": "x.pdf"},
    )
    text = err.format(verbose=True, audience="debug")
    assert "INPUT_NOT_FOUND" in text
    assert "stage=load" in text
    assert err.correlation_id in err.to_dict()["correlation_id"]
