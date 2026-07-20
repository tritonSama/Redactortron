"""Interactive CLI for Redactortron."""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from redactortron import __version__
from redactortron.core import DEFAULT_LABELS
from redactortron.exceptions import CategorySelectionError, RedactortronError
from redactortron.models import ScanResult
from redactortron.service import RedactortronService

console = Console()
logger = logging.getLogger("redactortron")

BANNER = f"""\
==================================
 REDACTORTRON v{__version__}
 Local AI-Powered Document Redaction
==================================\
"""


def print_banner() -> None:
    """Render the Redactortron branding banner with rich."""
    console.print(
        Panel(
            Text(BANNER, justify="center", style="bold cyan"),
            border_style="cyan",
            expand=False,
        )
    )


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def report_error(exc: RedactortronError, *, verbose: bool = False) -> None:
    """Print a public-safe error; log a richer Serilog-style internal event."""
    exc.log(audience="internal")
    console.print(
        Panel(
            exc.format(verbose=verbose, audience="debug" if verbose else "public"),
            title="[bold red]Redactortron error[/bold red]",
            border_style="red",
            expand=False,
        )
    )
    console.print(
        f"[dim]Correlation id:[/dim] {exc.correlation_id} "
        f"[dim](safe to share; full paths are not shown unless -v)[/dim]"
    )
    if verbose and exc.__cause__ is not None:
        console.print("[dim]Traceback:[/dim]")
        console.print("".join(traceback.format_exception(exc.__cause__)))


def _show_scan_summary(result: ScanResult) -> None:
    entities = result.all_entities
    table = Table(title="Detected Entities", show_lines=False)
    table.add_column("Page", justify="right", style="dim")
    table.add_column("Category", style="magenta")
    table.add_column("Text")
    table.add_column("Score", justify="right")

    for entity in entities[:50]:
        table.add_row(
            str(entity.page_index + 1),
            entity.category,
            entity.text[:60],
            f"{entity.score:.2f}",
        )

    if len(entities) > 50:
        table.caption = f"Showing 50 of {len(entities)} entities"

    console.print(table)
    console.print(
        f"[bold]{len(entities)}[/bold] entities across "
        f"[bold]{len(result.pages)}[/bold] page(s); "
        f"categories: [cyan]{', '.join(result.categories()) or '(none)'}[/cyan]"
    )


def select_categories(categories: Sequence[str]) -> List[str]:
    """Interactive checklist for entity categories to redact."""
    if not categories:
        console.print("[yellow]No entity categories detected — nothing to select.[/yellow]")
        return []

    choices = questionary.checkbox(
        "Select categories to redact (space to toggle, enter to confirm):",
        choices=list(categories),
    ).ask()

    if choices is None:
        console.print("[red]Selection cancelled.[/red]")
        return []
    return list(choices)


def run_scan_workflow(
    input_path: Path,
    output_path: Optional[Path],
    threshold: float,
    labels: Optional[List[str]],
    categories: Optional[List[str]],
    non_interactive: bool,
    *,
    verbose: bool = False,
) -> int:
    """Scan → select categories → blur pipeline via RedactortronService."""
    print_banner()
    console.print("[bold]Project:[/bold] Redactortron")
    console.print(f"[bold]Input:[/bold]  {input_path}")

    service = RedactortronService(labels=labels)
    try:
        with console.status("[bold cyan]Scanning document (OCR + GLiNER)…[/bold cyan]"):
            _summary, result = service.scan(
                input_path, threshold=threshold, labels=labels
            )
    except RedactortronError as exc:
        report_error(exc, verbose=verbose)
        return 1

    _show_scan_summary(result)

    if not result.all_entities:
        console.print("[yellow]No entities found. Exiting without redaction.[/yellow]")
        return 0

    try:
        if categories:
            selected = service.validate_categories(
                categories,
                available=result.categories(),
                require_non_empty=True,
            )
        elif non_interactive:
            raise CategorySelectionError(
                "--categories is required when using --yes / non-interactive mode.",
                hint="Example: redactortron scan -i doc.pdf --categories PERSON EMAIL --yes",
            )
        else:
            selected = select_categories(result.categories())
            if not selected:
                console.print("[yellow]No categories selected. Exiting.[/yellow]")
                return 0
            selected = service.validate_categories(
                selected,
                available=result.categories(),
                require_non_empty=True,
            )
    except RedactortronError as exc:
        report_error(exc, verbose=verbose)
        return 1

    dest = output_path or service.default_output_path(input_path)
    console.print(f"[bold]Redacting:[/bold] {', '.join(selected)}")
    console.print(f"[bold]Output:[/bold]  {dest}")

    try:
        with console.status("[bold cyan]Applying OpenCV blur…[/bold cyan]"):
            written = service.redact(
                source=input_path,
                categories=selected,
                output=dest,
                scan_result=result,
                threshold=threshold,
            )
    except RedactortronError as exc:
        report_error(exc, verbose=verbose)
        return 1

    console.print(f"[green]Done.[/green] Redacted file written to [bold]{written}[/bold]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redactortron",
        description=(
            "Redactortron — local AI-powered document redaction "
            "(docTR OCR + GLiNER + OpenCV blur)."
        ),
        epilog=(
            "Examples:\n"
            "  redactortron scan --input document.pdf\n"
            "  redactortron ui\n"
            "  redactortron serve\n"
            "  redactortron --input file.pdf --name Redactortron"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Redactortron {__version__}",
    )
    parser.add_argument(
        "--name",
        "--project",
        dest="project_name",
        default="Redactortron",
        help="Project / branding name shown in the CLI (default: Redactortron).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging and full error causes.",
    )

    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser(
        "scan",
        help="Scan a document, interactively select categories, then blur.",
        description="Scan → Select Categories → Blur workflow for Redactortron.",
    )
    scan.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the input PDF or image.",
    )
    scan.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Destination path for the redacted file (default: <stem>_redacted.<ext>).",
    )
    scan.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="GLiNER confidence threshold (default: 0.4).",
    )
    scan.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=f"Optional GLiNER labels (default: {', '.join(DEFAULT_LABELS)}).",
    )
    scan.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Pre-select categories to redact (skips interactive checklist).",
    )
    scan.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive mode (requires --categories).",
    )

    ui = sub.add_parser(
        "ui",
        help="Launch the local Gradio web UI.",
        description="Start Redactortron's local browser interface (Gradio).",
    )
    ui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    ui.add_argument("--port", type=int, default=7860, help="Port (default: 7860).")
    ui.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio share link (off by default).",
    )
    ui.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser window.",
    )

    serve = sub.add_parser(
        "serve",
        help="Launch the optional FastAPI HTTP API.",
        description="REST API for scan/redact (requires: pip install -e '.[api]').",
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (development only).",
    )

    parser.add_argument(
        "--input",
        "-i",
        dest="top_input",
        type=Path,
        default=None,
        help="Shortcut for `scan --input` (e.g. redactortron --input file.pdf).",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="top_output",
        type=Path,
        default=None,
        help="Shortcut output path when using top-level --input.",
    )
    parser.add_argument(
        "--threshold",
        dest="top_threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--categories",
        dest="top_categories",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yes",
        "-y",
        dest="top_yes",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Console entry point: ``redactortron = redactortron.cli:main``."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)

    if getattr(args, "project_name", None) and args.project_name != "Redactortron":
        console.print(f"[dim]Running as project:[/dim] {args.project_name}")

    if args.command == "scan":
        code = run_scan_workflow(
            input_path=args.input,
            output_path=args.output,
            threshold=args.threshold,
            labels=args.labels,
            categories=args.categories,
            non_interactive=args.yes,
            verbose=args.verbose,
        )
        raise SystemExit(code)

    if args.command == "ui":
        print_banner()
        try:
            from redactortron.webui import launch
        except Exception as exc:  # noqa: BLE001
            report_error(
                RedactortronError(
                    "Failed to import the Gradio Web UI.",
                    stage="init",
                    code="UI_IMPORT_ERROR",
                    cause=exc,
                    hint="Install Gradio: pip install -e .",
                ),
                verbose=args.verbose,
            )
            raise SystemExit(1) from exc

        console.print(
            f"[bold cyan]Opening local Web UI[/bold cyan] → "
            f"http://{args.host}:{args.port}"
        )
        try:
            launch(
                host=args.host,
                port=args.port,
                share=args.share,
                inbrowser=not args.no_browser,
            )
        except RedactortronError as exc:
            report_error(exc, verbose=args.verbose)
            raise SystemExit(1) from exc
        raise SystemExit(0)

    if args.command == "serve":
        print_banner()
        try:
            from redactortron.api import launch as launch_api
        except ImportError as exc:
            report_error(
                RedactortronError(
                    "API dependencies are not installed.",
                    stage="init",
                    code="API_IMPORT_ERROR",
                    cause=exc,
                    hint="Install with: pip install -e '.[api]'",
                ),
                verbose=args.verbose,
            )
            raise SystemExit(1) from exc

        console.print(
            f"[bold cyan]Starting HTTP API[/bold cyan] → "
            f"http://{args.host}:{args.port}/docs"
        )
        launch_api(host=args.host, port=args.port, reload=args.reload)
        raise SystemExit(0)

    if args.top_input is not None:
        code = run_scan_workflow(
            input_path=args.top_input,
            output_path=args.top_output,
            threshold=args.top_threshold if args.top_threshold is not None else 0.4,
            labels=None,
            categories=args.top_categories,
            non_interactive=bool(args.top_yes),
            verbose=args.verbose,
        )
        raise SystemExit(code)

    parser.print_help()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
