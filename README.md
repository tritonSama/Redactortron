# Redactortron

**Local AI-Powered Document Redaction** — scan documents, pick sensitive categories interactively, and blur them offline. One shared service powers the **CLI**, **Gradio Web UI**, and optional **HTTP API**.

## Features

- **Local LLM / NER** — entity detection runs on-device with [GLiNER](https://github.com/urchade/GLiNER); no cloud API required.
- **Interactive Selection** — category checklist in the CLI (`questionary`) or Web UI.
- **Category Detection** — OCR (docTR) + NER surfaces people, orgs, emails, phone numbers, SSNs, and more.
- **OpenCV Blur** — selected spans are permanently obscured with a strong Gaussian blur.
- **PDF & image support** — ingest PDFs or common image formats and write redacted output.
- **Local Web UI** — Gradio browser interface (`redactortron ui`).
- **Optional HTTP API** — FastAPI endpoints for scan/redact (`redactortron serve`).
- **Debuggable errors** — every failure reports **code**, **stage**, **context**, and a **hint**.

## Requirements

- Python **3.9+**
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) on your `PATH` (needed by `pdf2image` for PDF rendering)

## Installation

### One-shot installer (recommended)

Windows (PowerShell):

```powershell
.\scripts\install.ps1
# optional extras:
.\scripts\install.ps1 -Api -Dev
```

Or double-click / CMD: `scripts\install.bat`

Cross-platform:

```bash
python scripts/install_deps.py --with-poppler
python scripts/install_deps.py --api --dev --with-poppler
```

This installs the Python package (editable), pulls **Poppler** on Windows into `.tools/poppler` (for PDF rendering), and verifies imports.

### Manual pip

```bash
pip install -e .
pip install -e ".[api]"
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
redactortron scan --input document.pdf
redactortron scan --input document.pdf --categories PERSON EMAIL --yes -o out.pdf
redactortron --input document.pdf -v   # verbose errors / logging
```

### Web UI (Gradio)

```bash
python -m redactortron ui
# or
python -m redactortron.webui
```

Opens http://127.0.0.1:7860 (or the next free port if 7860 is busy).

### HTTP API (optional)

```bash
pip install -e ".[api]"
redactortron serve
# docs: http://127.0.0.1:8000/docs
```

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Service / model load status |
| `POST` | `/v1/scan` | Upload file → entity summary JSON |
| `POST` | `/v1/redact` | Upload file + categories → redacted download |

Errors return JSON shaped like:

```json
{
  "error": "INPUT_NOT_FOUND",
  "stage": "load",
  "message": "Input file not found: …",
  "hint": "Check the path spelling…",
  "context": { "path": "…" }
}
```

### Programmatic (shared service)

```python
from redactortron import RedactortronService

service = RedactortronService()
summary, result = service.scan("document.pdf")
path = service.redact(
    "document.pdf",
    categories=["PERSON", "EMAIL"],
    scan_result=result,
)
```

## Debugging & secure error serialization

Pipeline stages: `config` → `load` → `init` → `ocr` → `ner` → `select` → `blur` → `export`.

Errors are emitted as **Serilog-style structured events** (`@t`, `@l`, `@mt`, `Code`, `Stage`, `CorrelationId`, …) via `RedactortronError.to_event()` / `.to_dict()` / `.log()`.

**Security defaults (public audience — UI & API):**
- Absolute paths and usernames are redacted.
- Real filenames are hidden (`<redacted>/<file>.pdf`) — titles can be sensitive.
- Underlying exception **messages** are omitted (type only) — they often embed paths.
- A short `correlation_id` is shown so you can match UI errors to server logs without leaking PII.

**Internal / debug:** CLI `-v` uses `audience='debug'`; logs use `audience='internal'` (basename + path fingerprint, no full directory tree).

Example public error:

```text
[PDF_RENDER_ERROR] stage=load id=a1b2c3d4e5f6
  Problem: Failed to render PDF pages.
  Where:   path='<redacted>/<file>.pdf', dpi=200
  Hint:    Install Poppler and ensure `pdftoppm` is on PATH …
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Workflow

1. **Scan** — docTR OCR extracts text and word bounding boxes; GLiNER labels sensitive spans.
2. **Select Categories** — checklist / `--categories` / API form field.
3. **Blur** — OpenCV applies a heavy Gaussian blur over matching regions and writes the file.

## Project layout

```
redactortron/
├── __init__.py
├── exceptions.py    # Stage-aware error hierarchy
├── security_log.py  # Serilog-style events + path/PII redaction
├── service.py       # Shared facade (CLI / UI / API)
├── core.py          # RedactortronCore (OCR, GLiNER, blur)
├── cli.py           # Terminal UI
├── webui.py         # Gradio Web UI
├── api.py           # Optional FastAPI app
├── models.py
└── assets/
tests/
```

## License

MIT
