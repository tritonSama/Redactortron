# Redactortron

**Local AI-Powered Document Redaction** — scan documents, pick sensitive categories interactively, and blur them offline.

## Features

- **Local LLM / NER** — entity detection runs on-device with [GLiNER](https://github.com/urchade/GLiNER); no cloud API required.
- **Interactive Selection** — after scanning, choose which categories to redact via a `questionary` checklist or the local Gradio Web UI.
- **Category Detection** — OCR (docTR) + NER surfaces people, orgs, emails, phone numbers, SSNs, and more.
- **OpenCV Blur** — selected spans are permanently obscured with a strong Gaussian blur.
- **PDF & image support** — ingest PDFs or common image formats and write redacted output.
- **Local Web UI** — browser interface powered by Gradio (`redactortron ui`).

## Requirements

- Python **3.9+**
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) on your `PATH` (needed by `pdf2image` for PDF rendering)

## Installation

```bash
pip install -e .
```

For a fresh virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

## Usage

### Local Web UI (Gradio)

```bash
redactortron ui
# or
redactortron-ui
```

Opens http://127.0.0.1:7860 — upload a document, scan, pick categories, blur, and download.

```bash
redactortron ui --port 7861 --no-browser
```

### Interactive CLI scan → select → blur

```bash
redactortron scan --input document.pdf
```

Or the short form:

```bash
redactortron --input document.pdf
```

You will see the Redactortron banner, a summary of detected entities, then a checklist of categories. Selected regions are blurred and written to `document_redacted.pdf` by default.

### Non-interactive (scripted) redaction

```bash
redactortron scan --input document.pdf --categories PERSON EMAIL --yes -o out.pdf
```

### Options

| Flag | Description |
|------|-------------|
| `--input` / `-i` | Input PDF or image |
| `--output` / `-o` | Output path (default: `<stem>_redacted.<ext>`) |
| `--threshold` | GLiNER confidence threshold (default `0.4`) |
| `--labels` | Custom GLiNER label list |
| `--categories` | Pre-select categories (skips checklist) |
| `--yes` / `-y` | Non-interactive mode (requires `--categories`) |
| `--name` / `--project` | Branding name shown in the CLI (default: Redactortron) |
| `--version` | Print version |
| `-v` / `--verbose` | Debug logging |

Web UI (`redactortron ui`):

| Flag | Description |
|------|-------------|
| `--host` | Bind address (default `127.0.0.1`) |
| `--port` | Port (default `7860`) |
| `--share` | Temporary public Gradio link |
| `--no-browser` | Do not auto-open the browser |

```bash
redactortron --help
redactortron scan --help
```

## Workflow

1. **Scan** — docTR OCR extracts text and word bounding boxes; GLiNER labels sensitive spans.
2. **Select Categories** — interactive checklist (or `--categories`) chooses what to redact.
3. **Blur** — OpenCV applies a heavy Gaussian blur over matching regions and writes the file.

## Project layout

```
redactortron/
├── __init__.py      # __version__ = "0.1.0"
├── core.py          # RedactortronCore (OCR, GLiNER, blur)
├── cli.py           # Interactive CLI (questionary + rich)
├── webui.py         # Local Gradio Web UI
├── models.py        # BoundingBox, DetectedEntity, ScanResult
└── assets/          # Optional icons / fonts
```

## License

MIT
