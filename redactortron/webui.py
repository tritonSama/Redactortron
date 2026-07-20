"""Local Gradio web UI for Redactortron (uses RedactortronService)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import gradio as gr

from redactortron import __version__
from redactortron.exceptions import RedactortronError
from redactortron.models import ScanResult
from redactortron.service import RedactortronService

logger = logging.getLogger("redactortron.webui")

_service: Optional[RedactortronService] = None

# Cyber neon: pink + orange + cyan.
CYBER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Exo+2:wght@400;600;700&display=swap');

:root {
  --rt-pink: #ff2d95;
  --rt-pink-hot: #ff4fd8;
  --rt-orange: #ff6b2c;
  --rt-orange-hot: #ff9f1c;
  --rt-cyan: #2de2ff;
  --rt-cyan-hot: #7af6ff;
  --rt-bg: #0a0612;
  --rt-panel: rgba(18, 8, 28, 0.82);
  --rt-line: rgba(45, 226, 255, 0.28);
}

.gradio-container {
  font-family: 'Exo 2', system-ui, sans-serif !important;
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(255, 45, 149, 0.22), transparent 55%),
    radial-gradient(900px 500px at 100% 0%, rgba(255, 107, 44, 0.18), transparent 50%),
    radial-gradient(800px 480px at 50% 110%, rgba(45, 226, 255, 0.16), transparent 55%),
    linear-gradient(165deg, #07040f 0%, #12081c 45%, #0a0612 100%) !important;
  min-height: 100vh !important;
  color: #f7e9ff !important;
}

.gradio-container::before {
  content: "";
  pointer-events: none;
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255, 45, 149, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(45, 226, 255, 0.04) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 85%);
  z-index: 0;
}

.gradio-container > * { position: relative; z-index: 1; }

.rt-hero {
  border: 1px solid var(--rt-line);
  border-radius: 18px;
  padding: 1.35rem 1.5rem 1.2rem;
  margin-bottom: 1.1rem;
  background:
    linear-gradient(135deg,
      rgba(255, 45, 149, 0.14),
      rgba(255, 107, 44, 0.10) 45%,
      rgba(45, 226, 255, 0.10) 100%);
  box-shadow:
    0 0 0 1px rgba(45, 226, 255, 0.12),
    0 0 40px rgba(255, 45, 149, 0.16),
    inset 0 1px 0 rgba(255, 255, 255, 0.06);
  position: relative;
  overflow: hidden;
}

.rt-hero::after {
  content: "";
  position: absolute;
  top: -40%;
  right: -8%;
  width: 280px;
  height: 280px;
  background: radial-gradient(circle, rgba(45, 226, 255, 0.28), transparent 65%);
  filter: blur(8px);
  pointer-events: none;
}

.rt-kicker {
  font-family: 'Orbitron', sans-serif;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  font-size: 0.72rem;
  color: var(--rt-cyan-hot);
  margin: 0 0 0.35rem;
}

.rt-title {
  font-family: 'Orbitron', sans-serif;
  font-weight: 800;
  font-size: clamp(1.7rem, 3.5vw, 2.45rem);
  line-height: 1.1;
  margin: 0;
  background: linear-gradient(100deg, #fff 8%, var(--rt-pink-hot) 38%, var(--rt-orange-hot) 68%, var(--rt-cyan-hot) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 0 24px rgba(45, 226, 255, 0.2);
}

.rt-sub {
  margin: 0.55rem 0 0;
  max-width: 48rem;
  color: #d7c2e8;
  font-size: 0.98rem;
}

.rt-chiprow {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.85rem;
}

.rt-chip {
  font-family: 'Orbitron', sans-serif;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 45, 149, 0.45);
  background: rgba(255, 45, 149, 0.12);
  color: #ffd6ef;
}

.rt-chip.alt {
  border-color: rgba(255, 107, 44, 0.5);
  background: rgba(255, 107, 44, 0.12);
  color: #ffe0c2;
}

.rt-chip.cyan {
  border-color: rgba(45, 226, 255, 0.55);
  background: rgba(45, 226, 255, 0.12);
  color: #c8f7ff;
}

footer, .footer { display: none !important; }

button.primary, .gr-button-primary {
  background: linear-gradient(105deg, var(--rt-pink), var(--rt-orange) 55%, var(--rt-cyan)) !important;
  border: none !important;
  color: #14010c !important;
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  box-shadow: 0 0 22px rgba(45, 226, 255, 0.28) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

button.primary:hover, .gr-button-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 0 32px rgba(255, 45, 149, 0.4) !important;
}

.gr-group, .gr-box, .block, .form, .panel {
  border-radius: 14px !important;
}

label, .label-wrap span {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 0.04em !important;
  color: #c8f7ff !important;
}

.rt-foot {
  margin-top: 1rem;
  font-size: 0.85rem;
  color: #b79ac9;
  border-top: 1px solid var(--rt-line);
  padding-top: 0.75rem;
}

.rt-status code {
  color: var(--rt-cyan-hot);
}
"""


def get_service() -> RedactortronService:
    """Reuse a single service so OCR / GLiNER models stay loaded."""
    global _service
    if _service is None:
        _service = RedactortronService()
    return _service


def _entity_rows(result: ScanResult) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for entity in result.all_entities:
        rows.append(
            [
                entity.entity_id,
                entity.page_index + 1,
                entity.category,
                entity.text,
                round(entity.score, 3),
            ]
        )
    return rows


def _preview_pages(source: Path, max_pages: int = 3) -> List[Any]:
    frames = get_service().core.load_pages(source)
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames[:max_pages]]


def _items_for_categories(result: ScanResult, categories: Sequence[str]) -> List[str]:
    wanted = {c.strip().upper() for c in categories}
    return [
        e.display_label()
        for e in result.all_entities
        if e.category in wanted
    ]


def scan_document(
    file_obj: Any,
    threshold: float,
) -> Tuple[Any, Any, Any, Any, Any, str]:
    """Run OCR + GLiNER; populate category + itemized transaction checklists."""
    if file_obj is None:
        raise gr.Error("Upload a PDF or image first.")

    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    service = get_service()

    try:
        summary, result = service.scan(path, threshold=float(threshold))
    except RedactortronError as exc:
        exc.log(audience="internal")
        raise gr.Error(exc.format(audience="public")) from exc

    if not any(e.entity_id for e in result.all_entities):
        result.assign_entity_ids()

    categories = summary.categories
    item_choices = result.item_choices()
    rows = _entity_rows(result)
    summary_md = (
        f"<div class='rt-status'>⚡ Found <strong>{summary.entity_count}</strong> "
        f"itemized transactions/entities across "
        f"<strong>{summary.page_count}</strong> page(s).<br/>"
        f"Categories: <code>{', '.join(categories) or '(none)'}</code><br/>"
        f"<em>Select a category to grab all of that type, or tick individual items.</em></div>"
    )

    try:
        previews = _preview_pages(path)
    except RedactortronError as exc:
        logger.warning("Preview failed after scan: %s", exc.message)
        previews = []

    state: Dict[str, Any] = {
        "path": str(path),
        "result": result,
        "threshold": float(threshold),
        "item_choices": item_choices,
    }
    category_update = gr.update(
        choices=categories,
        value=categories,
        interactive=bool(categories),
    )
    items_update = gr.update(
        choices=item_choices,
        value=item_choices,
        interactive=bool(item_choices),
    )
    return state, rows, category_update, items_update, previews, summary_md


def sync_items_from_categories(
    selected_categories: Optional[List[str]],
    state: Optional[Dict[str, Any]],
) -> Any:
    """When categories change, select every item that belongs to those categories."""
    if not state or "result" not in state:
        return gr.update()
    result: ScanResult = state["result"]
    selected = _items_for_categories(result, selected_categories or [])
    return gr.update(value=selected)


def select_all_items(state: Optional[Dict[str, Any]]) -> Any:
    if not state:
        return gr.update()
    choices = state.get("item_choices") or []
    return gr.update(value=list(choices))


def clear_items() -> Any:
    return gr.update(value=[])


def redact_document(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    items: Optional[List[str]],
) -> Tuple[Optional[str], List[Any], str]:
    """Blur selected individual items (preferred) or fall back to categories."""
    if not state or "result" not in state:
        raise gr.Error("Scan a document before redacting.")

    path = Path(state["path"])
    result: ScanResult = state["result"]
    threshold = float(state.get("threshold", 0.4))
    service = get_service()

    suffix = path.suffix.lower() or ".pdf"
    out_suffix = ".pdf" if suffix == ".pdf" or len(result.pages) > 1 else suffix
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=out_suffix,
        prefix=f"{path.stem}_redacted_",
    )
    tmp.close()
    out_path = Path(tmp.name)

    selected_items = list(items or [])
    entity_ids = [ScanResult.entity_id_from_choice(c) for c in selected_items]
    entity_ids = [e for e in entity_ids if e]

    try:
        written = service.redact(
            source=path,
            categories=list(categories or []) if not entity_ids else None,
            entity_ids=entity_ids or None,
            output=out_path,
            scan_result=result,
            threshold=threshold,
        )
        frames = service.core.load_pages(written)
        previews = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[:3]]
        if entity_ids:
            detail = f"{len(entity_ids)} item(s): {', '.join(entity_ids[:12])}"
            if len(entity_ids) > 12:
                detail += "…"
        else:
            detail = ", ".join(c.upper() for c in (categories or []))
        status = (
            f"<div class='rt-status'>◆ Redacted <strong>{detail}</strong> → "
            f"<code>{written.name}</code> ready to download.</div>"
        )
        return str(written), previews, status
    except RedactortronError as exc:
        exc.log(audience="internal")
        raise gr.Error(exc.format(audience="public")) from exc


def _build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#fff1f7",
            c100="#ffd6eb",
            c200="#ffadd6",
            c300="#ff7ebd",
            c400="#ff4fa3",
            c500="#ff2d95",
            c600="#e0167a",
            c700="#b80d61",
            c800="#8c0c4c",
            c900="#64103b",
            c950="#3d0723",
        ),
        secondary_hue=gr.themes.Color(
            c50="#ecfeff",
            c100="#cffafe",
            c200="#a5f3fc",
            c300="#67e8f9",
            c400="#22d3ee",
            c500="#2de2ff",
            c600="#0891b2",
            c700="#0e7490",
            c800="#155e75",
            c900="#164e63",
            c950="#083344",
        ),
        neutral_hue="zinc",
        font=[gr.themes.GoogleFont("Exo 2"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Orbitron"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#0a0612",
        body_background_fill_dark="#0a0612",
        body_text_color="#f7e9ff",
        body_text_color_dark="#f7e9ff",
        block_background_fill="rgba(22, 10, 34, 0.88)",
        block_background_fill_dark="rgba(22, 10, 34, 0.88)",
        block_border_color="rgba(45, 226, 255, 0.28)",
        block_border_color_dark="rgba(45, 226, 255, 0.28)",
        block_label_text_color="#c8f7ff",
        block_label_text_color_dark="#c8f7ff",
        border_color_primary="rgba(255, 107, 44, 0.45)",
        border_color_primary_dark="rgba(255, 107, 44, 0.45)",
        button_primary_background_fill="linear-gradient(105deg, #ff2d95, #ff6b2c 55%, #2de2ff)",
        button_primary_background_fill_dark="linear-gradient(105deg, #ff2d95, #ff6b2c 55%, #2de2ff)",
        button_primary_text_color="#14010c",
        button_primary_text_color_dark="#14010c",
        slider_color="#2de2ff",
        slider_color_dark="#ff2d95",
    )


def build_app():
    """Construct the Gradio Blocks app.

    Returns:
        ``(demo, theme)`` — theme is passed to ``launch()`` for Gradio 6+.
    """
    theme = _build_theme()

    with gr.Blocks(
        title=f"Redactortron v{__version__}",
        css=CYBER_CSS,
    ) as demo:
        gr.HTML(
            f"""
            <div class="rt-hero">
              <p class="rt-kicker">Local · Offline · AI Redaction</p>
              <h1 class="rt-title">REDACTORTRON v{__version__}</h1>
              <p class="rt-sub">
                Scan → pick categories or individual transactions → blur.
                Pink · orange · cyan cyber deck, running entirely on your machine.
              </p>
              <div class="rt-chiprow">
                <span class="rt-chip">OCR</span>
                <span class="rt-chip alt">GLiNER</span>
                <span class="rt-chip cyan">Transactions</span>
                <span class="rt-chip">OpenCV Blur</span>
              </div>
            </div>
            """
        )

        state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Document uplink",
                    file_types=[
                        ".pdf",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".tif",
                        ".tiff",
                        ".bmp",
                        ".webp",
                    ],
                    type="filepath",
                )
                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="Detection threshold",
                )
                scan_btn = gr.Button("01 · Scan document", variant="primary")
                categories = gr.CheckboxGroup(
                    label="02 · Categories (select all of a type)",
                    choices=[],
                    interactive=False,
                )
                with gr.Row():
                    select_all_btn = gr.Button("Select all items")
                    clear_btn = gr.Button("Clear items")
                items = gr.CheckboxGroup(
                    label="03 · Individual transactions / entities",
                    choices=[],
                    interactive=False,
                )
                redact_btn = gr.Button("04 · Blur & export", variant="primary")
                status = gr.Markdown(
                    "<div class='rt-status'>Awaiting uplink — upload a file, then scan.</div>"
                )

            with gr.Column(scale=2):
                entities = gr.Dataframe(
                    headers=["ID", "Page", "Category", "Text", "Score"],
                    datatype=["str", "number", "str", "str", "number"],
                    label="Itemized entity matrix",
                    interactive=False,
                    wrap=True,
                )
                preview = gr.Gallery(
                    label="Visual feed",
                    columns=2,
                    height=420,
                    object_fit="contain",
                )
                download = gr.File(label="Redacted package")

        scan_btn.click(
            fn=scan_document,
            inputs=[file_in, threshold],
            outputs=[state, entities, categories, items, preview, status],
        )
        categories.change(
            fn=sync_items_from_categories,
            inputs=[categories, state],
            outputs=[items],
        )
        select_all_btn.click(fn=select_all_items, inputs=[state], outputs=[items])
        clear_btn.click(fn=clear_items, outputs=[items])
        redact_btn.click(
            fn=redact_document,
            inputs=[state, categories, items],
            outputs=[download, preview, status],
        )

        gr.HTML(
            "<p class='rt-foot'>Tip: tick a category to select every matching transaction, "
            "then fine-tune with the item list. First run downloads OCR / GLiNER weights.</p>"
        )

    return demo, theme


def _find_free_port(host: str, start: int, span: int = 20) -> int:
    """Return the first free TCP port in ``[start, start+span)``."""
    import socket

    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No free port in range {start}-{start + span - 1}. "
        "Pass a different --port or set GRADIO_SERVER_PORT."
    )


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    """Start the local Gradio server (auto-picks a free port if needed)."""
    app, theme = build_app()
    chosen = port
    try:
        chosen = _find_free_port(host, port)
    except OSError:
        chosen = port
    if chosen != port:
        logger.warning("Port %s busy — using %s instead", port, chosen)

    logger.info("Starting Redactortron Web UI on http://%s:%s", host, chosen)
    app.launch(
        server_name=host,
        server_port=chosen,
        share=share,
        inbrowser=inbrowser,
        theme=theme,
        css=CYBER_CSS,
    )


def main() -> None:
    """Entry point for ``redactortron-ui`` / ``python -m redactortron.webui``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    launch()


if __name__ == "__main__":
    main()
