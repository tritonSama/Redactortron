"""Local Gradio web UI for Redactortron (uses RedactortronService)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np

from redactortron import __version__
from redactortron.analysis import (
    DATE_CATEGORIES,
    build_date_tree,
    find_all_dates,
    find_word_matches,
    line_entities_for,
)
from redactortron.core import DEFAULT_LABELS
from redactortron.exceptions import RedactortronError
from redactortron.models import DetectedEntity, ScanResult
from redactortron.service import RedactortronService
from redactortron.statement import ParsedStatement, Transaction, parse_statement
from redactortron.taxonomy import MATRIX_VIEWS, view_allows_family

logger = logging.getLogger("redactortron.webui")

_service: Optional[RedactortronService] = None

CYAN = (255, 226, 45)  # BGR cyber highlight
CYAN_FILL = (255, 226, 45)
TRINIDAD_STROKE = (255, 255, 255)  # white box
TRINIDAD_FILL = (38, 17, 206)  # red fill in BGR (#ce1126)


def _is_trinidad_theme(mode: Optional[str]) -> bool:
    text = str(mode or "").lower()
    return "hitek" in text or "red" in text


def _highlight_colors(mode: Optional[str]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if _is_trinidad_theme(mode):
        return TRINIDAD_STROKE, TRINIDAD_FILL
    return CYAN, CYAN_FILL

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
}

.rt-sub {
  margin: 0.55rem 0 0;
  max-width: 48rem;
  color: #d7c2e8;
  font-size: 0.98rem;
}

.rt-chiprow { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.85rem; }

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
}

label, .label-wrap span {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 0.04em !important;
}

.rt-foot {
  margin-top: 1rem;
  font-size: 0.85rem;
  color: #b79ac9;
  border-top: 1px solid var(--rt-line);
  padding-top: 0.75rem;
}

.rt-status code { color: var(--rt-cyan-hot); }

.rt-highlight-note {
  font-size: 0.82rem;
  color: #7af6ff;
  margin: 0.35rem 0 0.6rem;
}

.rt-settings-note {
  font-size: 0.8rem;
  color: #b79ac9;
  margin: 0.25rem 0 0;
}
"""

# Injected live when the theme radio changes (Gradio 6-safe; no DOM class hacks).
THEME_CYBER_LABEL = "POC (Cyan / Pink / Orange)"
THEME_TRINIDAD_LABEL = "Hitek (Red / Black / White)"

TRINIDAD_THEME_CSS = """
/* Remap every cyber accent token → Trinidad red / white / black */
.gradio-container,
:root {
  --rt-pink: #ce1126 !important;
  --rt-pink-hot: #ff2a3c !important;
  --rt-orange: #ffffff !important;
  --rt-orange-hot: #f0f0f0 !important;
  --rt-cyan: #000000 !important;
  --rt-cyan-hot: #ffffff !important;
  --rt-bg: #1a0003 !important;
  --rt-line: rgba(255, 255, 255, 0.5) !important;

  --primary-50: #fff5f5 !important;
  --primary-100: #ffd6d9 !important;
  --primary-200: #ffadb3 !important;
  --primary-300: #f26a74 !important;
  --primary-400: #e2303d !important;
  --primary-500: #ce1126 !important;
  --primary-600: #a50e1f !important;
  --primary-700: #7d0b18 !important;
  --primary-800: #550810 !important;
  --primary-900: #2e0509 !important;
  --primary-950: #140204 !important;

  --secondary-50: #ffffff !important;
  --secondary-100: #f5f5f5 !important;
  --secondary-200: #e5e5e5 !important;
  --secondary-300: #cccccc !important;
  --secondary-400: #999999 !important;
  --secondary-500: #000000 !important;
  --secondary-600: #111111 !important;
  --secondary-700: #1a1a1a !important;
  --secondary-800: #0a0a0a !important;
  --secondary-900: #000000 !important;
  --secondary-950: #000000 !important;

  --color-accent: #ce1126 !important;
  --color-accent-soft: rgba(206, 17, 38, 0.25) !important;
  --border-color-accent: #ffffff !important;
  --border-color-primary: rgba(255, 255, 255, 0.4) !important;
  --link-text-color: #ffffff !important;
  --link-text-color-active: #ce1126 !important;
  --link-text-color-hover: #ffd6d9 !important;
  --checkbox-background-color-selected: #ce1126 !important;
  --checkbox-border-color-selected: #ffffff !important;
  --checkbox-border-color-focus: #ffffff !important;
  --checkbox-label-background-fill-selected: rgba(206, 17, 38, 0.35) !important;
  --checkbox-label-border-color-selected: #ffffff !important;
  --slider-color: #ce1126 !important;
  --button-primary-background-fill: linear-gradient(105deg, #ce1126, #000000 50%, #ffffff) !important;
  --button-primary-background-fill-hover: linear-gradient(105deg, #ff2a3c, #111111 50%, #ffffff) !important;
  --button-primary-text-color: #ce1126 !important;
  --button-secondary-background-fill: #000000 !important;
  --button-secondary-text-color: #ffffff !important;
  --body-background-fill: #1a0003 !important;
  --body-text-color: #fff8f8 !important;
  --block-background-fill: rgba(0, 0, 0, 0.78) !important;
  --block-border-color: rgba(255, 255, 255, 0.4) !important;
  --block-label-text-color: #ffffff !important;
  --block-label-background-fill: #000000 !important;
  --block-label-border-color: #ffffff !important;
  --block-title-text-color: #ffffff !important;
  --input-background-fill: rgba(0, 0, 0, 0.65) !important;
  --input-border-color: rgba(255, 255, 255, 0.35) !important;
  --input-placeholder-color: #bbbbbb !important;
}

.gradio-container {
  background:
    linear-gradient(135deg,
      #ce1126 0%, #ce1126 36%,
      #ffffff 36%, #ffffff 40%,
      #000000 40%, #000000 60%,
      #ffffff 60%, #ffffff 64%,
      #ce1126 64%) !important;
  color: #fff8f8 !important;
}
.gradio-container::before {
  opacity: 0.1 !important;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 0, 0, 0.08) 1px, transparent 1px) !important;
}

.rt-hero {
  background: rgba(0, 0, 0, 0.82) !important;
  border: 1px solid #ffffff !important;
  box-shadow: 0 0 0 2px #000, 0 12px 36px rgba(0, 0, 0, 0.55) !important;
}
.rt-kicker { color: #ffffff !important; }
.rt-title {
  background: linear-gradient(100deg, #ffffff 10%, #ce1126 50%, #ffffff 90%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  color: transparent !important;
  text-shadow: none !important;
}
.rt-sub { color: #f3e9e9 !important; }

.rt-chip,
.rt-chip.alt,
.rt-chip.cyan {
  border-color: #ffffff !important;
  color: #ffffff !important;
}
.rt-chip { background: #ce1126 !important; }
.rt-chip.alt { background: #000000 !important; }
.rt-chip.cyan { background: rgba(255, 255, 255, 0.2) !important; color: #fff !important; }

label, .label-wrap span, .block-label, span.svelte-1gfkntr {
  font-family: 'Orbitron', sans-serif !important;
}

/* Trinidad labels: black / white / red — not pink */
.block .label-wrap,
.block .label-wrap span,
.block-label,
.block-label span,
div.label-wrap,
div.label-wrap > span {
  background: #000000 !important;
  background-color: #000000 !important;
  color: #ffffff !important;
  border: 1px solid #ffffff !important;
  border-color: #ffffff !important;
}

button, .gr-button, button.primary, .gr-button-primary,
button.secondary, .gr-button-secondary {
  background: linear-gradient(105deg, #ce1126 0%, #000000 50%, #ffffff 100%) !important;
  border: 1px solid #ffffff !important;
  color: #ce1126 !important;
  box-shadow: 0 0 18px rgba(0, 0, 0, 0.55) !important;
}
button:hover, .gr-button:hover {
  filter: brightness(1.08);
}

.block, .form, .panel, .gr-group, .gr-box, .svelte-1edwuft,
.contain, .overflow-auto, table, .dataframe, .gallery {
  background: rgba(0, 0, 0, 0.78) !important;
  border-color: rgba(255, 255, 255, 0.4) !important;
  color: #fff8f8 !important;
}

input, textarea, select, .wrap, .secondary-wrap {
  background: rgba(0, 0, 0, 0.7) !important;
  border-color: rgba(255, 255, 255, 0.4) !important;
  color: #ffffff !important;
}

input[type="range"], .slider input {
  accent-color: #ce1126 !important;
}
.slider .svelte-11iul4p, [class*="slider"] {
  --slider-color: #ce1126 !important;
}

input[type="checkbox"], input[type="radio"] {
  accent-color: #ce1126 !important;
}
.selected, [aria-checked="true"], .checked {
  border-color: #ffffff !important;
  background-color: rgba(206, 17, 38, 0.45) !important;
  color: #ffffff !important;
}

a, .prose a { color: #ffffff !important; }
code, .rt-status code { color: #ffffff !important; background: rgba(206, 17, 38, 0.35) !important; }

.rt-foot {
  color: #f0dede !important;
  border-top-color: rgba(255, 255, 255, 0.45) !important;
}
.rt-highlight-note { color: #ffffff !important; }

/* Kill leftover pink / orange / cyan glows */
* {
  --tw-shadow-color: rgba(0, 0, 0, 0.45) !important;
}
"""

CYBER_THEME_CSS = """
.gradio-container,
:root {
  --rt-pink: #ff2d95 !important;
  --rt-pink-hot: #ff4fd8 !important;
  --rt-orange: #ff6b2c !important;
  --rt-orange-hot: #ff9f1c !important;
  --rt-cyan: #2de2ff !important;
  --rt-cyan-hot: #7af6ff !important;
  --rt-bg: #0a0612 !important;
  --rt-line: rgba(45, 226, 255, 0.28) !important;

  --primary-50: #fff1f7 !important;
  --primary-100: #ffd6eb !important;
  --primary-200: #ffadd6 !important;
  --primary-300: #ff7ebd !important;
  --primary-400: #ff4fa3 !important;
  --primary-500: #ff2d95 !important;
  --primary-600: #e0167a !important;
  --primary-700: #b80d61 !important;
  --primary-800: #8c0c4c !important;
  --primary-900: #64103b !important;
  --primary-950: #3d0723 !important;

  --secondary-50: #ecfeff !important;
  --secondary-100: #cffafe !important;
  --secondary-200: #a5f3fc !important;
  --secondary-300: #67e8f9 !important;
  --secondary-400: #22d3ee !important;
  --secondary-500: #2de2ff !important;
  --secondary-600: #0891b2 !important;
  --secondary-700: #0e7490 !important;
  --secondary-800: #155e75 !important;
  --secondary-900: #164e63 !important;
  --secondary-950: #083344 !important;

  --color-accent: #ff2d95 !important;
  --color-accent-soft: rgba(255, 45, 149, 0.2) !important;
  --border-color-accent: #ff6b2c !important;
  --slider-color: #2de2ff !important;
  --button-primary-background-fill: linear-gradient(105deg, #ff2d95, #ff6b2c 55%, #2de2ff) !important;
  --button-primary-text-color: #14010c !important;
  --body-background-fill: #0a0612 !important;
  --body-text-color: #f7e9ff !important;
  --block-background-fill: rgba(22, 10, 34, 0.88) !important;
  --block-border-color: rgba(45, 226, 255, 0.28) !important;
  --block-label-text-color: #ffffff !important;
  --block-label-background-fill: #ff2d95 !important;
  --block-label-border-color: #ff2d95 !important;
  --block-title-text-color: #c8f7ff !important;
}

.gradio-container {
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(255, 45, 149, 0.22), transparent 55%),
    radial-gradient(900px 500px at 100% 0%, rgba(255, 107, 44, 0.18), transparent 50%),
    radial-gradient(800px 480px at 50% 110%, rgba(45, 226, 255, 0.16), transparent 55%),
    linear-gradient(165deg, #07040f 0%, #12081c 45%, #0a0612 100%) !important;
  color: #f7e9ff !important;
}
.gradio-container::before {
  opacity: 1 !important;
  background-image:
    linear-gradient(rgba(255, 45, 149, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(45, 226, 255, 0.04) 1px, transparent 1px) !important;
}
.rt-hero {
  background:
    linear-gradient(135deg,
      rgba(255, 45, 149, 0.14),
      rgba(255, 107, 44, 0.10) 45%,
      rgba(45, 226, 255, 0.10) 100%) !important;
  border-color: rgba(45, 226, 255, 0.28) !important;
  box-shadow:
    0 0 0 1px rgba(45, 226, 255, 0.12),
    0 0 40px rgba(255, 45, 149, 0.16),
    inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
}
.rt-kicker { color: #7af6ff !important; }
.rt-title {
  background: linear-gradient(100deg, #fff 8%, #ff4fd8 38%, #ff9f1c 68%, #7af6ff 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  color: transparent !important;
}
.rt-sub { color: #d7c2e8 !important; }
.rt-chip {
  border-color: rgba(255, 45, 149, 0.45) !important;
  background: rgba(255, 45, 149, 0.12) !important;
  color: #ffd6ef !important;
}
.rt-chip.alt {
  border-color: rgba(255, 107, 44, 0.5) !important;
  background: rgba(255, 107, 44, 0.12) !important;
  color: #ffe0c2 !important;
}
.rt-chip.cyan {
  border-color: rgba(45, 226, 255, 0.55) !important;
  background: rgba(45, 226, 255, 0.12) !important;
  color: #c8f7ff !important;
}
label, .label-wrap span { font-family: 'Orbitron', sans-serif !important; }

/* Pink labels only for Pink / Cyan / Orange theme */
.block .label-wrap,
.block .label-wrap span,
.block-label,
.block-label span,
div.label-wrap,
div.label-wrap > span {
  background: #ff2d95 !important;
  background-color: #ff2d95 !important;
  color: #ffffff !important;
  border-color: #ff2d95 !important;
}

button.primary, .gr-button-primary {
  background: linear-gradient(105deg, #ff2d95, #ff6b2c 55%, #2de2ff) !important;
  color: #14010c !important;
  box-shadow: 0 0 22px rgba(45, 226, 255, 0.28) !important;
  border: none !important;
}
.block, .form, .panel, .gr-group, .gr-box {
  background: rgba(22, 10, 34, 0.88) !important;
  border-color: rgba(45, 226, 255, 0.28) !important;
}
.rt-foot { color: #b79ac9 !important; border-top-color: rgba(45, 226, 255, 0.28) !important; }
.rt-status code, .rt-highlight-note { color: #7af6ff !important; }
input[type="range"] { accent-color: #2de2ff !important; }
input[type="checkbox"], input[type="radio"] { accent-color: #ff2d95 !important; }
"""


def render_theme_css(mode: str) -> str:
    """Return a silent HTML ``<style>`` tag that restyles the live GUI."""
    css = TRINIDAD_THEME_CSS if _is_trinidad_theme(mode) else CYBER_THEME_CSS
    return f"<style id='rt-live-theme'>{css}</style>"


def get_service() -> RedactortronService:
    global _service
    if _service is None:
        _service = RedactortronService()
    return _service


def _visible_entities(
    result: ScanResult,
    *,
    categories: Optional[Sequence[str]],
    view: str,
) -> List[DetectedEntity]:
    wanted = {c.strip().upper() for c in (categories or [])}
    out: List[DetectedEntity] = []
    for entity in result.all_entities:
        if entity.category not in wanted:
            continue
        if not view_allows_family(view, entity.family):
            continue
        out.append(entity)
    return out


def _entity_rows(
    entities: Sequence[DetectedEntity],
    *,
    show_meta: bool,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for entity in entities:
        family = {"transaction": "TX", "personal": "PI"}.get(entity.family, "OTHER")
        rows.append(
            [
                entity.entity_id if show_meta else "—",
                entity.page_index + 1,
                family,
                entity.category,
                entity.text,
                round(entity.score, 3) if show_meta else None,
            ]
        )
    return rows


def _matrix_headers(show_meta: bool) -> List[str]:
    # Schema stays fixed so Gradio Dataframe updates cleanly.
    _ = show_meta
    return ["ID", "Page", "Family", "Category", "Text", "Score"]


def _draw_cyan_highlights(
    frame_bgr: np.ndarray,
    entities: Sequence[DetectedEntity],
    padding: int = 3,
    theme_mode: Optional[str] = None,
) -> np.ndarray:
    """Overlay theme-colored boxes + translucent fill on selected regions."""
    stroke, fill = _highlight_colors(theme_mode)
    out = frame_bgr.copy()
    overlay = out.copy()
    height, width = out.shape[:2]
    for entity in entities:
        box = entity.box.clamp(width, height)
        x0 = max(0, box.x_min - padding)
        y0 = max(0, box.y_min - padding)
        x1 = min(width, box.x_max + padding)
        y1 = min(height, box.y_max + padding)
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, thickness=-1)
        cv2.rectangle(out, (x0, y0), (x1, y1), stroke, thickness=2)
        label = entity.entity_id or entity.category[:8]
        cv2.putText(
            out,
            label,
            (x0, max(14, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            stroke,
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)
    return out


def _preview_with_highlights(
    source: Path,
    entities: Sequence[DetectedEntity],
    max_pages: Optional[int] = None,
    theme_mode: Optional[str] = None,
) -> List[Any]:
    frames = get_service().core.load_pages(source)
    previews: List[Any] = []
    for idx, frame in enumerate(frames[:max_pages]):
        page_ents = [e for e in entities if e.page_index == idx]
        highlighted = _draw_cyan_highlights(
            frame, page_ents, theme_mode=theme_mode
        )
        previews.append(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    return previews


def _tx_map(state: Optional[Dict[str, Any]]) -> Dict[str, Transaction]:
    statement: Optional[ParsedStatement] = (state or {}).get("statement")
    if not statement:
        return {}
    return {t.entity_id: t for t in statement.transactions if t.entity_id}


def _item_label(
    entity: DetectedEntity,
    tx_map: Dict[str, Transaction],
    show_meta: bool,
) -> str:
    """Checklist label; statement transactions mirror the structure tree."""
    tx = tx_map.get(entity.entity_id)
    if tx is None:
        return entity.display_label(show_meta=show_meta)
    section = tx.section or "(no section)"
    first_line = tx.description.split("\n", 1)[0].strip() or "(no description)"
    if len(first_line) > 46:
        first_line = first_line[:43] + "…"
    amount = f" · {tx.amount_text}" if tx.amount_text else ""
    core = (
        f"p{tx.page} · {section} · #{tx.transaction_number} · "
        f"{tx.date} · {first_line}{amount}"
    )
    if show_meta:
        return f"{entity.entity_id} · {core}"
    return core


def _order_items(
    entities: Sequence[DetectedEntity],
    state: Optional[Dict[str, Any]],
    tx_map: Dict[str, Transaction],
) -> List[DetectedEntity]:
    """Group like the statement tree: page → section → transaction number,
    followed by the page's remaining entities in reading order."""
    statement: Optional[ParsedStatement] = (state or {}).get("statement")
    rank: Dict[str, int] = {}
    if statement:
        for index, tx in enumerate(statement.transactions):
            rank[tx.entity_id] = index

    def key(entity: DetectedEntity) -> Tuple[int, int, int, int]:
        tx = tx_map.get(entity.entity_id)
        if tx is not None:
            return (entity.page_index, 0, rank.get(entity.entity_id, 0), 0)
        return (entity.page_index, 1, 0, entity.box.y_min)

    return sorted(entities, key=key)


def _selection_bundle(
    state: Dict[str, Any],
    categories: Optional[Sequence[str]],
    view: str,
    show_meta: bool,
    selected_items: Optional[Sequence[str]] = None,
    theme_mode: Optional[str] = None,
) -> Tuple[Any, List[List[Any]], List[Any], List[str], List[str]]:
    """Recompute checklist, matrix, and themed preview from current filters.

    Returns:
        items_update, rows, previews, choices, selected_labels
    """
    result: ScanResult = state["result"]
    path = Path(state["path"])
    tx_map = _tx_map(state)
    visible = _order_items(
        _visible_entities(result, categories=categories, view=view), state, tx_map
    )
    choices = [_item_label(e, tx_map, show_meta) for e in visible]
    choice_ids = {ScanResult.entity_id_from_choice(c) for c in choices}
    id_to_label = {ScanResult.entity_id_from_choice(c): c for c in choices}

    if selected_items is None:
        value = list(choices)
    else:
        value = [
            id_to_label[ScanResult.entity_id_from_choice(c)]
            for c in selected_items
            if ScanResult.entity_id_from_choice(c) in choice_ids
        ]

    selected_ids = {ScanResult.entity_id_from_choice(c) for c in value}
    selected_entities = [e for e in visible if e.entity_id in selected_ids]

    try:
        previews = _preview_with_highlights(
            path, selected_entities, theme_mode=theme_mode
        )
    except RedactortronError as exc:
        logger.warning("Highlight preview failed: %s", exc.message)
        previews = []

    rows = _entity_rows(visible, show_meta=show_meta)
    items_update = gr.update(choices=choices, value=value, interactive=bool(choices))
    return items_update, rows, previews, choices, value


def _parse_labels(labels_text: Optional[str]) -> Optional[List[str]]:
    """Turn the editable prompt textarea into a GLiNER label list."""
    parts = [
        token.strip().lower()
        for chunk in str(labels_text or "").splitlines()
        for token in chunk.split(",")
        if token.strip()
    ]
    return parts or None


def scan_document(
    file_obj: Any,
    threshold: float,
    labels_text: str,
    view: str,
    show_meta: bool,
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, Any, str]:
    if file_obj is None:
        raise gr.Error("Upload a PDF or image first.")

    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    service = get_service()

    try:
        summary, result = service.scan(
            path,
            threshold=float(threshold),
            labels=_parse_labels(labels_text),
        )
    except RedactortronError as exc:
        exc.log(audience="internal")
        raise gr.Error(exc.format(audience="public")) from exc

    if not any(e.entity_id for e in result.all_entities):
        result.assign_entity_ids()

    state: Dict[str, Any] = {
        "path": str(path),
        "result": result,
        "threshold": float(threshold),
    }

    # Statement parser runs automatically so items are grouped like the tree.
    statement_tree = (
        "_No statement structure detected — press **Parse statement** to retry._"
    )
    statement_json_val = "[]"
    header_update = gr.update(choices=[], value=None)
    statement: Optional[ParsedStatement] = None
    try:
        statement = _ensure_statement(state)
    except Exception:
        logger.exception("Statement parsing failed; continuing without it")
    if statement and statement.transactions:
        statement_tree = statement.tree_markdown()
        statement_json_val = statement.to_json()
        header_update = gr.update(
            choices=list(state.get("headers", {}).keys()),
            value=None,
            interactive=True,
        )

    categories = result.categories()
    items_update, rows, previews, _, _ = _selection_bundle(
        state,
        categories=categories,
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=None,
        theme_mode=theme_mode,
    )

    dates = [e for e in result.all_entities if e.category in DATE_CATEGORIES]
    date_tree_val = (
        build_date_tree(result, dates)
        if dates
        else "_Press **Find every date** to build the per-page date tree._"
    )

    personal = sum(1 for e in result.all_entities if e.family == "personal")
    tx_count = len(statement.transactions) if statement else 0
    mark = "White/red" if _is_trinidad_theme(theme_mode) else "Cyan"
    summary_md = (
        f"<div class='rt-status'>⚡ <strong>{len(result.all_entities)}</strong> items · "
        f"<strong>{personal}</strong> personal info · "
        f"<strong>{tx_count}</strong> statement transactions · "
        f"<strong>{summary.page_count}</strong> page(s).<br/>"
        f"Items are grouped page → section → transaction. "
        f"Use the header picker to (un)select a whole section. "
        f"{mark} boxes mark the current selection in Visual feed.</div>"
    )

    category_update = gr.update(
        choices=categories,
        value=categories,
        interactive=bool(categories),
    )
    return (
        state,
        rows,
        category_update,
        items_update,
        previews,
        summary_md,
        statement_tree,
        statement_json_val,
        header_update,
        date_tree_val,
    )


def refresh_filters(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, str]:
    """Category / view / meta toggles → hide items + refresh matrix + preview."""
    if not state or "result" not in state:
        return gr.update(), [], [], "<div class='rt-status'>Scan a document first.</div>"

    items_update, rows, previews, _choices, selected_value = _selection_bundle(
        state,
        categories=categories or [],
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=items,
        theme_mode=theme_mode,
    )
    mark = "White/red" if _is_trinidad_theme(theme_mode) else "Cyan"
    note = (
        f"<p class='rt-highlight-note'>{mark} highlight active · "
        f"{len(selected_value)} selected · matrix view <code>{view}</code></p>"
    )
    return items_update, rows, previews, note


def refresh_preview_from_items(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[List[Any], str]:
    if not state or "result" not in state:
        return [], "<div class='rt-status'>Scan a document first.</div>"
    _, _, previews, _, _ = _selection_bundle(
        state,
        categories=categories or [],
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=items or [],
        theme_mode=theme_mode,
    )
    ids = [ScanResult.entity_id_from_choice(c) for c in (items or [])]
    mark = "White/red" if _is_trinidad_theme(theme_mode) else "Cyan"
    note = (
        f"<p class='rt-highlight-note'>{mark} highlight · "
        f"{len(ids)} item(s) selected</p>"
    )
    return previews, note


def select_visible_items(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
) -> Any:
    if not state or "result" not in state:
        return gr.update()
    result: ScanResult = state["result"]
    tx_map = _tx_map(state)
    visible = _order_items(
        _visible_entities(result, categories=categories, view=view or "All"),
        state,
        tx_map,
    )
    choices = [_item_label(e, tx_map, bool(show_meta)) for e in visible]
    return gr.update(value=choices)


def clear_items() -> Any:
    return gr.update(value=[])


def _apply_added_entities(
    state: Dict[str, Any],
    added: Sequence[DetectedEntity],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
    note_text: str,
    select_added: bool = True,
) -> Tuple[Any, Any, Any, Any, str]:
    """Refresh categories/items/matrix/preview after synthetic entities land."""
    result: ScanResult = state["result"]
    new_cats = list(categories or [])
    for entity in added:
        if entity.category not in new_cats:
            new_cats.append(entity.category)

    selected = list(items or [])
    if select_added:
        # Bare entity ids remap cleanly inside _selection_bundle.
        selected += [e.entity_id for e in added]
    items_update, rows, previews, _choices, value = _selection_bundle(
        state,
        categories=new_cats,
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=selected,
        theme_mode=theme_mode,
    )
    category_update = gr.update(
        choices=result.categories(),
        value=new_cats,
        interactive=True,
    )
    note = (
        f"<p class='rt-highlight-note'>{note_text} · "
        f"{len(value)} item(s) now selected</p>"
    )
    return category_update, items_update, rows, previews, note


def _entity_ids_for_matches(
    result: ScanResult, matches: Sequence[DetectedEntity]
) -> List[str]:
    """Ids of match entities, whether freshly added or already registered."""
    ids: List[str] = []
    for match in matches:
        if match.entity_id:
            ids.append(match.entity_id)
            continue
        page = result.pages[match.page_index]
        existing = next(
            (
                e
                for e in page.entities
                if e.box == match.box and e.category == match.category
            ),
            None,
        )
        if existing and existing.entity_id:
            ids.append(existing.entity_id)
    return ids


def _find_word_apply(
    state: Optional[Dict[str, Any]],
    term: str,
    select: bool,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    if not state or "result" not in state:
        raise gr.Error("Scan a document before searching for a word.")
    needle = (term or "").strip()
    if not needle:
        raise gr.Error("Type a word or phrase to find.")

    result: ScanResult = state["result"]
    matches = find_word_matches(result, needle)
    result.add_entities(matches)
    match_ids = set(_entity_ids_for_matches(result, matches))

    current = {ScanResult.entity_id_from_choice(c) for c in (items or [])}
    new_ids = (current | match_ids) if select else (current - match_ids)

    new_cats = list(categories or [])
    if select:
        for match in matches:
            if match.category not in new_cats:
                new_cats.append(match.category)

    items_update, rows, previews, _choices, value = _selection_bundle(
        state,
        categories=new_cats,
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=sorted(new_ids),
        theme_mode=theme_mode,
    )
    cat_update = gr.update(
        choices=result.categories(), value=new_cats, interactive=True
    )
    if not matches:
        head = f"No matches for “{needle}”"
    else:
        verb = "Selected" if select else "Deselected"
        head = f"{verb} {len(match_ids)} match(es) for “{needle}”"
    note = (
        f"<p class='rt-highlight-note'>{head} · "
        f"{len(value)} item(s) now selected</p>"
    )
    return cat_update, items_update, rows, previews, note


def find_word_select_handler(
    state: Optional[Dict[str, Any]],
    term: str,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    return _find_word_apply(
        state, term, True, categories, view, show_meta, items, theme_mode
    )


def find_word_deselect_handler(
    state: Optional[Dict[str, Any]],
    term: str,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    return _find_word_apply(
        state, term, False, categories, view, show_meta, items, theme_mode
    )


def add_lines_handler(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    if not state or "result" not in state:
        raise gr.Error("Scan a document first.")
    result: ScanResult = state["result"]
    ids = [ScanResult.entity_id_from_choice(c) for c in (items or [])]
    targets = result.entities_for_ids([i for i in ids if i])
    if not targets:
        raise gr.Error(
            "Select at least one item first (e.g. a date or transaction number), "
            "then this grabs its full line."
        )
    lines = line_entities_for(result, targets)
    added = result.add_entities(lines)
    note = (
        f"Captured {len(lines)} full line(s) connected to {len(targets)} "
        f"selected item(s) — added as LINE items."
    )
    return _apply_added_entities(
        state, added, categories, view, show_meta, items, theme_mode, note
    )


def find_dates_handler(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    if not state or "result" not in state:
        raise gr.Error("Scan a document first.")
    result: ScanResult = state["result"]
    found = find_all_dates(result)
    added = result.add_entities(found)
    note = (
        f"Date sweep complete: {len(found)} date(s) located "
        f"({len(added)} new, added as DATE FOUND items)."
    )
    return _apply_added_entities(
        state, added, categories, view, show_meta, items, theme_mode, note
    )


def _build_header_index(statement: ParsedStatement) -> Dict[str, List[str]]:
    """Header choice → transaction entity ids (whole-document and per-page)."""
    per_section: Dict[str, List[str]] = {}
    per_page: Dict[Tuple[int, str], List[str]] = {}
    for tx in statement.transactions:
        if not tx.entity_id:
            continue
        section = tx.section or "(no section)"
        per_section.setdefault(section, []).append(tx.entity_id)
        per_page.setdefault((tx.page, section), []).append(tx.entity_id)

    index: Dict[str, List[str]] = {}
    for section, ids in per_section.items():
        index[f"{section} — all pages ({len(ids)})"] = ids
    for (page, section), ids in per_page.items():
        index[f"Page {page} — {section} ({len(ids)})"] = ids
    return index


def _ensure_statement(state: Dict[str, Any]) -> ParsedStatement:
    """Parse the statement once and register each transaction as an item."""
    if "statement" in state:
        return state["statement"]

    result: ScanResult = state["result"]
    statement = parse_statement(result)
    for tx in statement.transactions:
        entity = DetectedEntity(
            text=tx.description or f"{tx.date} {tx.amount_text}".strip(),
            label="transaction",
            score=1.0,
            page_index=tx.page_index,
            box=tx.box,
        )
        added = result.add_entities([entity])
        if added:
            tx.entity_id = added[0].entity_id
        else:
            page = result.pages[tx.page_index]
            match = next(
                (
                    e
                    for e in page.entities
                    if e.box == tx.box and e.category == "TRANSACTION"
                ),
                None,
            )
            tx.entity_id = match.entity_id if match else ""
    state["statement"] = statement
    state["headers"] = _build_header_index(statement)
    return statement


def parse_statement_handler(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str, str, str]:
    if not state or "result" not in state:
        raise gr.Error("Scan a document before parsing the statement.")

    result: ScanResult = state["result"]
    before = {e.entity_id for e in result.all_entities}
    statement = _ensure_statement(state)
    added = [e for e in result.all_entities if e.entity_id not in before]

    sections = {(t.page, t.section) for t in statement.transactions}
    pages = {t.page for t in statement.transactions}
    note = (
        f"Parsed {len(statement.transactions)} transaction(s) in "
        f"{len(sections)} section(s) across {len(pages)} page(s) — "
        f"each is now a selectable TRANSACTION item."
    )
    cat_update, items_update, rows, previews, note_html = _apply_added_entities(
        state, added, categories, view, show_meta, items, theme_mode, note,
        select_added=False,
    )
    header_update = gr.update(
        choices=list(state.get("headers", {}).keys()),
        value=None,
        interactive=True,
    )
    return (
        cat_update,
        items_update,
        rows,
        previews,
        note_html,
        statement.tree_markdown(),
        statement.to_json(),
        header_update,
    )


def _find_target_apply(
    state: Optional[Dict[str, Any]],
    target: str,
    select: bool,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str, str]:
    if not state or "result" not in state:
        raise gr.Error("Scan a document first.")
    needle = (target or "").strip()
    if not needle:
        raise gr.Error("Type the target text to search transactions for.")

    result: ScanResult = state["result"]
    statement = _ensure_statement(state)

    matches = statement.search(needle)
    match_ids = {t.entity_id for t in matches if t.entity_id}

    current = {ScanResult.entity_id_from_choice(c) for c in (items or [])}
    new_ids = (current | match_ids) if select else (current - match_ids)

    new_cats = list(categories or [])
    if select and match_ids and "TRANSACTION" not in new_cats:
        new_cats.append("TRANSACTION")

    items_update, rows, previews, _choices, value = _selection_bundle(
        state,
        categories=new_cats,
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=sorted(new_ids),
        theme_mode=theme_mode,
    )
    cat_update = gr.update(
        choices=result.categories(), value=new_cats, interactive=True
    )
    verb = "selected" if select else "deselected"
    note = (
        f"<p class='rt-highlight-note'>Target text matched "
        f"{len(matches)} transaction(s) ({verb}) · "
        f"{len(value)} item(s) now selected</p>"
    )
    return (
        cat_update,
        items_update,
        rows,
        previews,
        note,
        statement.to_json(matches),
    )


def find_target_select_handler(
    state: Optional[Dict[str, Any]],
    target: str,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str, str]:
    return _find_target_apply(
        state, target, True, categories, view, show_meta, items, theme_mode
    )


def find_target_deselect_handler(
    state: Optional[Dict[str, Any]],
    target: str,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str, str]:
    return _find_target_apply(
        state, target, False, categories, view, show_meta, items, theme_mode
    )


def _apply_header_selection(
    state: Optional[Dict[str, Any]],
    header: Optional[str],
    select: bool,
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    if not state or "result" not in state:
        raise gr.Error("Scan a document first.")
    if "statement" not in state:
        _ensure_statement(state)
    headers: Dict[str, List[str]] = state.get("headers", {})
    ids = set(headers.get(header or "", []))
    if not ids:
        raise gr.Error("Pick a header (page · section) from the dropdown first.")

    result: ScanResult = state["result"]
    current = {
        ScanResult.entity_id_from_choice(c) for c in (items or [])
    }
    new_ids = (current | ids) if select else (current - ids)

    new_cats = list(categories or [])
    if select and "TRANSACTION" not in new_cats:
        new_cats.append("TRANSACTION")

    items_update, rows, previews, _choices, value = _selection_bundle(
        state,
        categories=new_cats,
        view=view or "All",
        show_meta=bool(show_meta),
        selected_items=sorted(new_ids),
        theme_mode=theme_mode,
    )
    cat_update = gr.update(
        choices=result.categories(), value=new_cats, interactive=True
    )
    verb = "Selected" if select else "Unselected"
    note = (
        f"<p class='rt-highlight-note'>{verb} {len(ids)} transaction(s) under "
        f"“{header}” · {len(value)} item(s) now selected</p>"
    )
    return cat_update, items_update, rows, previews, note


def select_header_handler(
    state: Optional[Dict[str, Any]],
    header: Optional[str],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    return _apply_header_selection(
        state, header, True, categories, view, show_meta, items, theme_mode
    )


def unselect_header_handler(
    state: Optional[Dict[str, Any]],
    header: Optional[str],
    categories: Optional[List[str]],
    view: str,
    show_meta: bool,
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Any, Any, Any, Any, str]:
    return _apply_header_selection(
        state, header, False, categories, view, show_meta, items, theme_mode
    )


def build_tree_handler(state: Optional[Dict[str, Any]]) -> str:
    if not state or "result" not in state:
        return "_Scan a document first, then build the date tree._"
    result: ScanResult = state["result"]
    dates = [e for e in result.all_entities if e.category in DATE_CATEGORIES]
    if not dates:
        dates = result.add_entities(find_all_dates(result))
    if not dates:
        return "_No dates detected anywhere in this document._"
    return build_date_tree(result, dates)


def redact_document(
    state: Optional[Dict[str, Any]],
    categories: Optional[List[str]],
    items: Optional[List[str]],
    theme_mode: str,
) -> Tuple[Optional[str], List[Any], str]:
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
        selected_entities = result.entities_for_ids(entity_ids) if entity_ids else (
            result.entities_for_categories(list(categories or []))
        )
        previews = _preview_with_highlights(
            path, selected_entities, theme_mode=theme_mode
        )
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
    theme = _build_theme()

    with gr.Blocks(
        title=f"Redactortron v{__version__}",
    ) as demo:
        gr.HTML(
            f"""
            <div class="rt-hero">
              <p class="rt-kicker">Local · Offline · AI Redaction</p>
              <h1 class="rt-title">REDACTORTRON v{__version__}</h1>
              <p class="rt-sub">
                Personal information and transactions in their own sections,
                a word finder, full-line capture, and a per-page date tree —
                running entirely on your machine.
              </p>
              <div class="rt-chiprow">
                <span class="rt-chip">Personal information</span>
                <span class="rt-chip alt">Transactions</span>
                <span class="rt-chip cyan">Date tree</span>
              </div>
            </div>
            """
        )

        state = gr.State(None)
        theme_css = gr.HTML(render_theme_css(THEME_CYBER_LABEL))

        with gr.Accordion("Settings", open=False):
            theme_mode = gr.Radio(
                label="GUI theme",
                choices=[THEME_CYBER_LABEL, THEME_TRINIDAD_LABEL],
                value=THEME_CYBER_LABEL,
            )
            show_meta = gr.Checkbox(
                label="Show ID & score in matrix / item list",
                value=True,
            )

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
                with gr.Accordion("01 · Detection prompt & parameters", open=False):
                    labels_box = gr.Textbox(
                        label="Category prompt (edit / add labels, comma-separated)",
                        value=", ".join(DEFAULT_LABELS),
                        lines=4,
                        placeholder="person, address, account number, merchant, …",
                    )
                    threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.4,
                        step=0.05,
                        label="Detection threshold",
                    )
                scan_btn = gr.Button("02 · Scan document", variant="primary")

                matrix_view = gr.Radio(
                    label="Section (Personal information / Transactions)",
                    choices=list(MATRIX_VIEWS),
                    value="All",
                )
                categories = gr.CheckboxGroup(
                    label="03 · Categories (unchecked = hidden below)",
                    choices=[],
                    interactive=False,
                )

                with gr.Accordion("Find & analyze", open=True):
                    word_box = gr.Textbox(
                        label="Find a specific word or phrase",
                        placeholder="e.g. Walmart",
                    )
                    with gr.Row():
                        find_select_btn = gr.Button("Find + select")
                        find_deselect_btn = gr.Button("Find + deselect")
                    line_btn = gr.Button(
                        "Capture full line(s) of selected items (date / txn #)"
                    )
                    with gr.Row():
                        dates_btn = gr.Button("Find every date")
                        tree_btn = gr.Button("Build date tree (per page)")

                with gr.Accordion("Statement parser (headers → lines)", open=True):
                    parse_btn = gr.Button(
                        "Re-parse statement into sections & transactions"
                    )
                    header_dd = gr.Dropdown(
                        label="Header (section) — pick, then select/unselect all",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    with gr.Row():
                        select_header_btn = gr.Button("Select all in header")
                        unselect_header_btn = gr.Button("Unselect all in header")
                    target_box = gr.Textbox(
                        label="Target text (search all transactions)",
                        placeholder="e.g. Real Time Payment Credit Recd From…",
                    )
                    with gr.Row():
                        target_select_btn = gr.Button("Search + select")
                        target_deselect_btn = gr.Button("Search + deselect")

                with gr.Row():
                    select_all_btn = gr.Button("Select visible items")
                    clear_btn = gr.Button("Clear items")
                items = gr.CheckboxGroup(
                    label="04 · Individual items (grouped page → section → transaction)",
                    choices=[],
                    interactive=False,
                )
                redact_btn = gr.Button("05 · Blur & export", variant="primary")
                status = gr.Markdown(
                    "<div class='rt-status'>Awaiting uplink — upload a file, then scan.</div>"
                )

            with gr.Column(scale=2):
                highlight_note = gr.HTML(
                    "<p class='rt-highlight-note'>Selection highlight appears after scan.</p>"
                )
                entities = gr.Dataframe(
                    headers=_matrix_headers(True),
                    label="Itemized entity matrix",
                    interactive=False,
                    wrap=True,
                )
                with gr.Accordion(
                    "Statement structure (Bank → Company → Year → Month → "
                    "Page → Section → Transactions)",
                    open=True,
                ):
                    statement_tree_md = gr.Markdown(
                        "_Scan, then press **Parse statement** to build the "
                        "hierarchy and one JSON record per transaction._"
                    )
                    statement_json = gr.Code(
                        label="Transaction records (JSON)",
                        language="json",
                        value="[]",
                    )
                with gr.Accordion("Date tree (per page)", open=True):
                    date_tree_md = gr.Markdown(
                        "_Scan a document, then press **Build date tree** to map "
                        "every date and the information connected to it, page by page._"
                    )
                preview = gr.Gallery(
                    label="Visual feed (highlighted = selected)",
                    columns=2,
                    height=480,
                    object_fit="contain",
                )
                download = gr.File(label="Redacted package")

        def _on_theme_change(
            mode: str,
            state_val: Optional[Dict[str, Any]],
            cats: Optional[List[str]],
            view: str,
            meta: bool,
            item_vals: Optional[List[str]],
        ):
            css_html = render_theme_css(mode)
            if not state_val or "result" not in state_val:
                return css_html, gr.update(), gr.update()
            previews, note = refresh_preview_from_items(
                state_val, cats, view, meta, item_vals, mode
            )
            return css_html, previews, note

        theme_mode.change(
            fn=_on_theme_change,
            inputs=[theme_mode, state, categories, matrix_view, show_meta, items],
            outputs=[theme_css, preview, highlight_note],
        )

        scan_btn.click(
            fn=scan_document,
            inputs=[file_in, threshold, labels_box, matrix_view, show_meta, theme_mode],
            outputs=[
                state,
                entities,
                categories,
                items,
                preview,
                status,
                statement_tree_md,
                statement_json,
                header_dd,
                date_tree_md,
            ],
        )

        for trigger in (categories, matrix_view, show_meta):
            trigger.change(
                fn=refresh_filters,
                inputs=[state, categories, matrix_view, show_meta, items, theme_mode],
                outputs=[items, entities, preview, highlight_note],
            )

        items.change(
            fn=refresh_preview_from_items,
            inputs=[state, categories, matrix_view, show_meta, items, theme_mode],
            outputs=[preview, highlight_note],
        )

        select_all_btn.click(
            fn=select_visible_items,
            inputs=[state, categories, matrix_view, show_meta],
            outputs=[items],
        ).then(
            fn=refresh_preview_from_items,
            inputs=[state, categories, matrix_view, show_meta, items, theme_mode],
            outputs=[preview, highlight_note],
        )

        clear_btn.click(fn=clear_items, outputs=[items]).then(
            fn=refresh_preview_from_items,
            inputs=[state, categories, matrix_view, show_meta, items, theme_mode],
            outputs=[preview, highlight_note],
        )

        tool_inputs = [state, categories, matrix_view, show_meta, items, theme_mode]
        tool_outputs = [categories, items, entities, preview, highlight_note]

        word_inputs = [
            state, word_box, categories, matrix_view, show_meta, items, theme_mode,
        ]
        find_select_btn.click(
            fn=find_word_select_handler,
            inputs=word_inputs,
            outputs=tool_outputs,
        )
        find_deselect_btn.click(
            fn=find_word_deselect_handler,
            inputs=word_inputs,
            outputs=tool_outputs,
        )
        word_box.submit(
            fn=find_word_select_handler,
            inputs=word_inputs,
            outputs=tool_outputs,
        )
        line_btn.click(
            fn=add_lines_handler,
            inputs=tool_inputs,
            outputs=tool_outputs,
        )
        dates_btn.click(
            fn=find_dates_handler,
            inputs=tool_inputs,
            outputs=tool_outputs,
        ).then(
            fn=build_tree_handler,
            inputs=[state],
            outputs=[date_tree_md],
        )
        tree_btn.click(
            fn=build_tree_handler,
            inputs=[state],
            outputs=[date_tree_md],
        )

        statement_outputs = tool_outputs + [statement_tree_md, statement_json, header_dd]
        parse_btn.click(
            fn=parse_statement_handler,
            inputs=tool_inputs,
            outputs=statement_outputs,
        )
        header_inputs = [
            state, header_dd, categories, matrix_view, show_meta, items, theme_mode,
        ]
        select_header_btn.click(
            fn=select_header_handler,
            inputs=header_inputs,
            outputs=tool_outputs,
        )
        unselect_header_btn.click(
            fn=unselect_header_handler,
            inputs=header_inputs,
            outputs=tool_outputs,
        )
        target_inputs = [
            state, target_box, categories, matrix_view, show_meta, items, theme_mode,
        ]
        target_select_btn.click(
            fn=find_target_select_handler,
            inputs=target_inputs,
            outputs=tool_outputs + [statement_json],
        )
        target_deselect_btn.click(
            fn=find_target_deselect_handler,
            inputs=target_inputs,
            outputs=tool_outputs + [statement_json],
        )
        target_box.submit(
            fn=find_target_select_handler,
            inputs=target_inputs,
            outputs=tool_outputs + [statement_json],
        )

        redact_btn.click(
            fn=redact_document,
            inputs=[state, categories, items, theme_mode],
            outputs=[download, preview, status],
        )

        gr.HTML(
            "<p class='rt-foot'>Section switch: Personal information vs Transactions. "
            "Word matches, captured lines, and found dates appear as selectable items "
            "in 04 and blur on export.</p>"
        )

    return demo, theme


def _find_free_port(host: str, start: int, span: int = 20) -> int:
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    launch()


if __name__ == "__main__":
    main()
