"""Document analysis helpers: word search, date discovery, lines, date trees."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from redactortron.models import (
    BoundingBox,
    DetectedEntity,
    PageResult,
    ScanResult,
    WordSpan,
)

WORD_MATCH_LABEL = "word match"
LINE_LABEL = "line"
DATE_FOUND_LABEL = "date found"

# Categories treated as dates when building the tree.
DATE_CATEGORIES = {"DATE", "TRANSACTION DATE", "DATE FOUND"}

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

DATE_REGEXES = [
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$"),
    re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}$"),
    re.compile(
        rf"^(?:{_MONTHS})\.?,?\s+\d{{1,2}}(?:st|nd|rd|th)?,?(?:\s+\d{{2,4}})?$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?(?:\s+\d{{2,4}})?$",
        re.IGNORECASE,
    ),
]


def _union_box(boxes: Sequence[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x_min=min(b.x_min for b in boxes),
        y_min=min(b.y_min for b in boxes),
        x_max=max(b.x_max for b in boxes),
        y_max=max(b.y_max for b in boxes),
    )


def _words_by_line(page: PageResult) -> Dict[int, List[Tuple[int, WordSpan]]]:
    grouped: Dict[int, List[Tuple[int, WordSpan]]] = {}
    for index, word in enumerate(page.words):
        grouped.setdefault(word.line_index, []).append((index, word))
    return grouped


def entity_line_indices(page: PageResult, entity: DetectedEntity) -> List[int]:
    """Line numbers an entity sits on (char offsets first, geometry fallback)."""
    if entity.start_char is not None and entity.end_char is not None:
        hits = {
            w.line_index
            for w in page.words
            if w.line_index >= 0
            and w.start_char < entity.end_char
            and w.end_char > entity.start_char
        }
        if hits:
            return sorted(hits)

    box = entity.box
    hits = {
        w.line_index
        for w in page.words
        if w.line_index >= 0 and w.box.y_min < box.y_max and w.box.y_max > box.y_min
    }
    return sorted(hits)


def find_word_matches(result: ScanResult, term: str) -> List[DetectedEntity]:
    """Locate a word or phrase in the OCR text (case-insensitive)."""
    needle = " ".join(term.strip().lower().split())
    if not needle:
        return []
    tokens = needle.split()

    matches: List[DetectedEntity] = []
    for page in result.pages:
        if len(tokens) == 1:
            for word in page.words:
                if needle in word.text.lower():
                    matches.append(
                        DetectedEntity(
                            text=word.text,
                            label=WORD_MATCH_LABEL,
                            score=1.0,
                            page_index=page.page_index,
                            box=word.box,
                        )
                    )
            continue

        # Phrases must appear as consecutive words on the same line.
        for line_words in _words_by_line(page).values():
            words = [w for _, w in line_words]
            for start in range(len(words) - len(tokens) + 1):
                window = words[start : start + len(tokens)]
                joined = " ".join(w.text.lower() for w in window)
                if needle in joined:
                    matches.append(
                        DetectedEntity(
                            text=" ".join(w.text for w in window),
                            label=WORD_MATCH_LABEL,
                            score=1.0,
                            page_index=page.page_index,
                            box=_union_box([w.box for w in window]),
                        )
                    )
    return matches


def line_entities_for(
    result: ScanResult,
    entities: Sequence[DetectedEntity],
) -> List[DetectedEntity]:
    """Full-line entities covering every line the given entities touch.

    Selecting these redacts the whole row of information connected to,
    e.g., a date or transaction number.
    """
    lines: List[DetectedEntity] = []
    seen: set = set()
    for entity in entities:
        page = result.pages[entity.page_index]
        for line_index in entity_line_indices(page, entity):
            key = (entity.page_index, line_index)
            if key in seen:
                continue
            seen.add(key)
            box = page.line_box(line_index)
            if box is None:
                continue
            lines.append(
                DetectedEntity(
                    text=page.line_text(line_index),
                    label=LINE_LABEL,
                    score=1.0,
                    page_index=entity.page_index,
                    box=box,
                )
            )
    return lines


def find_all_dates(result: ScanResult) -> List[DetectedEntity]:
    """Carefully scan every page for date-shaped strings (1–4 word windows)."""
    found: List[DetectedEntity] = []
    for page in result.pages:
        for line_words in _words_by_line(page).values():
            used: set = set()
            # Prefer longer windows ("December 14, 2022") over fragments.
            for size in (4, 3, 2, 1):
                for start in range(len(line_words) - size + 1):
                    indices = [line_words[start + k][0] for k in range(size)]
                    if any(i in used for i in indices):
                        continue
                    words = [line_words[start + k][1] for k in range(size)]
                    text = " ".join(w.text for w in words)
                    if any(rx.match(text) for rx in DATE_REGEXES):
                        found.append(
                            DetectedEntity(
                                text=text,
                                label=DATE_FOUND_LABEL,
                                score=1.0,
                                page_index=page.page_index,
                                box=_union_box([w.box for w in words]),
                            )
                        )
                        used.update(indices)
    return found


def build_date_tree(
    result: ScanResult,
    dates: Sequence[DetectedEntity],
) -> str:
    """Markdown tree: per page, each date with the line and entities tied to it."""
    if not result.pages:
        return "_No pages scanned._"

    sections: List[str] = []
    for page in result.pages:
        sections.append(f"### Page {page.page_index + 1}")
        page_dates = sorted(
            (d for d in dates if d.page_index == page.page_index),
            key=lambda e: (e.box.y_min, e.box.x_min),
        )
        if not page_dates:
            sections.append("_No dates found on this page._")
            continue

        for date_entity in page_dates:
            eid = date_entity.entity_id or "—"
            sections.append(f"- **{date_entity.text}** (`{eid}` · {date_entity.category})")
            for line_index in entity_line_indices(page, date_entity):
                line_text = page.line_text(line_index).strip()
                if line_text:
                    sections.append(f"  - line: `{line_text}`")
                linked = [
                    e
                    for e in page.entities
                    if e is not date_entity
                    and line_index in entity_line_indices(page, e)
                ]
                for entity in linked:
                    link_id = entity.entity_id or "—"
                    sections.append(
                        f"    - `{link_id}` · {entity.category} · {entity.text}"
                    )
    return "\n".join(sections)
