"""Data models for entities and page geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Absolute pixel coordinates for a detected region.

    Attributes:
        x_min: Left edge (inclusive).
        y_min: Top edge (inclusive).
        x_max: Right edge (exclusive).
        y_max: Bottom edge (exclusive).
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """Return a copy clipped to image bounds."""
        return BoundingBox(
            x_min=max(0, min(self.x_min, width)),
            y_min=max(0, min(self.y_min, height)),
            x_max=max(0, min(self.x_max, width)),
            y_max=max(0, min(self.y_max, height)),
        )


@dataclass
class WordSpan:
    """A single OCR word with its page-relative geometry."""

    text: str
    box: BoundingBox
    page_index: int
    start_char: int
    end_char: int


from redactortron.taxonomy import category_family


@dataclass
class DetectedEntity:
    """A named entity located on a document page."""

    text: str
    label: str
    score: float
    page_index: int
    box: BoundingBox
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    entity_id: str = ""

    @property
    def category(self) -> str:
        """Normalized category key used for interactive selection."""
        return self.label.strip().upper()

    @property
    def family(self) -> str:
        """``account``, ``transaction``, or ``other``."""
        return category_family(self.category)

    def display_label(self, *, show_meta: bool = True) -> str:
        """Human-readable line for checklists (transactions / entities)."""
        snippet = self.text.replace("\n", " ").strip()
        if len(snippet) > 64:
            snippet = snippet[:61] + "..."
        family_tag = "TX" if self.family == "transaction" else (
            "ACCT" if self.family == "account" else "OTHER"
        )
        if show_meta:
            eid = self.entity_id or "?"
            return (
                f"{eid} · p{self.page_index + 1} · [{family_tag}] {self.category} · "
                f"{snippet} ({self.score:.2f})"
            )
        return f"p{self.page_index + 1} · [{family_tag}] {self.category} · {snippet}"


@dataclass
class PageResult:
    """OCR + entity results for one rendered page."""

    page_index: int
    width: int
    height: int
    full_text: str
    words: List[WordSpan] = field(default_factory=list)
    entities: List[DetectedEntity] = field(default_factory=list)


@dataclass
class ScanResult:
    """Aggregate scan output across all pages."""

    source_path: str
    pages: List[PageResult] = field(default_factory=list)

    @property
    def all_entities(self) -> List[DetectedEntity]:
        entities: List[DetectedEntity] = []
        for page in self.pages:
            entities.extend(page.entities)
        return entities

    def assign_entity_ids(self) -> None:
        """Stamp stable ``E0001``-style ids onto every detected entity."""
        for index, entity in enumerate(self.all_entities, start=1):
            entity.entity_id = f"E{index:04d}"

    def categories(self) -> List[str]:
        """Sorted unique entity categories found in the document."""
        return sorted({entity.category for entity in self.all_entities})

    def entities_for_categories(self, categories: List[str]) -> List[DetectedEntity]:
        selected = {c.strip().upper() for c in categories}
        return [e for e in self.all_entities if e.category in selected]

    def entities_for_ids(self, entity_ids: List[str]) -> List[DetectedEntity]:
        wanted = {i.strip().upper() for i in entity_ids if str(i).strip()}
        return [e for e in self.all_entities if e.entity_id.upper() in wanted]

    def item_choices(
        self,
        *,
        categories: Optional[List[str]] = None,
        view: str = "All",
        show_meta: bool = True,
    ) -> List[str]:
        """Checklist labels, optionally filtered by active categories + matrix view."""
        from redactortron.taxonomy import view_allows_family

        wanted = None
        if categories is not None:
            wanted = {c.strip().upper() for c in categories}
        labels: List[str] = []
        for entity in self.all_entities:
            if wanted is not None and entity.category not in wanted:
                continue
            if not view_allows_family(view, entity.family):
                continue
            labels.append(entity.display_label(show_meta=show_meta))
        return labels

    @staticmethod
    def entity_id_from_choice(choice: str) -> str:
        """Extract ``E0001`` from a checklist display label."""
        token = str(choice).strip().split("·", 1)[0].strip().upper()
        return token
