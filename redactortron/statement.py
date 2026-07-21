"""Intelligent bank-statement parser.

Converts OCR output into a structured transaction dataset:

* every page is parsed independently;
* section headers are detected ("Deposits and Additions", "Checks Paid", …);
* a transaction begins whenever a line starts with a date (MM/DD) and owns
  every continuation line until the next date, section, or page end;
* the amount always belongs to the transaction that began with the date;
* each transaction becomes one complete, independently selectable record.

Hierarchy: Bank > Company > Year > Month > Page > Section > Transaction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from redactortron.models import BoundingBox, PageResult, ScanResult, WordSpan

# ---------------------------------------------------------------------------
# Regexes / vocabularies
# ---------------------------------------------------------------------------

DATE_START_RE = re.compile(r"^(\d{1,2}/\d{1,2})(?:/\d{2,4})?\b")
AMOUNT_RE = re.compile(r"-?\$?\d[\d,]*\.\d{2}")

SECTION_KEYWORDS: Tuple[str, ...] = (
    "DEPOSITS AND ADDITIONS",
    "DEPOSITS AND CREDITS",
    "DEPOSITS AND OTHER ADDITIONS",
    "CHECKS PAID",
    "ATM & DEBIT CARD WITHDRAWALS",
    "ATM AND DEBIT CARD WITHDRAWALS",
    "ELECTRONIC WITHDRAWALS",
    "OTHER WITHDRAWALS",
    "WITHDRAWALS AND DEBITS",
    "FEES",
    "SERVICE CHARGES",
    "DAILY ENDING BALANCE",
    "CHECKING SUMMARY",
    "SAVINGS SUMMARY",
)

BANK_NAMES: Dict[str, str] = {
    "JPMORGAN": "Chase Bank",
    "CHASE": "Chase Bank",
    "BANK OF AMERICA": "Bank of America",
    "WELLS FARGO": "Wells Fargo",
    "CITIBANK": "Citibank",
    "CAPITAL ONE": "Capital One",
    "PNC": "PNC Bank",
    "TD BANK": "TD Bank",
    "U.S. BANK": "U.S. Bank",
    "US BANK": "U.S. Bank",
    "NAVY FEDERAL": "Navy Federal Credit Union",
    "TRUIST": "Truist",
    "REGIONS": "Regions Bank",
}

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_MONTH_RE = re.compile(
    r"(" + "|".join(MONTH_NAMES) + r")\s+\d{1,2},?\s+(\d{4})",
    re.IGNORECASE,
)
_NUMERIC_PERIOD_RE = re.compile(r"(\d{1,2})/\d{1,2}/(\d{2,4})")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One visual line on a page, rebuilt from OCR word geometry."""

    text: str
    box: BoundingBox
    words: List[WordSpan] = field(default_factory=list)


@dataclass
class Transaction:
    """One complete statement record (date line + all continuation lines)."""

    page: int  # 1-based page number
    page_index: int
    section: str
    transaction_number: int
    date: str
    description: str
    amount: Optional[float]
    amount_text: str
    box: BoundingBox
    entity_id: str = ""

    def to_record(self) -> Dict[str, object]:
        return {
            "page": self.page,
            "section": self.section,
            "transaction_number": self.transaction_number,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "entity_id": self.entity_id,
        }

    def searchable_text(self) -> str:
        return " ".join(
            f"{self.date} {self.description} {self.amount_text}".lower().split()
        )


@dataclass
class ParsedStatement:
    bank: str
    company: str
    year: str
    month: str
    transactions: List[Transaction] = field(default_factory=list)

    def search(self, target: str) -> List[Transaction]:
        """Every transaction whose full record (incl. continuation lines)
        contains the target text."""
        needle = " ".join(str(target or "").lower().split())
        if not needle:
            return []
        return [t for t in self.transactions if needle in t.searchable_text()]

    def to_json(self, transactions: Optional[List[Transaction]] = None) -> str:
        chosen = self.transactions if transactions is None else transactions
        return json.dumps([t.to_record() for t in chosen], indent=2)

    def tree_markdown(self) -> str:
        lines = [
            f"### {self.bank}",
            f"- **{self.company}**",
            f"  - **{self.year}**",
            f"    - **{self.month}**",
        ]
        by_page: Dict[int, List[Transaction]] = {}
        for tx in self.transactions:
            by_page.setdefault(tx.page, []).append(tx)
        if not by_page:
            lines.append("      - _No transactions detected._")
            return "\n".join(lines)

        for page in sorted(by_page):
            lines.append(f"      - **Page {page}**")
            by_section: Dict[str, List[Transaction]] = {}
            for tx in by_page[page]:
                by_section.setdefault(tx.section, []).append(tx)
            for section, txs in by_section.items():
                title = section or "(no section)"
                lines.append(
                    f"        - **{title}** ({len(txs)} transaction(s))"
                )
                for tx in txs:
                    snippet = tx.description.replace("\n", " ⏎ ").strip()
                    if len(snippet) > 70:
                        snippet = snippet[:67] + "…"
                    amount = tx.amount_text or "—"
                    eid = f" `{tx.entity_id}`" if tx.entity_id else ""
                    lines.append(
                        f"          - {tx.transaction_number} · {tx.date} · "
                        f"{snippet or '(no description)'} · {amount}{eid}"
                    )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Row reconstruction
# ---------------------------------------------------------------------------


def build_rows(page: PageResult) -> List[Row]:
    """Cluster OCR words into visual lines by vertical overlap.

    docTR line indices split a visual row across blocks (description column
    vs amount column), so geometry is the reliable grouping key.
    """
    words = sorted(page.words, key=lambda w: (w.box.y_min + w.box.y_max) / 2)
    clusters: List[List[WordSpan]] = []
    bands: List[Tuple[int, int]] = []

    for word in words:
        if clusters:
            y0, y1 = bands[-1]
            overlap = min(word.box.y_max, y1) - max(word.box.y_min, y0)
            smallest = max(1, min(word.box.y_max - word.box.y_min, y1 - y0))
            if overlap > 0.5 * smallest:
                clusters[-1].append(word)
                bands[-1] = (min(y0, word.box.y_min), max(y1, word.box.y_max))
                continue
        clusters.append([word])
        bands.append((word.box.y_min, word.box.y_max))

    rows: List[Row] = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda w: w.box.x_min)
        box = BoundingBox(
            x_min=min(w.box.x_min for w in ordered),
            y_min=min(w.box.y_min for w in ordered),
            x_max=max(w.box.x_max for w in ordered),
            y_max=max(w.box.y_max for w in ordered),
        ).clamp(page.width, page.height)
        rows.append(
            Row(text=" ".join(w.text for w in ordered), box=box, words=ordered)
        )
    return rows


# ---------------------------------------------------------------------------
# Section / amount / date helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(text.upper().split())


def detect_section(row_text: str) -> Optional[str]:
    up = _normalize(row_text)
    if not up or up.startswith("TOTAL") or DATE_START_RE.match(up):
        return None
    for keyword in SECTION_KEYWORDS:
        if keyword in up and len(up) <= len(keyword) + 24:
            return keyword.title().replace("Atm", "ATM").replace("& ", "& ")
    return None


def parse_money(token: str) -> Optional[float]:
    cleaned = token.replace("$", "").replace(",", "").strip()
    negative = cleaned.endswith("-")
    cleaned = cleaned.rstrip("-")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative and value > 0 else value


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------


class _TxBuilder:
    def __init__(
        self,
        page_index: int,
        section: str,
        number: int,
        date: str,
        first_line: str,
        amount_text: str,
        box: BoundingBox,
    ) -> None:
        self.page_index = page_index
        self.section = section
        self.number = number
        self.date = date
        self.lines = [first_line] if first_line else []
        self.amount_text = amount_text
        self.boxes = [box]

    def add_row(self, row: Row) -> None:
        self.lines.append(row.text)
        self.boxes.append(row.box)
        if not self.amount_text:
            trailing = AMOUNT_RE.findall(row.text)
            if trailing:
                self.amount_text = trailing[-1]

    def finish(self) -> Transaction:
        box = BoundingBox(
            x_min=min(b.x_min for b in self.boxes),
            y_min=min(b.y_min for b in self.boxes),
            x_max=max(b.x_max for b in self.boxes),
            y_max=max(b.y_max for b in self.boxes),
        )
        return Transaction(
            page=self.page_index + 1,
            page_index=self.page_index,
            section=self.section,
            transaction_number=self.number,
            date=self.date,
            description="\n".join(self.lines).strip(),
            amount=parse_money(self.amount_text) if self.amount_text else None,
            amount_text=self.amount_text,
            box=box,
        )


def parse_page(page: PageResult) -> List[Transaction]:
    """Parse one page independently: header → sections → date-anchored rows."""
    transactions: List[Transaction] = []
    section = ""
    counter = 0
    current: Optional[_TxBuilder] = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            transactions.append(current.finish())
            current = None

    for row in build_rows(page):
        text = row.text.strip()
        if not text:
            continue

        if _normalize(text).startswith("TOTAL"):
            # Section totals close the running transaction and are not records.
            flush()
            continue

        found_section = detect_section(text)
        if found_section is not None:
            flush()
            section = found_section
            counter = 0
            continue

        date_match = DATE_START_RE.match(text)
        if date_match:
            flush()
            counter += 1
            rest = text[date_match.end():].strip()
            amount_text = ""
            trailing = list(AMOUNT_RE.finditer(rest))
            if trailing:
                last = trailing[-1]
                amount_text = last.group(0)
                if last.end() >= len(rest):
                    rest = rest[: last.start()].rstrip()
            current = _TxBuilder(
                page_index=page.page_index,
                section=section,
                number=counter,
                date=date_match.group(0),
                first_line=rest,
                amount_text=amount_text,
                box=row.box,
            )
            continue

        if current is not None:
            # Continuation line — never a separate transaction.
            current.add_row(row)

    flush()
    return transactions


# ---------------------------------------------------------------------------
# Statement-level metadata (bank / company / year / month)
# ---------------------------------------------------------------------------


def _detect_bank(text_upper: str) -> str:
    for key, name in BANK_NAMES.items():
        if key in text_upper:
            return name
    return "Unknown Bank"


def _detect_company(rows: List[Row]) -> str:
    for row in rows[:30]:
        text = row.text.strip()
        upper = text.upper()
        if "BANK" in upper or any(key in upper for key in BANK_NAMES):
            continue
        if detect_section(text):
            continue
        words = text.split()
        has_digits = any(ch.isdigit() for ch in text)
        if (
            len(words) >= 2
            and len(text) >= 5
            and not has_digits
            and text == text.upper()
            and any(ch.isalpha() for ch in text)
        ):
            return text
    return "Unknown Company"


def _detect_period(full_text: str) -> Tuple[str, str]:
    """Return (year, month) from the statement period, preferring the end date."""
    matches = list(_MONTH_RE.finditer(full_text))
    if matches:
        last = matches[-1]
        return last.group(2), last.group(1).title()
    numeric = list(_NUMERIC_PERIOD_RE.finditer(full_text))
    if numeric:
        last = numeric[-1]
        year = last.group(2)
        if len(year) == 2:
            year = "20" + year
        month_num = max(1, min(12, int(last.group(1))))
        return year, MONTH_NAMES[month_num - 1]
    return "Unknown Year", "Unknown Month"


def parse_statement(result: ScanResult) -> ParsedStatement:
    """Parse the whole document: metadata from page 1, each page independently."""
    transactions: List[Transaction] = []
    for page in result.pages:
        transactions.extend(parse_page(page))

    bank = "Unknown Bank"
    company = "Unknown Company"
    year, month = "Unknown Year", "Unknown Month"
    if result.pages:
        first = result.pages[0]
        rows = build_rows(first)
        page_text = " ".join(r.text for r in rows) or first.full_text
        bank = _detect_bank(page_text.upper())
        company = _detect_company(rows)
        year, month = _detect_period(page_text)

    return ParsedStatement(
        bank=bank,
        company=company,
        year=year,
        month=month,
        transactions=transactions,
    )
