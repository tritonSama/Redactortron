"""Tests for the intelligent bank-statement parser."""

from __future__ import annotations

from redactortron.models import BoundingBox, PageResult, ScanResult, WordSpan
from redactortron.statement import (
    build_rows,
    detect_section,
    parse_money,
    parse_page,
    parse_statement,
)


def _word(text: str, x: int, y: int, page: int = 0) -> WordSpan:
    width = max(8, 7 * len(text))
    return WordSpan(
        text=text,
        box=BoundingBox(x, y, x + width, y + 10),
        page_index=page,
        start_char=0,
        end_char=len(text),
    )


def _line(text: str, y: int, page: int = 0) -> list:
    words = []
    x = 0
    for token in text.split():
        words.append(_word(token, x, y, page))
        x += 7 * len(token) + 10
    return words


def _statement_page() -> PageResult:
    words = []
    words += _line("JPMorgan Chase Bank, N.A.", 0)
    words += _line("HITEK FILMS LLC", 20)
    words += _line("December 01, 2022 through December 30, 2022", 40)
    words += _line("DEPOSITS AND ADDITIONS", 60)
    words += _line("DATE DESCRIPTION AMOUNT", 80)
    words += _line("12/12 Wire Reversal 154.00", 100)
    words += _line("B/O: JPMorgan Chase Bank National", 120)
    words += _line("Org: Fx", 140)
    words += _line("12/13 Zelle Payment From Bob 20.00", 160)
    words += _line("Total Deposits and Additions $174.00", 180)
    words += _line("ELECTRONIC WITHDRAWALS", 200)
    words += _line("12/14 Online Payment To Vendor 1,250.50", 220)
    return PageResult(
        page_index=0,
        width=800,
        height=400,
        full_text=" ".join(w.text for w in words),
        words=words,
    )


def _result() -> ScanResult:
    return ScanResult(source_path="statement.pdf", pages=[_statement_page()])


def test_build_rows_groups_by_visual_line() -> None:
    rows = build_rows(_statement_page())
    texts = [r.text for r in rows]
    assert "12/12 Wire Reversal 154.00" in texts
    assert "HITEK FILMS LLC" in texts


def test_detect_section_and_money() -> None:
    assert detect_section("DEPOSITS AND ADDITIONS") == "Deposits And Additions"
    assert detect_section("Deposits and Additions (continued)") is not None
    assert detect_section("Total Deposits and Additions $174.00") is None
    assert detect_section("12/12 Wire Reversal 154.00") is None
    assert parse_money("$1,250.50") == 1250.50
    assert parse_money("154.00-") == -154.00
    assert parse_money("garbage") is None


def test_parse_page_transactions() -> None:
    txs = parse_page(_statement_page())
    assert len(txs) == 3

    first = txs[0]
    assert first.section == "Deposits And Additions"
    assert first.transaction_number == 1
    assert first.date == "12/12"
    assert first.amount == 154.00
    assert first.description == (
        "Wire Reversal\nB/O: JPMorgan Chase Bank National\nOrg: Fx"
    )

    second = txs[1]
    assert second.transaction_number == 2
    assert second.date == "12/13"
    assert second.amount == 20.00

    third = txs[2]
    assert third.section == "Electronic Withdrawals"
    assert third.transaction_number == 1  # counter resets per section
    assert third.amount == 1250.50


def test_parse_statement_hierarchy_and_search() -> None:
    statement = parse_statement(_result())
    assert statement.bank == "Chase Bank"
    assert statement.company == "HITEK FILMS LLC"
    assert statement.year == "2022"
    assert statement.month == "December"

    # Search must include continuation lines.
    matches = statement.search("B/O: JPMorgan Chase Bank National")
    assert len(matches) == 1
    assert matches[0].date == "12/12"
    assert statement.search("nonexistent text") == []

    tree = statement.tree_markdown()
    assert "### Chase Bank" in tree
    assert "HITEK FILMS LLC" in tree
    assert "December" in tree
    assert "Page 1" in tree
    assert "Deposits And Additions" in tree

    js = statement.to_json()
    assert '"transaction_number": 1' in js
    assert '"date": "12/12"' in js
    assert '"amount": 154.0' in js
