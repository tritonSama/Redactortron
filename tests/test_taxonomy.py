"""Tests for personal-info vs transaction taxonomy."""

from __future__ import annotations

from redactortron.taxonomy import (
    category_family,
    view_allows_family,
)


def test_category_family_split() -> None:
    assert category_family("ACCOUNT NUMBER") == "personal"
    assert category_family("TRANSACTION AMOUNT") == "transaction"
    assert category_family("MERCHANT") == "transaction"
    assert category_family("PERSON") == "personal"
    assert category_family("WORD MATCH") == "other"


def test_matrix_view_filters() -> None:
    assert view_allows_family("All", "personal")
    assert view_allows_family("All", "transaction")
    assert view_allows_family("All", "other")
    assert view_allows_family("Personal information", "personal")
    assert not view_allows_family("Personal information", "transaction")
    assert view_allows_family("Transactions", "transaction")
    assert not view_allows_family("Transactions", "personal")
