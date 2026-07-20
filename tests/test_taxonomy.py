"""Tests for account vs transaction taxonomy."""

from __future__ import annotations

from redactortron.taxonomy import (
    category_family,
    view_allows_family,
)


def test_category_family_split() -> None:
    assert category_family("ACCOUNT NUMBER") == "account"
    assert category_family("TRANSACTION AMOUNT") == "transaction"
    assert category_family("MERCHANT") == "transaction"
    assert category_family("PERSON") == "account"


def test_matrix_view_filters() -> None:
    assert view_allows_family("All", "account")
    assert view_allows_family("All", "transaction")
    assert view_allows_family("Account information", "account")
    assert not view_allows_family("Account information", "transaction")
    assert view_allows_family("Account transactions", "transaction")
    assert not view_allows_family("Account transactions", "account")
