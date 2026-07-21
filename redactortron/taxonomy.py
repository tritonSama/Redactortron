"""Category families: personal information vs transactions."""

from __future__ import annotations

from typing import FrozenSet

# Identity / holder data (name, address, account coordinates…).
PERSONAL_INFO_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "EMAIL",
        "PHONE NUMBER",
        "ADDRESS",
        "CREDIT CARD",
        "SSN",
        "ACCOUNT NUMBER",
        "ROUTING NUMBER",
        "BANK NAME",
    }
)

# Line-item / payment activity on statements and stubs.
TRANSACTION_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "TRANSACTION AMOUNT",
        "TRANSACTION DATE",
        "MERCHANT",
        "VENDOR",
        "INVOICE NUMBER",
        "CHECK NUMBER",
        "PAYMENT METHOD",
        "TRANSACTION ID",
        "BALANCE",
        "CURRENCY AMOUNT",
        "DEPOSIT",
        "WITHDRAWAL",
        "DATE",
        # Full statement records produced by the statement parser.
        "TRANSACTION",
    }
)

FAMILY_PERSONAL = "personal"
FAMILY_TRANSACTION = "transaction"
FAMILY_OTHER = "other"

MATRIX_VIEWS = (
    "All",
    "Personal information",
    "Transactions",
)


def category_family(category: str) -> str:
    key = category.strip().upper()
    in_personal = key in PERSONAL_INFO_CATEGORIES
    in_tx = key in TRANSACTION_CATEGORIES
    if in_personal and not in_tx:
        return FAMILY_PERSONAL
    if in_tx:
        # Shared labels like DATE lean transaction.
        return FAMILY_TRANSACTION
    return FAMILY_OTHER


def view_allows_family(view: str, family: str) -> bool:
    if view == "Personal information":
        return family == FAMILY_PERSONAL
    if view == "Transactions":
        return family == FAMILY_TRANSACTION
    return True
