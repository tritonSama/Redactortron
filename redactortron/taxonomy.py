"""Category families: account information vs transactions."""

from __future__ import annotations

from typing import FrozenSet

# Holder / identity / banking coordinates (not line-item activity).
ACCOUNT_INFO_CATEGORIES: FrozenSet[str] = frozenset(
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
    }
)

FAMILY_ACCOUNT = "account"
FAMILY_TRANSACTION = "transaction"
FAMILY_OTHER = "other"

MATRIX_VIEWS = (
    "All",
    "Account information",
    "Account transactions",
)


def category_family(category: str) -> str:
    key = category.strip().upper()
    in_account = key in ACCOUNT_INFO_CATEGORIES
    in_tx = key in TRANSACTION_CATEGORIES
    if in_account and not in_tx:
        return FAMILY_ACCOUNT
    if in_tx and not in_account:
        return FAMILY_TRANSACTION
    if in_account and in_tx:
        # Prefer transaction for shared labels like DATE when viewing txs.
        return FAMILY_TRANSACTION
    return FAMILY_OTHER


def view_allows_family(view: str, family: str) -> bool:
    if view == "All":
        return True
    if view == "Account information":
        return family in {FAMILY_ACCOUNT, FAMILY_OTHER}
    if view == "Account transactions":
        return family == FAMILY_TRANSACTION
    return True
