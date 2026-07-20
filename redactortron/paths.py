"""Resolve optional native tool locations (Poppler, etc.)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_poppler_path() -> Optional[str]:
    """Return a directory containing ``pdftoppm``, if known.

    Lookup order:
      1. ``REDACTORTRON_POPPLER_PATH`` env var
      2. ``.tools/poppler/**/Library/bin`` (installer download layout)
      3. ``None`` — rely on system PATH
    """
    env = os.environ.get("REDACTORTRON_POPPLER_PATH", "").strip()
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return str(candidate)

    # Repo-local install from scripts/install_deps.py
    here = Path(__file__).resolve().parent.parent
    tools = here / ".tools" / "poppler"
    if tools.is_dir():
        for name in ("pdftoppm.exe", "pdftoppm"):
            matches = list(tools.rglob(name))
            if matches:
                return str(matches[0].parent)
    return None
