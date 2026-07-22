"""Resolve the external publication-figure directory without hard-coding a host path.

The local MIMO-Rutgers layout keeps the code repository below ``1-Codes/`` and official
publication PDFs below ``6-Writing/Aim3/Figures/``.  Set
``AIM3_PUBLICATION_FIGURES_DIR`` to override that sibling-tree convention on another host.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_FIGURES_ENV = "AIM3_PUBLICATION_FIGURES_DIR"


def publication_figures_dir(
    explicit: str | Path | None = None,
    *,
    create: bool = False,
) -> Path | None:
    """Return the configured official publication-figure directory when available."""

    configured = explicit or os.environ.get(PUBLICATION_FIGURES_ENV)
    if configured is not None:
        destination = Path(configured).expanduser().resolve()
    else:
        destination = PROJECT_ROOT.parents[1] / "6-Writing" / "Aim3" / "Figures"
        if not destination.parent.is_dir():
            return None
    if create:
        destination.mkdir(parents=True, exist_ok=True)
    elif not destination.is_dir():
        return None
    return destination
