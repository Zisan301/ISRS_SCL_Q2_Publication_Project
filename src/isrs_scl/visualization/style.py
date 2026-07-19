"""Publication plotting style."""

from __future__ import annotations

import matplotlib as mpl


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.figsize": (3.5, 2.5),
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
