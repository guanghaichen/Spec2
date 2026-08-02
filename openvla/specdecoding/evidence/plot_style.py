"""Publication defaults sized for the ICLR two-column template."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl


# The official ICLR 2026 style sets \textwidth to 5.5 true inches. ICLR is a
# single-column format, so full-width figures must not use an IEEE-style 6.75
# inch double-column canvas.
ICLR_SINGLE_COLUMN_IN = 3.25
ICLR_DOUBLE_COLUMN_IN = 5.5
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
}


@contextmanager
def iclr_style():
    """Use readable vector-first defaults without requiring system fonts."""
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        yield


def save_figure(fig, output_stem) -> None:
    """Save an editable vector PDF and a high-resolution review PNG."""
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
