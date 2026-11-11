"""Shared matplotlib style: readable at IEEE two-column width (3.5 in)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.5
DBL_W = 7.16

METHODS = ["tiny_int8", "tiny_f32", "lstm", "persist", "zscore", "random"]
LABELS = {"tiny_int8": "Tiny MLP, INT8", "tiny_f32": "Tiny MLP, float32", "lstm": "LSTM 2x80, float32",
          "persist": "Persistence", "zscore": "Rolling z-score", "random": "Random scores"}
COLORS = {"tiny_int8": "#1f4e9c", "tiny_f32": "#4c9be8", "lstm": "#d9822b", "persist": "#2e8b57",
          "zscore": "#8e6c3a", "random": "#7f7f7f"}


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
        "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
        "font.family": "serif",
    })
    return plt


def save(fig, path):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".png"), dpi=300)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
