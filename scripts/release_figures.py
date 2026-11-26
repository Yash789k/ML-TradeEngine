"""Publication figures from frozen numeric artifacts, no retraining."""

from pathlib import Path
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--release", type=Path, required=True)
p.add_argument("--output", type=Path, default=Path("output/pdf/figures"))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
r = pd.read_parquet(a.release / "experiment/returns.parquet").loc[
    "2026-07-13":"2026-09-25"
]
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, ax = plt.subplots(figsize=(8.4, 3.5))
for name, label, color in [
    ("trend", "Selected trend rule", "#d97706"),
    ("adaptive_hybrid", "Adaptive hybrid", "#7c3aed"),
    ("ml", "ML policy", "#0d9488"),
    ("equal_weight", "Costed equal-weight benchmark", "#1e3a8a"),
    ("cash", "Cash", "#94a3b8"),
]:
    ax.plot(
        r.index,
        10000 * (1 + r[name]).cumprod(),
        label=label,
        color=color,
        linewidth=1.7,
    )
ax.set_ylabel("Rebased account value (USD)")
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(a.output / "audit_accounts.png", dpi=220)
plt.close(fig)
