"""
d0_distribution_multiyear.py -- how well-identified is the GibbsKernel's d_0 across 2010-2020?

Reads the per-window audit CSVs of the *_gibbs_lml runs (33 layer-years) and reports the
spread of the fitted d_transition_km (d_0, km from coast at the sigmoid midpoint of the
Gibbs lengthscale) per layer and per year, annotated with the coefficient of variation
(CV = std / mean). CV of 0.6-0.7 means window-to-window scatter of +/-60-70% of the mean,
i.e. d_0 is not pinned down by any single 45-day window.

Outputs:
  AEResults/aeplots/vertical_delta/d0_distribution_2010_2020.png
  AEResults/aelogs/d0_distribution_multiyear_stats.csv
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots 1-3 of the validated reference palette (blue, orange, aqua), one per layer.
# Ink for text/axes stays neutral; the colour only marks layer identity on boxes/bars.
LAYERS = [
    # key,        title,                      audit folder tag,        colour,    d_0 upper bound (km)
    ("d0_100",    "Skin (0-100 m)",           "_timelsfix_lml",        "#2a78d6", 700.0),
    ("d150_400",  "Source (150-400 m)",       "_timelsfix_lml",        "#eb6834", 700.0),
    ("d500_1000", "Background (500-1000 m)",  "_timelsfix_d1500_lml",  "#1baf7a", 1500.0),
]
LOWER_BOUND_KM = 50.0
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_audit(base_dir, layer_key, tag, year):
    # Loads one layer-year's per-window audit CSV (one row per rolling 45-day window).
    # Returns the d_transition_km column (km) as a float array.
    run = (f"californiav3_{year}0101_{year}1231_res0_5x0_5_t10_0_{layer_key}"
           f"_3dgibbs_w45{tag}")
    path = os.path.join(base_dir, "AEResults", "aelogs", run, f"audit_{run}.csv")
    return pd.read_csv(path)["d_transition_km"].to_numpy(dtype=float)


def cv(x):
    # Coefficient of variation with sample std (ddof=1): scatter as a fraction of the mean.
    return float(np.std(x, ddof=1) / np.mean(x))


def summarise(d0, upper, year, layer):
    # One stats row: spread of d_0 across the windows of a layer-year (or pooled years).
    # frac_at_*: share of windows whose fitted d_0 sits within 1% of a bound, i.e. the
    # optimiser hit a wall rather than finding an interior optimum.
    return dict(layer=layer, year=year, n_windows=len(d0), mean_km=np.mean(d0),
                median_km=np.median(d0), std_km=np.std(d0, ddof=1), cv=cv(d0),
                frac_at_lower=float(np.mean(d0 <= LOWER_BOUND_KM * 1.01)),
                frac_at_upper=float(np.mean(d0 >= upper * 0.99)))


def run_d0_distribution_multiyear(base_dir=".", years=range(2010, 2021)):
    """
    Build the multi-year d_0 distribution figure and stats table.

    base_dir: repo root (folder containing AEResults/).
    years: calendar years to include.
    Returns the stats DataFrame (per layer-year rows plus a pooled row per layer, year='all').
    """
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey="row")
    rows = []
    for j, (key, title, tag, colour, upper) in enumerate(LAYERS):
        per_year = {y: load_audit(base_dir, key, tag, y) for y in years}
        pooled = np.concatenate(list(per_year.values()))
        for y, d0 in per_year.items():
            rows.append(summarise(d0, upper, y, key))
        rows.append(summarise(pooled, upper, "all", key))

        # --- top row: one box per year, CV printed above each box ---
        ax = axes[0, j]
        bp = ax.boxplot(list(per_year.values()), positions=list(per_year), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK_2, linewidth=1), capprops=dict(color=INK_2))
        for patch in bp["boxes"]:
            patch.set(facecolor=colour, alpha=0.55, edgecolor=colour, linewidth=1)
        for y, d0 in per_year.items():
            ax.text(y, 1560, f"{cv(d0):.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.axhline(upper, color=INK_2, linestyle="--", linewidth=1)
        ax.axhline(LOWER_BOUND_KM, color=INK_2, linestyle="--", linewidth=1)
        ax.set_ylim(0, 1700)
        ax.set_xlim(min(years) - 0.8, max(years) + 0.8)
        ax.set_xticks(list(years))
        ax.set_xticklabels([str(y)[2:] for y in years])
        ax.set_xlabel("year (20YY)")
        ax.set_title(f"{title}\npooled CV = {cv(pooled):.2f}  (n = {len(pooled)} windows)",
                     fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        if j == 0:
            ax.set_ylabel("fitted d_0 (km from coast), per window")

        # --- bottom row: pooled histogram across all years and windows ---
        ax = axes[1, j]
        ax.hist(pooled, bins=np.linspace(0, 1600, 33), color=colour, alpha=0.7,
                edgecolor=SURFACE, linewidth=1)
        ax.axvline(upper, color=INK_2, linestyle="--", linewidth=1)
        ax.axvline(LOWER_BOUND_KM, color=INK_2, linestyle="--", linewidth=1)
        s = rows[-1]
        # Stats block goes where no bars/bound lines are: Skin/Source (upper bound 700 km) are
        # empty right of the bound, so use the right half; Background bars span the axes but sit
        # low (< ~17 windows) between its two bound spikes, so use mid-axes.
        if upper <= 700:
            # Right-aligned four-line block in the empty region right of the 700 km bound.
            txt = (f"CV = {s['cv']:.2f}\nmedian = {s['median_km']:.0f} km\n"
                   f"at lower bound: {100 * s['frac_at_lower']:.0f}%\n"
                   f"at upper bound: {100 * s['frac_at_upper']:.0f}%")
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK)
        else:
            txt = (f"CV = {s['cv']:.2f}   median = {s['median_km']:.0f} km\n"
                   f"at lower bound: {100 * s['frac_at_lower']:.0f}%   "
                   f"at upper bound: {100 * s['frac_at_upper']:.0f}%")
            ax.text(0.5, 0.95, txt, transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color=INK)
        ax.set_xlabel("fitted d_0 (km from coast)")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        if j == 0:
            ax.set_ylabel("windows (all years)")
        for a in (axes[0, j], ax):
            a.spines[["top", "right"]].set_visible(False)

    fig.suptitle("GibbsKernel d_0 is not pinned down by single windows: scatter across 2010-2020\n"
                 "numbers above boxes = CV per year; dashed lines = optimiser bounds "
                 "(50 km lower; 700 km upper for Skin/Source, 1500 km for Background)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plot_dir = os.path.join(base_dir, "AEResults", "aeplots", "vertical_delta")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "d0_distribution_2010_2020.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    stats = pd.DataFrame(rows)
    stats_path = os.path.join(base_dir, "AEResults", "aelogs", "d0_distribution_multiyear_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"saved {plot_path}\nsaved {stats_path}")
    return stats


if __name__ == "__main__":
    st = run_d0_distribution_multiyear(os.path.dirname(os.path.abspath(__file__)))
    print(st[st.year == "all"].round(3).to_string(index=False))
    print(st[st.year != "all"].groupby("layer").cv.describe().round(2).to_string())
