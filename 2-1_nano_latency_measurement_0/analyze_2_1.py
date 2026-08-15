#!/usr/bin/env python3
"""
analyze_2_1.py
===============
Reproduces Fig 2-1 (latency pipeline stage breakdown, Notion Experiment 2)
from the 27-scenario combined summary CSV (3 precisions x 3 people-counts x
3 power modes, produced by beta_S6_latency_harness.py -- bundled here as
beta_S6_latency_harness.py / 2_1_integrity_check.py for reference).

Extract 2-1_csv.7z here first -- it unpacks to "2-1_csv/", containing the
27 raw per-scenario result folders (summary.jsonl, tegrastats.log, 9
per-frame CSVs each) plus 2_1_all_27_scenarios_combined.csv, the pre-parsed
summary this script actually reads. This chart only uses n_people=1 (fixed)
across the 3 precisions x 3 power modes -- the people-count scaling,
per-frame, and tegrastats/thermal angles in the raw data aren't plotted here.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.labelsize"] = 15
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "2-1_csv"
SRC_CSV = DATA_DIR / "2_1_all_27_scenarios_combined.csv"
CHARTS_DIR = Path(__file__).parent / "charts"

ENGINE_ORDER = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
ENGINE_LABEL = {"fp32": "FP32", "fp16": "FP16", "modelopt_int8_excl_cv4": "INT8\nmixed"}
STAGE_COLS = ["capture_ms_mean", "preprocess_ms_mean", "gpu_ms_mean", "decode_ms_mean", "logic_ms_mean"]
STAGE_LABEL = {"capture_ms_mean": "capture", "preprocess_ms_mean": "preprocess", "gpu_ms_mean": "GPU inference",
               "decode_ms_mean": "decode", "logic_ms_mean": "logic"}
STAGE_COLOR = {"capture_ms_mean": "#B0B0B0", "preprocess_ms_mean": "#DD8452", "gpu_ms_mean": "#4C72B0",
               "decode_ms_mean": "#55A868", "logic_ms_mean": "#8172B2"}
STAGE_LETTER = {"capture_ms_mean": "a", "preprocess_ms_mean": "b", "gpu_ms_mean": "c",
                 "decode_ms_mean": "d", "logic_ms_mean": "e"}


def savefig(fig, name):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg, +.pdf)")


POWER_MODE_ORDER = ["MAXN_SUPER", "25W", "15W"]
POWER_MODE_LABEL = {"MAXN_SUPER": "MAX", "25W": "25W", "15W": "15W"}


def main():
    if not SRC_CSV.exists():
        raise SystemExit(f"expected {SRC_CSV} -- extract 2-1_csv.7z here first")
    CHARTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(SRC_CSV)

    # 9 bars: 3 precisions x 3 power modes, each stage averaged across all
    # three tested people-counts (1/2/3) rather than fixed at n_people=1 --
    # decode/logic scale slightly with the number of detected people (more
    # per-person work each frame), so averaging across the full 1-3 range
    # tested elsewhere in this study is more representative than reporting
    # only the single-person case.
    rows = []
    for engine in ENGINE_ORDER:
        for mode in POWER_MODE_ORDER:
            sub = df[(df["engine_short"] == engine) & (df["mode"] == mode)]
            avg_row = sub[STAGE_COLS].mean()
            avg_row["engine_short"] = engine
            avg_row["mode"] = mode
            rows.append(avg_row)
    plot_df = pd.DataFrame(rows).reset_index(drop=True)

    x = list(range(len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    # Embeds at 3.4in single-column width; native canvas is wider than that
    # so the 9 grouped x-tick labels (precision + power mode, up to 3 lines
    # each) have room to breathe, so fonts are back-calculated to land at
    # the intended size once LaTeX shrinks it down -- see Fig 5's _SHRINK5
    # in analyze_1_3.py for the same pattern. No in-plot legend: the stage
    # letters (a)-(e) are labeled directly on the first bar group and
    # decoded in the caption instead, since a legend box would collide with
    # the taller multi-line x-tick labels at this width.
    _SHRINK4 = 3.4 / 9.891
    bottom = [0.0] * len(plot_df)
    for stage in STAGE_COLS:
        vals = plot_df[stage].values
        letter = STAGE_LETTER[stage]
        ax.bar(x, vals, bottom=bottom, width=0.6,
               facecolor=STAGE_COLOR[stage], edgecolor="black", linewidth=0.6)
        # letter label only on the first bar (FP32/MAXN_SUPER) -- position
        # taught there carries over to the other 8 bars, matching Fig 1-5-7.
        # "logic" (e) is too thin a slice to hold a centered label, so it
        # goes just above the bar instead of inside its segment.
        top = bottom[0] + vals[0]
        if stage == "logic_ms_mean":
            ax.text(x[0], top - 0.05, letter, ha="center", va="bottom", fontsize=8 / _SHRINK4)
        else:
            ax.text(x[0], bottom[0] + vals[0] / 2, letter, ha="center", va="center", fontsize=8 / _SHRINK4)
        bottom = [b + v for b, v in zip(bottom, vals)]

    xlabels = [f"{ENGINE_LABEL[r.engine_short]}\n{POWER_MODE_LABEL[r.mode]}" for r in plot_df.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7 / _SHRINK4)
    ax.set_ylabel("latency (ms)", fontsize=9 / _SHRINK4)
    ax.tick_params(axis="y", labelsize=8 / _SHRINK4)
    fig.tight_layout()
    savefig(fig, "2-1_latency_stage_breakdown_by_power_mode")

    print("\nDONE")


if __name__ == "__main__":
    main()
