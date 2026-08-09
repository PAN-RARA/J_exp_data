#!/usr/bin/env python3
"""
analyze_2_1.py
===============
Reproduces the Notion 2-1 latency measurement chart (Fig 2-1) from the
27-scenario combined summary CSV (3 precisions x 3 people-counts x 3 power
modes, produced by beta_S6_latency_harness.py). This script only uses the
MAXN_SUPER power mode (full performance, the deployed setting) -- the 15W/25W
scenarios are in the same CSV but not plotted here.

Source: C:/Users/user/pose_quant_env/exported_charts/2-1/2_1_all_27_scenarios_combined.csv
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

SRC_CSV = Path(r"C:\Users\user\pose_quant_env\exported_charts\2-1\2_1_all_27_scenarios_combined.csv")
CHARTS_DIR = Path(__file__).parent / "charts"

ENGINE_ORDER = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
ENGINE_LABEL = {"fp32": "FP32", "fp16": "FP16", "modelopt_int8_excl_cv4": "INT8(mixed)"}
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


def main():
    CHARTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(SRC_CSV)
    df = df[df["n_people"] == 1].copy()

    # 9 bars: 3 precisions x 3 power modes (fixed at n_people=1), each
    # stacked by pipeline stage (capture -> preprocess -> GPU -> decode ->
    # logic).
    rows = []
    for engine in ENGINE_ORDER:
        for mode in POWER_MODE_ORDER:
            sub = df[(df["engine_short"] == engine) & (df["mode"] == mode)]
            rows.append(sub.iloc[0])
    plot_df = pd.DataFrame(rows).reset_index(drop=True)

    x = list(range(len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0.0] * len(plot_df)
    for stage in STAGE_COLS:
        vals = plot_df[stage].values
        letter = STAGE_LETTER[stage]
        ax.bar(x, vals, bottom=bottom, width=0.6, label=f"({letter}) {STAGE_LABEL[stage]}",
               facecolor=STAGE_COLOR[stage], edgecolor="black", linewidth=0.6)
        # letter label only on the first bar (FP32/MAXN_SUPER) -- position
        # taught there carries over to the other 8 bars, matching Fig 1-5-7.
        # "logic" (e) is too thin a slice to hold a centered label, so it
        # goes just above the bar instead of inside its segment.
        top = bottom[0] + vals[0]
        if stage == "logic_ms_mean":
            ax.text(x[0], top + 0.15, letter, ha="center", va="bottom", fontsize=12)
        else:
            ax.text(x[0], bottom[0] + vals[0] / 2, letter, ha="center", va="center", fontsize=12)
        bottom = [b + v for b, v in zip(bottom, vals)]

    xlabels = [f"{ENGINE_LABEL[r.engine_short]}\n{r.mode}" for r in plot_df.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel("latency (ms)")
    # reversed so the legend's top-to-bottom order matches the stack's
    # top-to-bottom order (logic on top of both the bars and the legend).
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=12, loc="upper right")
    fig.tight_layout()
    savefig(fig, "2-1_latency_stage_breakdown_by_power_mode")

    print("\nDONE")


if __name__ == "__main__":
    main()
