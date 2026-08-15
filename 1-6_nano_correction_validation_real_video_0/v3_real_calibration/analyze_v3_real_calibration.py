#!/usr/bin/env python3
"""
Final analysis for the v3 real-video distance-adaptive threshold validation.
Consumes:
  - results/{fp32,fp16,int8,modelopt_int8,modelopt_int8_excl_cv4,
    modelopt_int8_excl_cv4_fp32}.csv -- deployed-model TensorRT inference,
    produced on the Nano by batch_infer_calib_eval_split.py.
  - results/yolo11x_fp32.csv -- reference-curve inference, produced on the PC
    by infer_yolo11x_reference.py (plain onnxruntime, yolo11x is never
    quantized/deployed so this doesn't need the Nano).

Fixes three problems found in the v1/v2 pipelines (see README.md for the
full trail and the numbers each fix produced):
  1. own_mean/own_std (the deployed model's own per-distance R normalization)
     was computed from SYNTHETIC data in v1/v2 -- real own_std turns out to
     be ~2-3x smaller than the synthetic-derived value, distorting the
     correction. Fixed here: computed from real video (the calib split)
     instead.
  2. Naively reusing the same real-video sample for both calibration and
     evaluation reintroduces a same-sample-reuse concern. Fixed here: the
     frame pool was split into disjoint calib/eval halves at generation time
     (generate_calib_eval_split_frames.py) -- confirmed zero source-frame
     overlap between them. This script re-partitions its OWN way on top of
     that (see CALIB_FRACTION below) since own_mean/own_std turned out to
     need far less data than the miss-rate evaluation does (own_mean/own_std
     estimates were already stable via bootstrap at n=250 per distance,
     while eval benefits from as much data as possible) -- so a 50/50 split
     wastes eval precision for no calibration benefit. Whatever fraction is
     used, calib and eval are re-partitioned from the disjoint pool
     (ignoring the original generation-time calib/eval folder labels, which
     were an even 50/50 split), so this is still leak-free regardless of
     CALIB_FRACTION.
  3. A handful of low-keypoint-confidence detections (confirmed by pulling
     actual frames: the model locking onto background clutter instead of
     the person) produced R values wildly outside the plausible range
     (>1, sometimes >3, vs a normal ~0.2-0.45), which skewed miss rate at
     specific distances even though they weren't real pose measurements at
     all. Fixed here: filter with the SAME confidence thresholds the
     deployed beta_S6.py pipeline already applies (CONF_SHOULDER_THRESHOLD =
     0.5, CONF_WRIST_THRESHOLD = 0.4, both L/R) -- this offline analysis
     pipeline had been missing that filter, not the deployed system.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
ADAPTIVE_PCTL = 95
CALIB_FRACTION = 0.2  # own_mean/own_std estimates are stable well below this; eval gets the rest
SPLIT_SEED = 42

# matches beta_S6_latency_harness.py exactly -- do not drift from these without
# re-checking the deployed pipeline's own values first.
CONF_SHOULDER_THRESHOLD = 0.5
CONF_WRIST_THRESHOLD = 0.4

DEPLOYED_VARIANTS = ["fp32", "fp16", "int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]
TARGET_DISTANCES = [250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shoulder_px"] = np.sqrt((df["left_shoulder_x"] - df["right_shoulder_x"]) ** 2 +
                                 (df["left_shoulder_y"] - df["right_shoulder_y"]) ** 2)
    df["wrist_px"] = np.sqrt((df["left_wrist_x"] - df["right_wrist_x"]) ** 2 +
                              (df["left_wrist_y"] - df["right_wrist_y"]) ** 2)
    df["R"] = df["wrist_px"] / df["shoulder_px"]
    return df


def apply_deployed_confidence_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Same rule as beta_S6.py's conf_ok: both shoulders >= 0.5 AND both wrists >= 0.4."""
    shoulder_ok = (df["left_shoulder_conf"] >= CONF_SHOULDER_THRESHOLD) & (df["right_shoulder_conf"] >= CONF_SHOULDER_THRESHOLD)
    wrist_ok = (df["left_wrist_conf"] >= CONF_WRIST_THRESHOLD) & (df["right_wrist_conf"] >= CONF_WRIST_THRESHOLD)
    return df[shoulder_ok & wrist_ok].copy()


def make_calib_eval_split(frames: pd.DataFrame) -> pd.DataFrame:
    """Fresh disjoint split across the full frame pool, ignoring the original
    generation-time calib/eval folder labels (that was an even 50/50 split;
    this rebalances toward eval, see CALIB_FRACTION)."""
    rng = np.random.default_rng(SPLIT_SEED)
    frames = frames.copy()
    frames["v3_split"] = rng.choice(["calib", "eval"], size=len(frames), p=[CALIB_FRACTION, 1 - CALIB_FRACTION])
    return frames


def build_curve(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("target_distance_cm").agg(
        R_mean=("R", "mean"), R_std=("R", "std"),
        R_p95=("R", lambda s: s.quantile(ADAPTIVE_PCTL / 100)),
        n=("R", "count"),
    ).reset_index()


def main():
    x11_path = RESULTS_DIR / "yolo11x_fp32.csv"
    if not x11_path.exists():
        raise SystemExit(f"expected {x11_path} -- run infer_yolo11x_reference.py on the PC first")

    x11 = add_derived_columns(pd.read_csv(x11_path))
    x11 = apply_deployed_confidence_filter(x11)
    key = ["n_people", "target_distance_cm", "frame_file"]
    x11_frames = x11[key].drop_duplicates()
    x11 = x11.merge(make_calib_eval_split(x11_frames), on=key, how="left")

    x11_calib = x11[x11["v3_split"] == "calib"]
    x11_curve = build_curve(x11_calib)
    x11_mean = dict(zip(x11_curve["target_distance_cm"], x11_curve["R_mean"]))
    x11_std = dict(zip(x11_curve["target_distance_cm"], x11_curve["R_std"]))
    x11_thr = dict(zip(x11_curve["target_distance_cm"], x11_curve["R_p95"]))
    print("=== yolo11x reference curve (calib split, confidence-filtered) ===")
    print(x11_curve.round(4).to_string(index=False))
    print()

    all_summaries = []
    for variant in DEPLOYED_VARIANTS:
        v_path = RESULTS_DIR / f"{variant}.csv"
        if not v_path.exists():
            print(f"[SKIP] {v_path} not found (engine missing on Nano run?)")
            continue

        v = add_derived_columns(pd.read_csv(v_path))
        v = apply_deployed_confidence_filter(v)
        v_frames = v[key].drop_duplicates()
        v = v.merge(make_calib_eval_split(v_frames), on=key, how="left")

        v_calib = v[v["v3_split"] == "calib"]
        own_curve = build_curve(v_calib)
        own_mean = dict(zip(own_curve["target_distance_cm"], own_curve["R_mean"]))
        own_std = dict(zip(own_curve["target_distance_cm"], own_curve["R_std"]))

        v_eval = v[v["v3_split"] == "eval"].copy()
        v_eval["R_corrected"] = (
            v_eval["target_distance_cm"].map(x11_mean)
            + (v_eval["R"] - v_eval["target_distance_cm"].map(own_mean))
            * (v_eval["target_distance_cm"].map(x11_std) / v_eval["target_distance_cm"].map(own_std))
        )
        v_eval["threshold"] = v_eval["target_distance_cm"].map(x11_thr)
        v_eval["miss_corrected"] = v_eval["R_corrected"] >= v_eval["threshold"]

        summary = v_eval.groupby("target_distance_cm").agg(
            miss_corrected=("miss_corrected", "mean"), n=("R", "count"),
            R_raw_mean=("R", "mean"),
            R_corrected_mean=("R_corrected", "mean"), R_corrected_std=("R_corrected", "std")).reset_index()
        summary["variant"] = variant
        summary["miss_corrected_pct"] = (summary["miss_corrected"] * 100).round(2)
        all_summaries.append(summary)

        print(f"=== {variant}: eval-split corrected miss rate ===")
        print(summary[["target_distance_cm", "miss_corrected_pct", "n"]].to_string(index=False))
        print(f"  max: {summary['miss_corrected_pct'].max()}% at {summary.loc[summary['miss_corrected_pct'].idxmax(), 'target_distance_cm']}cm")
        print()

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "v3_final_miss_rate_by_variant.csv", index=False)
        print(f"Saved combined results to {RESULTS_DIR / 'v3_final_miss_rate_by_variant.csv'}")
        make_fig12(combined, x11_curve)


def make_fig12(combined: pd.DataFrame, x11_curve: pd.DataFrame):
    """Twin of Fig 1-3-6 (analyze_1_3.py): raw vs corrected R vs distance,
    relative to the yolo11x target/threshold. Same figsize, marker/fill/
    color scheme, and legend layout as the v1/v2 versions of this figure,
    for visual consistency across the paper's figures -- only the
    underlying data source changed (v3: real-video calibration, disjoint
    calib/eval split, deployed confidence filter)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "Times New Roman"
    import matplotlib.pyplot as plt

    FIG2_VARIANTS = ["fp16", "modelopt_int8_excl_cv4"]
    CHART_LABEL_SHORT = {"fp16": "FP16", "modelopt_int8_excl_cv4": "INT8-mix"}
    CHART_COLOR = {"fp16": "#e07b39", "modelopt_int8_excl_cv4": "#4a9c6d"}
    fill = {"fp16": None, "modelopt_int8_excl_cv4": "white"}

    fig, ax = plt.subplots(figsize=(15, 10.0))
    _SHRINK12 = 3.4 / 9.5
    _SHRINK12_OTHER = 3.4 / 14.87
    h_target = ax.plot(x11_curve["target_distance_cm"], x11_curve["R_mean"], color="tab:blue", linestyle="-",
                        linewidth=2, alpha=0.5, label="11x target")[0]
    h_thresh = ax.plot(x11_curve["target_distance_cm"], x11_curve["R_p95"], color="black", linestyle=":",
                        linewidth=1.5, label="11x threshold")[0]
    handles_for_legend = {("yolo11x", "target"): (h_target, "11x target"),
                          ("yolo11x", "threshold"): (h_thresh, "11x threshold")}
    for variant in FIG2_VARIANTS:
        sub = combined[combined["variant"] == variant].sort_values("target_distance_cm")
        color = CHART_COLOR[variant]
        label_raw = f"{CHART_LABEL_SHORT[variant]} - raw"
        label_corr = f"{CHART_LABEL_SHORT[variant]} - corr"
        h1, = ax.plot(sub["target_distance_cm"], sub["R_raw_mean"], color=color, linestyle="--", marker="s",
                    markersize=18, markerfacecolor=fill[variant], label=label_raw)
        h2 = ax.errorbar(sub["target_distance_cm"], sub["R_corrected_mean"], yerr=sub["R_corrected_std"], color=color,
                    linestyle="-", marker="o", markersize=18, markerfacecolor=fill[variant], capsize=3, label=label_corr)
        handles_for_legend[(variant, "raw")] = (h1, label_raw)
        handles_for_legend[(variant, "corrected")] = (h2, label_corr)
    # no real/simulated boundary line -- v3 draws every target distance (including
    # 250-400cm) through the same randomized pool-draw + rescale mechanism, so
    # there's no methodology split left to mark on the x-axis.
    ax.set_xlabel("distance (cm)", fontsize=9 / _SHRINK12_OTHER)
    ax.set_ylabel("R (wrist_dist / shoulder_dist)", fontsize=9 / _SHRINK12_OTHER)
    ax.tick_params(axis="both", labelsize=8 / _SHRINK12_OTHER)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
    fig.tight_layout()
    legend_order = [("fp16", "raw"), ("modelopt_int8_excl_cv4", "raw"),
                     ("fp16", "corrected"), ("modelopt_int8_excl_cv4", "corrected"),
                     ("yolo11x", "target"), ("yolo11x", "threshold")]
    handles = [handles_for_legend[k][0] for k in legend_order]
    labels = [handles_for_legend[k][1] for k in legend_order]
    ax.legend(handles, labels, fontsize=10 / _SHRINK12, loc="upper left", ncol=3,
              labelspacing=0.4, columnspacing=0.8, handlelength=2.5, handletextpad=0.4, borderpad=0.8)
    out_base = RESULTS_DIR / "v3_1-6-2_R_value_raw_vs_corrected_vs_distance"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 12 (v3) to {out_base}.pdf/.png")


if __name__ == "__main__":
    main()
