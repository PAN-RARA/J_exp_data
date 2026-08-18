#!/usr/bin/env python3
"""
analyze_1_6.py
===============
Reproduces the Notion 1-6 numbers and charts from the per-detection CSVs in
1-6_csv.7z, extracted into the sibling "1-6_csv/" folder (flat, prefixed
by source: real_{variant}.csv, simulated_{variant}.csv).
Produces the current Fig 1-6-1. Also still computes this script's OWN
(v1-methodology, synthetic-calibrated) Fig 1-6-2, but that now saves into
charts/superseded/ as "..._v1" -- the canonical charts/1-6-2_... is produced
by analyze_v3_real_calibration.py instead (real-video calibration; see
v3_real_calibration_README.md).

Validates whether 1-3/1-5's full correction methodology (yolo11x-calibrated
threshold + per-variant ratio correction) generalizes from synthetic data to
genuine real human video. 250-400cm is real footage (21 distance-labeled
videos, 1/2/3 people); 275-550cm is simulated by canvas-preserving downscale
of the nearest real frames (see nano_scripts/simulate_far_distance_frames.py
methodology, Zhang et al. 2025 precedent for resolution-as-distance-proxy).

yolo11x target curve (2026-08-07): unlike 1-3/1-5, which use yolo11x-pose's
SYNTHETIC self-referential FP32 curve as the correction target, this script
uses a REAL-VIDEO yolo11x-pose curve instead (bundled here as
yolo11x-pose.real_video.csv -- computed by run_yolo11x_real_video.py,
onnxruntime CUDA EP inference directly on Test_21/*.mp4 frames + the same
canvas-preserving downscale for the simulated range, then top-N
confidence-filtered per frame to drop glass-reflection false positives, same
filter as the v8n variants below). This was a deliberate methodology
upgrade: the original synthetic-only yolo11x target meant the correction's
real-world validity rested on an unverified assumption (that v8n's
real-vs-synthetic domain gap matches yolo11x's). Using a real-video target
removes that assumption. NOTE: the corrected miss-rate numbers (Fig 1-6-1)
are algebraically IDENTICAL regardless of which yolo11x curve is used here
-- the target's mean/std cancel out of the corrected-vs-threshold test (see
R_corrected formula below: threshold and R_corrected both scale by the same
x11_mean/x11_std, so only (R_raw-own_mean)/own_std, entirely target-curve-
independent, determines a pass/fail). Fig 1-6-2, which plots the target
curve directly, is the one number that DOES depend on this choice.

External shared dependency (documented, not duplicated -- same pattern as
analyze_1_3.py/analyze_1_5.py): each v8n variant's OWN self-referential
mean/std curve (own_mean/own_std below) is still fit on SYNTHETIC data,
reusing 1-3's synthetic CSVs:
    SYNTH_DIR = Path(__file__).resolve().parent.parent.parent / "1-3_nano_quant_palms_together_synthetic_samples_0" / "1-3_csv"
Extract 1-3_csv.7z there first (sibling repo folder) if not already done.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.labelsize"] = 13
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = ROOT.parent / "1-3_nano_quant_palms_together_synthetic_samples_0" / "1-3_csv"
DATA_DIR = ROOT / "1-6_csv"
YOLO11X_REAL_VIDEO_CSV = DATA_DIR / "yolo11x-pose.real_video.csv"
CHARTS_DIR = ROOT / "charts"
# Empirical 95th-percentile threshold, not mean+1.645*std -- see analyze_1_3.py's
# ADAPTIVE_PCTL comment: Shapiro-Wilk rejects normality for yolo11x-pose's own
# R distribution at every distance bin, and the parametric formula under-
# estimates the true 95th percentile (systematically too strict).
ADAPTIVE_PCTL = 95
OVERLAP_DISTANCES = [250, 300, 350, 400]
SIM_DISTANCES = [275, 325, 375, 425, 450, 475, 500, 525, 550]
ALL_DISTANCES = sorted(OVERLAP_DISTANCES + SIM_DISTANCES)

VARIANTS = ["fp32", "fp16", "int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]
CHART_VARIANTS = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
CHART_COLOR = {"fp32": "#555555", "fp16": "#e07b39", "modelopt_int8_excl_cv4": "#4a9c6d"}
CHART_LABEL_SHORT = {"fp32": "FP32", "fp16": "FP16", "modelopt_int8_excl_cv4": "INT8-mix"}
# IEEE print figures fall back to grayscale -- each variant gets its own
# (raw_marker, corrected_marker) pair so all 3 stay shape-distinguishable.
CHART_MARKERS = {"fp32": ("o", "^"), "fp16": ("s", "D"), "modelopt_int8_excl_cv4": ("P", "X")}


def _dist(df, kp):
    return np.sqrt((df[f"left_{kp}_x"] - df[f"right_{kp}_x"]) ** 2 + (df[f"left_{kp}_y"] - df[f"right_{kp}_y"]) ** 2)


def r_self(df):
    shoulder_px = _dist(df, "shoulder")
    wrist_px = _dist(df, "wrist")
    return pd.DataFrame({"distance_cm": df["distance_cm"].values, "R": (wrist_px / shoulder_px).values})


def curve_from_self(df):
    r = r_self(df)
    return r.groupby("distance_cm").agg(
        R_mean=("R", "mean"), R_std=("R", "std"), R_p95=("R", lambda s: s.quantile(ADAPTIVE_PCTL / 100))
    ).reset_index()


def top_n_filter(df):
    """Per frame, keep only the top-N (N=n_people_expected) detections by
    box_conf -- drops glass-reflection false positives (conf 0.25-0.37,
    real people ~0.94, cleanly separable)."""
    return (df.sort_values("box_conf", ascending=False)
              .groupby(["source_video", "frame_idx"], group_keys=False)
              .apply(lambda g: g.head(int(g["n_people_expected"].iloc[0]))))


def savefig(fig, name):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg, +.pdf)")


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {DATA_DIR} -- extract 1-6_csv.7z here first")
    if not SYNTH_DIR.is_dir():
        raise SystemExit(f"expected sibling 1-3 synthetic CSVs at {SYNTH_DIR} -- extract 1-3_csv.7z in the "
                          f"sibling 1-3_nano_quant_palms_together_synthetic_samples_0 folder first")
    if not YOLO11X_REAL_VIDEO_CSV.exists():
        raise SystemExit(f"expected {YOLO11X_REAL_VIDEO_CSV} -- should be bundled next to this script")
    CHARTS_DIR.mkdir(exist_ok=True)

    # --- REAL-video yolo11x target curve (top-N confidence filtered) ---
    x11_raw = pd.read_csv(YOLO11X_REAL_VIDEO_CSV)
    x11_filtered = top_n_filter(x11_raw)
    x11_curve = curve_from_self(x11_filtered)
    x11_curve["threshold"] = x11_curve["R_p95"]
    x11_mean = dict(zip(x11_curve["distance_cm"], x11_curve["R_mean"]))
    x11_std = dict(zip(x11_curve["distance_cm"], x11_curve["R_std"]))
    x11_thr = dict(zip(x11_curve["distance_cm"], x11_curve["threshold"]))
    print("yolo11x target curve (REAL-video, top-N filtered):")
    print(x11_curve[x11_curve["distance_cm"].isin(ALL_DISTANCES)].round(4).to_string(index=False))

    rvalue_rows, missrate_rows = [], []
    for variant in VARIANTS:
        synth_df = pd.read_csv(SYNTH_DIR / f"synth_{variant}.csv")
        own_curve = curve_from_self(synth_df)
        own_mean = dict(zip(own_curve["distance_cm"], own_curve["R_mean"]))
        own_std = dict(zip(own_curve["distance_cm"], own_curve["R_std"]))

        for source, distances in [("real", OVERLAP_DISTANCES), ("simulated", SIM_DISTANCES)]:
            df = pd.read_csv(DATA_DIR / f"{source}_{variant}.csv")
            df = df[df["distance_cm"].isin(distances)].copy()
            r = r_self(df)
            r = r[r["distance_cm"].isin(distances)].copy()

            thr = r["distance_cm"].map(x11_thr)
            R_corrected = r["distance_cm"].map(x11_mean) + (r["R"] - r["distance_cm"].map(own_mean)) * (
                r["distance_cm"].map(x11_std) / r["distance_cm"].map(own_std))

            for d in distances:
                sub = r[r["distance_cm"] == d]
                sub_corr = R_corrected[r["distance_cm"] == d]
                if len(sub) == 0:
                    continue
                rvalue_rows.append({
                    "variant": variant, "distance_cm": d, "source": source,
                    "R_raw_mean": sub["R"].mean(), "R_raw_std": sub["R"].std(),
                    "R_corrected_mean": sub_corr.mean(), "R_corrected_std": sub_corr.std(),
                })
                missrate_rows.append({
                    "variant": variant, "distance_cm": d, "source": source,
                    "miss_fixed": (sub["R"] >= 0.4).mean(),
                    "miss_raw_adaptive": (sub["R"] >= x11_thr[d]).mean(),
                    "miss_corrected_adaptive": (sub_corr >= x11_thr[d]).mean(),
                    "n": len(sub),
                })
        print(f"done: {variant}")

    rvalue_df = pd.DataFrame(rvalue_rows).sort_values(["variant", "distance_cm"])
    missrate_df = pd.DataFrame(missrate_rows).sort_values(["variant", "distance_cm"])
    rvalue_df.to_csv(DATA_DIR / "R_value_by_distance.csv", index=False)
    missrate_df.to_csv(DATA_DIR / "full_250_550_combined.csv", index=False)
    print(f"\nsaved R_value_by_distance.csv, full_250_550_combined.csv")

    # ============================================================
    # Fig 1-6-1: original (fixed 0.4) vs new (corrected R + yolo11x threshold)
    # ============================================================
    # Marker shape now identifies the variant (circle=FP32, triangle=FP16,
    # square=INT8(mixed)) and stays constant between original/new so the
    # same-variant pairing is visually obvious; filled=original, hollow
    # (white face, so the line reads as stopping at the marker) = new.
    CHART_SHAPE = {"fp32": "o", "fp16": "^", "modelopt_int8_excl_cv4": "s"}
    fig, ax = plt.subplots(figsize=(11, 6.2))
    handles_for_legend = {}
    for variant in CHART_VARIANTS:
        sub = missrate_df[missrate_df["variant"] == variant].sort_values("distance_cm")
        color = CHART_COLOR[variant]
        shape = CHART_SHAPE[variant]
        h1, = ax.plot(sub["distance_cm"], sub["miss_fixed"], color=color, linestyle="--", marker=shape,
                markersize=7, linewidth=1.6, label=f"{CHART_LABEL_SHORT[variant]} - original")
        h2, = ax.plot(sub["distance_cm"], sub["miss_corrected_adaptive"], color=color, linestyle="-", marker=shape,
                markersize=7, linewidth=1.8, markerfacecolor="white", label=f"{CHART_LABEL_SHORT[variant]} - new")
        handles_for_legend[(variant, "original")] = h1
        handles_for_legend[(variant, "new")] = h2
    ax.axvline(400, color="gray", linestyle=":", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(405, y_top - 0.10 * (y_top - ax.get_ylim()[0]), "real data | simulated ->", fontsize=13, color="gray")
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("miss rate (fraction of person-detections missed)")
    # 3 rows x 2 cols, each row = one variant's (original, new) pair --
    # column-major fill with ncol=2 puts the first half of the handle list
    # in column 1 and the second half in column 2.
    legend_order = [(v, role) for role in ("original", "new") for v in CHART_VARIANTS]
    handles = [handles_for_legend[k] for k in legend_order]
    labels = [f"{CHART_LABEL_SHORT[k[0]]} - {k[1]}" for k in legend_order]
    ax.legend(handles, labels, loc="upper left", ncol=2, fontsize=12)
    fig.tight_layout()
    savefig(fig, "1-6-1_r_correction_vs_distance_real_and_simulated")

    # Fig 1-6-2 (v1, synthetic-calibrated methodology) used to be generated
    # here. Retired: analyze_v3_real_calibration.py now produces the
    # canonical charts/1-6-2_... using real-video calibration instead -- see
    # v3_real_calibration_README.md for why (own_mean/own_std fit on
    # synthetic data was the main issue). rvalue_df above is still needed
    # for R_value_by_distance.csv; only the plotting code for
    # this specific figure was removed.

    print("\nDONE")


if __name__ == "__main__":
    main()
