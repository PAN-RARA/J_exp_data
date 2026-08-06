#!/usr/bin/env python3
"""
analyze_1_6.py
===============
Reproduces the Notion 1-6 numbers and charts (Fig 1-6-1, Fig 1-6-2) from the
per-detection CSVs in 1-6_csv.7z. Extract that archive here first (it
unpacks to "exported_charts/1-6/..." directly, matching the paths below).

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
    SYNTH_DIR = Path(__file__).parent.parent / "1-3_nano_quant_palms_together_synthetic_samples_0" / "1-3_csv"
Extract 1-3_csv.7z there first (sibling repo folder) if not already done.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SYNTH_DIR = ROOT.parent / "1-3_nano_quant_palms_together_synthetic_samples_0" / "1-3_csv"
REAL_DIR = ROOT / "exported_charts" / "1-6" / "real_video_results_filtered"
SIM_DIR = ROOT / "exported_charts" / "1-6" / "real_video_results_simulated_filtered"
YOLO11X_REAL_VIDEO_CSV = ROOT / "yolo11x-pose.real_video.csv"
CHARTS_DIR = ROOT / "charts"
K = 1.645
OVERLAP_DISTANCES = [250, 300, 350, 400]
SIM_DISTANCES = [275, 325, 375, 425, 450, 475, 500, 525, 550]
ALL_DISTANCES = sorted(OVERLAP_DISTANCES + SIM_DISTANCES)

VARIANTS = ["fp32", "fp16", "int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]
CHART_VARIANTS = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
CHART_COLOR = {"fp32": "#555555", "fp16": "#e07b39", "modelopt_int8_excl_cv4": "#4a9c6d"}
CHART_LABEL_SHORT = {"fp32": "FP32", "fp16": "FP16", "modelopt_int8_excl_cv4": "INT8(mixed)"}


def _dist(df, kp):
    return np.sqrt((df[f"left_{kp}_x"] - df[f"right_{kp}_x"]) ** 2 + (df[f"left_{kp}_y"] - df[f"right_{kp}_y"]) ** 2)


def r_self(df):
    shoulder_px = _dist(df, "shoulder")
    wrist_px = _dist(df, "wrist")
    return pd.DataFrame({"distance_cm": df["distance_cm"].values, "R": (wrist_px / shoulder_px).values})


def curve_from_self(df):
    r = r_self(df)
    return r.groupby("distance_cm").agg(R_mean=("R", "mean"), R_std=("R", "std")).reset_index()


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
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg)")


def main():
    if not REAL_DIR.is_dir() or not SIM_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {REAL_DIR} / {SIM_DIR} -- extract 1-6_csv.7z here first")
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
    x11_curve["threshold"] = x11_curve["R_mean"] + K * x11_curve["R_std"]
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

        for source, data_dir, distances in [("real", REAL_DIR, OVERLAP_DISTANCES), ("simulated", SIM_DIR, SIM_DISTANCES)]:
            df = pd.read_csv(data_dir / f"{variant}.csv")
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
    rvalue_df.to_csv(CHARTS_DIR / "real_video_1_6_R_value_by_distance.csv", index=False)
    missrate_df.to_csv(CHARTS_DIR / "real_video_1_6_full_250_550_combined.csv", index=False)
    print(f"\nsaved real_video_1_6_R_value_by_distance.csv, real_video_1_6_full_250_550_combined.csv")

    # ============================================================
    # Fig 1-6-1: original (fixed 0.4) vs new (corrected R + yolo11x threshold)
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for variant in CHART_VARIANTS:
        sub = missrate_df[missrate_df["variant"] == variant].sort_values("distance_cm")
        color = CHART_COLOR[variant]
        ax.plot(sub["distance_cm"], sub["miss_fixed"], "o--", color=color, alpha=0.85,
                label=f"{CHART_LABEL_SHORT[variant]} - original (fixed 0.4, beta_S6.py)")
        ax.plot(sub["distance_cm"], sub["miss_corrected_adaptive"], "^-", color=color,
                label=f"{CHART_LABEL_SHORT[variant]} - new (corrected R vs yolo11x threshold)")
    ax.axvline(400, color="gray", linestyle=":", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(405, y_top - 0.02 * (y_top - ax.get_ylim()[0]), "real data | simulated ->", fontsize=8, color="gray")
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("miss rate (fraction of person-detections missed)")
    ax.set_title("1-6: real human video (250-550cm) - original (fixed 0.4) vs new (1-3/1-5 correction) method\n"
                  "250-400cm = genuine real footage; 275-550cm = simulated via canvas-preserving downscale from real 250/300/350/400cm frames")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    fig.tight_layout()
    savefig(fig, "1-6-1_r_correction_vs_distance_real_and_simulated")

    # ============================================================
    # Fig 1-6-2: raw vs corrected R, vs REAL-video yolo11x target/threshold
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.plot(x11_curve["distance_cm"], x11_curve["R_mean"], "k-", linewidth=2, label="yolo11x-pose REAL-video (target)")
    ax.plot(x11_curve["distance_cm"], x11_curve["threshold"], "k:", linewidth=1.5, label="threshold (yolo11x mean+1.645std)")
    for variant in CHART_VARIANTS:
        sub = rvalue_df[rvalue_df["variant"] == variant].sort_values("distance_cm")
        color = CHART_COLOR[variant]
        ax.errorbar(sub["distance_cm"], sub["R_raw_mean"], yerr=sub["R_raw_std"], fmt="o--", color=color, alpha=0.85,
                    capsize=2, label=f"{CHART_LABEL_SHORT[variant]} - raw R")
        ax.errorbar(sub["distance_cm"], sub["R_corrected_mean"], yerr=sub["R_corrected_std"], fmt="^-", color=color,
                    capsize=2, label=f"{CHART_LABEL_SHORT[variant]} - corrected R")
    ax.axvline(400, color="gray", linestyle=":", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(405, y_top - 0.02 * (y_top - ax.get_ylim()[0]), "real data | simulated ->", fontsize=8, color="gray")
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("R value (wrist_dist / shoulder_dist)")
    ax.set_title("1-6: real human video R value, raw vs corrected, vs yolo11x target/threshold (250-550cm)\n"
                  "[REAL-video yolo11x target] 250-400cm = genuine real footage; 275-550cm = simulated via canvas-preserving downscale")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    fig.tight_layout()
    savefig(fig, "1-6-2_R_value_raw_vs_corrected_vs_distance")

    print("\nDONE")


if __name__ == "__main__":
    main()
