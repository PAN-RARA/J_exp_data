#!/usr/bin/env python3
"""
analyze_1_3.py
===============
Reproduces the Notion 1-3 numbers and charts (Fig 1-3-1 through 1-3-7) from
the per-detection CSVs in 1-3_csv.7z. Extract that archive into a sibling
folder named "1-3_csv" (i.e. next to this script) before running.

Covers three analyses:
  1. OKS (Object Keypoint Similarity), same self-referential metric and
     Hungarian box-IoU matching as analyze_1_2.py -- see that script's
     docstring for the full metric rationale. Produces the 17-keypoint
     errbar chart (Fig 1-3-1) and shoulder/wrist-vs-distance/n_people
     charts (Fig 1-3-2/3).
  2. R value (wrist_dist / shoulder_dist -- the actual gesture-detection
     ratio, computed directly from a variant's own keypoints, no OKS
     involved since OKS discards the raw coordinates R needs). Produces
     R-vs-distance with the 0.4/0.5 candidate thresholds (Fig 1-3-4) and
     R-vs-n_people (Fig 1-3-5).
  3. Distance-adaptive correction: v8n-pose's own R value runs systematically
     higher than yolo11x-pose's (a much larger/more-accurate reference model,
     69.5 vs 69.2 COCO pose mAP50-95 per Ultralytics' official benchmarks,
     with FEWER params than v8x) at long range -- v8n's own capacity limit,
     not quantization noise, since it shows up almost identically on FP16
     and INT8(mixed). The correction rescales each detection's R using a
     13-bin (one per true synthetic distance, 250-550cm) mean+std transform
     onto yolo11x's distribution:
         R_corrected = v11x_mean(x) + (R_raw - v8n_mean(x)) * (v11x_std(x) / v8n_std(x))
     where x = the detection's own shoulder_px (the only distance proxy
     available at deployment, no depth sensor). Produces the raw-vs-corrected
     R curve (Fig 1-3-6) and the corresponding miss-rate-vs-distance chart
     (Fig 1-3-7, R >= 0.4 treated as a missed gesture). legacy INT8 is
     excluded from the correction (only FP16/ModelOpt family) since the
     legacy TensorRT calibrator toolchain is being phased out.

     Also reproduces the "adaptive threshold only" comparison point used in
     the Notion 結論 (moving the threshold per-distance-bin via
     R_mean+1.645*R_std on yolo11x, WITHOUT rescaling R itself) -- printed
     as text only (no current Notion figure), to keep that conclusion
     sentence's numbers reproducible too.

Requires an external reference file this repo does NOT include (it's
shared across 1-3/1-5/1-6, ~10MB, not worth duplicating per-experiment):
yolo11x-pose's own FP32 inference on the same synthetic images. Point
YOLO11X_FP32_CSV at wherever that file lives, or ask for a copy of
pose_quant_env/xtier_fp32_pray_results/yolo11x-pose.synth_fp32.csv.
Everything else is self-contained.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Times New Roman"
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

DATA_DIR = Path(__file__).parent / "1-3_csv"
CHARTS_DIR = Path(__file__).parent / "charts"
YOLO11X_FP32_CSV = Path(r"C:\Users\user\pose_quant_env\xtier_fp32_pray_results\yolo11x-pose.synth_fp32.csv")

VARIANTS = ["fp16", "int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]
CORRECTION_VARIANTS = ["fp16", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]  # excludes legacy INT8
LABELS = {
    "fp16": "FP16", "int8": "INT8(legacy)", "modelopt_int8": "INT8(ModelOpt full)",
    "modelopt_int8_excl_cv4": "INT8(ModelOpt mixed→FP16)",
    "modelopt_int8_excl_cv4_fp32": "INT8(ModelOpt mixed→FP32)",
}
VARIANT_COLOR = {"fp16": "#4C72B0", "int8": "#C44E52", "modelopt_int8": "#DD8452",
                  "modelopt_int8_excl_cv4": "#55A868", "modelopt_int8_excl_cv4_fp32": "#8172B2"}
# IEEE print figures fall back to grayscale -- pair each variant with its own
# marker shape so series stay distinguishable by shape alone, not just color.
VARIANT_MARKER = {"fp16": "o", "int8": "s", "modelopt_int8": "^",
                   "modelopt_int8_excl_cv4": "D", "modelopt_int8_excl_cv4_fp32": "v"}
PLOT_VARIANTS = ["int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]  # excludes fp16, matches Fig 1-3-1
SHOULDER_RATIO_THRESHOLD = 0.80
K = 1.645  # ~95% one-sided coverage for the adaptive-threshold comparison

COCO_SIGMA = {
    "nose": 0.026, "left_eye": 0.025, "right_eye": 0.025,
    "left_ear": 0.035, "right_ear": 0.035,
    "left_shoulder": 0.079, "right_shoulder": 0.079,
    "left_elbow": 0.072, "right_elbow": 0.072,
    "left_wrist": 0.062, "right_wrist": 0.062,
    "left_hip": 0.107, "right_hip": 0.107,
    "left_knee": 0.087, "right_knee": 0.087,
    "left_ankle": 0.089, "right_ankle": 0.089,
}
KP_ORDER = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "left_elbow", "right_elbow", "left_knee", "right_knee",
            "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
GROUP_BOUNDS = [("head", 0, 5), ("trunk", 5, 9), ("mid-limb", 9, 13), ("extremity", 13, 17)]
BOX_COLS = ["box_x1", "box_y1", "box_x2", "box_y2"]
MIN_IOU = 0.3


# ============================================================
# shared: fast Hungarian box-IoU matching (see analyze_1_2.py docstring for
# why matching is needed and why this beats the naive per-image full-frame
# filter -- same fix, ~10min -> seconds for ~6500 files)
# ============================================================
def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(ref_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    other_groups = dict(list(other_df.groupby("filename")))
    rows = []
    for filename, ref_g in ref_df.groupby("filename"):
        other_g = other_groups.get(filename)
        if other_g is None or len(ref_g) == 0:
            continue
        iou = box_iou_matrix(ref_g[BOX_COLS].to_numpy(), other_g[BOX_COLS].to_numpy())
        row_ind, col_ind = linear_sum_assignment(-iou)
        rows.extend({"filename": filename, "ref_idx": ref_g.index[r], "other_idx": other_g.index[c]}
                    for r, c in zip(row_ind, col_ind) if iou[r, c] >= MIN_IOU)
    return pd.DataFrame(rows, columns=["filename", "ref_idx", "other_idx"])


def pdist(ax, ay, bx, by):
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def obj_scale(ref: pd.DataFrame) -> pd.Series:
    return np.sqrt((ref["box_x2"] - ref["box_x1"]) * (ref["box_y2"] - ref["box_y1"]))


def oks(ref: pd.DataFrame, oth: pd.DataFrame, kp: str, s: pd.Series) -> pd.Series:
    d2 = (ref[f"{kp}_x"] - oth[f"{kp}_x"]) ** 2 + (ref[f"{kp}_y"] - oth[f"{kp}_y"]) ** 2
    return np.exp(-d2 / (2 * (s ** 2) * (COCO_SIGMA[kp] ** 2)))


def load_csvs() -> dict[str, pd.DataFrame]:
    dfs = {"fp32": pd.read_csv(DATA_DIR / "synth_fp32.csv")}
    for v in VARIANTS:
        dfs[v] = pd.read_csv(DATA_DIR / f"synth_{v}.csv")
    return dfs


def per_detection(fp32_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    """One row per matched detection: filename, distance_cm, n_people,
    shoulder_ratio (failure-mode filter), shoulder_px (distance proxy for
    the correction curve), R (this variant's own wrist/shoulder ratio),
    and per-keypoint OKS for every keypoint in KP_ORDER."""
    m = match_detections(fp32_df, variant_df)
    ref = fp32_df.loc[m["ref_idx"]].reset_index(drop=True)
    oth = variant_df.loc[m["other_idx"]].reset_index(drop=True)
    s = obj_scale(ref)

    shoulder_fp32 = pdist(ref.left_shoulder_x, ref.left_shoulder_y, ref.right_shoulder_x, ref.right_shoulder_y)
    shoulder_var = pdist(oth.left_shoulder_x, oth.left_shoulder_y, oth.right_shoulder_x, oth.right_shoulder_y)
    wrist_var = pdist(oth.left_wrist_x, oth.left_wrist_y, oth.right_wrist_x, oth.right_wrist_y)

    out = {
        "filename": m["filename"].values,
        "distance_cm": ref["distance_cm"].values,
        "n_people": ref["n_people_expected"].values,
        "shoulder_ratio": (shoulder_var / shoulder_fp32).values,
        "shoulder_px": shoulder_var.values,
        "R": (wrist_var / shoulder_var).values,
    }
    for kp in KP_ORDER:
        out[kp] = oks(ref, oth, kp, s).values
    return pd.DataFrame(out)


def self_R(df: pd.DataFrame) -> pd.DataFrame:
    """R and shoulder_px computed directly from a model's own keypoints, no
    matching needed (used for the yolo11x-pose reference, which is only
    read for its own R distribution, never matched against v8n)."""
    shoulder_px = pdist(df.left_shoulder_x, df.left_shoulder_y, df.right_shoulder_x, df.right_shoulder_y)
    wrist_px = pdist(df.left_wrist_x, df.left_wrist_y, df.right_wrist_x, df.right_wrist_y)
    return pd.DataFrame({
        "filename": df["filename"].values,
        "distance_cm": df["distance_cm"].values,
        "shoulder_px": shoulder_px.values,
        "R": (wrist_px / shoulder_px).values,
    })


def savefig(fig, name):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg, +.pdf)")


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {DATA_DIR} -- extract 1-3_csv.7z here first")
    CHARTS_DIR.mkdir(exist_ok=True)

    print("loading csvs...", flush=True)
    dfs = load_csvs()
    print(f"  fp32: {len(dfs['fp32'])} rows, distances={sorted(dfs['fp32'].distance_cm.unique())}, "
          f"n_people={sorted(dfs['fp32'].n_people_expected.unique())}")

    detections = {}
    for v in VARIANTS:
        print(f"matching {v}...", flush=True)
        det = per_detection(dfs["fp32"], dfs[v])
        excluded = (det["shoulder_ratio"] < SHOULDER_RATIO_THRESHOLD).sum()
        print(f"  {v}: excluded {excluded}/{len(det)} detections (shoulder_ratio < {SHOULDER_RATIO_THRESHOLD})")
        detections[v] = det[det["shoulder_ratio"] >= SHOULDER_RATIO_THRESHOLD].reset_index(drop=True)

    # ============================================================
    # 1. OKS: 17-keypoint table + CV, shoulder/wrist vs distance/n_people
    # ============================================================
    print("\n" + "=" * 100)
    print("1a. Per-keypoint mean OKS + coefficient of variation across the 17 keypoints (Fig 1-3-1)")
    print("=" * 100)
    kp_mean, kp_sem = {}, {}
    for v in PLOT_VARIANTS:
        det = detections[v]
        per_image = det.groupby("filename")[KP_ORDER].mean()
        n = len(per_image)
        kp_mean[v] = per_image.mean().reindex(KP_ORDER)
        kp_sem[v] = (per_image.std(ddof=1) / np.sqrt(n)).reindex(KP_ORDER)
        cv = kp_mean[v].std(ddof=0) / kp_mean[v].mean()
        print(f"-- {LABELS[v]} -- (17-keypoint CV = {cv:.4f})")
        for kp in KP_ORDER:
            print(f"  {kp:<16} {kp_mean[v][kp]:.4f}")

    x = np.arange(len(KP_ORDER))
    offsets = np.linspace(-0.24, 0.24, len(PLOT_VARIANTS))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, v in enumerate(PLOT_VARIANTS):
        ax.errorbar(x + offsets[i], kp_mean[v].values, yerr=kp_sem[v].values,
                    fmt=VARIANT_MARKER[v], markersize=6, capsize=3, color=VARIANT_COLOR[v], label=LABELS[v])
    ax.set_xticks(x)
    ax.set_xticklabels(KP_ORDER, rotation=45, ha="right")
    ax.set_ylabel("mean OKS vs FP32 (1.0 = identical), ±1 SEM")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.4)
    for name, start, end in GROUP_BOUNDS:
        if start > 0:
            ax.axvline(start - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.text((start + end - 1) / 2, 1.005, name, ha="center", fontsize=10, color="dimgray")
    ax.legend(fontsize=9, loc="lower right", markerscale=0.65)
    fig.tight_layout()
    savefig(fig, "oks_1_3_errbar")

    print("\n" + "=" * 100)
    print("1b. Shoulder/wrist mean OKS by distance & n_people (Fig 1-3-2/3)")
    print("=" * 100)
    by_distance, by_npeople = {}, {}
    for v in VARIANTS:
        det = detections[v].assign(
            shoulder=detections[v][["left_shoulder", "right_shoulder"]].mean(axis=1),
            wrist=detections[v][["left_wrist", "right_wrist"]].mean(axis=1),
        )
        per_image = det.groupby("filename").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"),
                                                  distance_cm=("distance_cm", "first"), n_people=("n_people", "first"))
        by_distance[v] = per_image.groupby("distance_cm").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"))
        by_npeople[v] = per_image.groupby("n_people").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"))
    print("-- wrist by distance --")
    print(pd.DataFrame({LABELS[v]: by_distance[v]["wrist"] for v in VARIANTS}).round(4).to_string())
    print("-- wrist by n_people --")
    print(pd.DataFrame({LABELS[v]: by_npeople[v]["wrist"] for v in VARIANTS}).round(4).to_string())

    for label, by_x, xlabel, fname, xticks in [
        ("distance", by_distance, "distance (cm)", "oks_1_3_shoulder_wrist_vs_distance", None),
        ("n_people", by_npeople, "n_people", "oks_1_3_shoulder_wrist_vs_npeople", [1, 2, 3, 4, 5]),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for v in VARIANTS:
            d = by_x[v]
            # linestyle carries the shoulder/wrist role, marker carries the
            # variant identity -- keeps both axes readable in grayscale.
            ax.plot(d.index, d["shoulder"], color=VARIANT_COLOR[v], linestyle="-", marker=VARIANT_MARKER[v], markersize=5,
                    label=f"{LABELS[v]} – shoulder")
            ax.plot(d.index, d["wrist"], color=VARIANT_COLOR[v], linestyle="--", marker=VARIANT_MARKER[v], markersize=5,
                    markerfacecolor="none", label=f"{LABELS[v]} – wrist")
        ax.set_xlabel(xlabel)
        if xticks:
            ax.set_xticks(xticks)
        ax.set_ylabel("mean OKS vs FP32 (1.0 = identical)")
        ax.set_ylim(0.7, 1.02)
        ax.legend(fontsize=7.5, loc="center right")
        fig.tight_layout()
        savefig(fig, fname)

    # ============================================================
    # 2. R value: vs distance (with 0.4/0.5 threshold lines), vs n_people
    # ============================================================
    print("\n" + "=" * 100)
    print("2. R value (wrist_dist/shoulder_dist) by distance & n_people (Fig 1-3-4/5)")
    print("=" * 100)
    R_by_distance, R_by_npeople = {}, {}
    for v in VARIANTS:
        det = detections[v]
        per_image = det.groupby("filename").agg(R=("R", "mean"), distance_cm=("distance_cm", "first"), n_people=("n_people", "first"))
        R_by_distance[v] = per_image.groupby("distance_cm")["R"].mean()
        R_by_npeople[v] = per_image.groupby("n_people")["R"].mean()
    print("-- R by distance --")
    print(pd.DataFrame({LABELS[v]: R_by_distance[v] for v in VARIANTS}).round(4).to_string())
    print("-- R by n_people --")
    print(pd.DataFrame({LABELS[v]: R_by_npeople[v] for v in VARIANTS}).round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v in VARIANTS:
        d = R_by_distance[v]
        ax.plot(d.index, d.values, color=VARIANT_COLOR[v], marker="o", markersize=4, label=LABELS[v])
    ax.axhline(0.4, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("R (wrist_dist / shoulder_dist)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "r_vs_distance_thresholds_1_3")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v in VARIANTS:
        d = R_by_npeople[v]
        ax.plot(d.index, d.values, color=VARIANT_COLOR[v], marker="o", markersize=4, label=LABELS[v])
    ax.set_xlabel("n_people")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylabel("R (wrist_dist / shoulder_dist)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    savefig(fig, "r_vs_npeople_1_3")

    print("\n-- miss rate (R >= threshold), per-person, thresholds 0.4 and 0.5 --")
    for thresh in (0.4, 0.5):
        print(f"  threshold {thresh}:")
        for v in VARIANTS:
            det = detections[v]
            miss = det.groupby("distance_cm")["R"].apply(lambda s: (s >= thresh).mean())
            print(f"    {LABELS[v]:<28}", " ".join(f"{d}:{100*r:.0f}%" for d, r in miss.items()))

    # ============================================================
    # 3. Distance-adaptive correction (v8n-mixed -> yolo11x-pose FP32 rescale)
    # ============================================================
    if not YOLO11X_FP32_CSV.is_file():
        print(f"\n[skipped] correction analysis needs {YOLO11X_FP32_CSV} (external reference, not part of this repo)")
        print("\nDONE (partial -- OKS and R sections complete, correction section skipped)")
        return

    print("\n" + "=" * 100)
    print("3. Distance-adaptive correction: v8n-mixed -> yolo11x-pose FP32 rescale (Fig 1-3-6/7)")
    print("=" * 100)
    yolo11x_df = pd.read_csv(YOLO11X_FP32_CSV)
    x11_self = self_R(yolo11x_df)
    x11_stats = x11_self.groupby("distance_cm").agg(R_mean=("R", "mean"), R_std=("R", "std")).reset_index()
    # separate copy WITH yolo11x's own shoulder_mid -- needed only for the 3b
    # adaptive-threshold-only comparison below, which (per the original
    # adaptive_threshold_1_3.py) calibrates its threshold curve on yolo11x's
    # OWN shoulder_px axis, unlike the ratio-correction curve just below
    # (Fig 1-3-6/7) which deliberately uses v8n-mixed's shoulder_px instead
    # (that's the axis actually available at deployment for looking up a v8n
    # detection's own correction).
    x11_stats_own_axis = x11_self.groupby("distance_cm").agg(
        shoulder_mid=("shoulder_px", "mean"), R_mean=("R", "mean"), R_std=("R", "std")
    ).reset_index()

    mixed_self = self_R(dfs["modelopt_int8_excl_cv4"])
    mixed_stats = mixed_self.groupby("distance_cm").agg(shoulder_mid=("shoulder_px", "mean"), R_mean=("R", "mean"), R_std=("R", "std")).reset_index()
    curve = mixed_stats.merge(x11_stats, on="distance_cm", suffixes=("_mixed", "_x11")).sort_values("shoulder_mid")
    print("Correction curve (13 true-distance bins, x-axis = v8n-mixed's own shoulder_px):")
    print(curve.round(4).to_string(index=False))

    def interp(x_new, col):
        return np.interp(x_new, curve["shoulder_mid"], curve[col])

    def correct_R(shoulder_px, R_raw):
        return interp(shoulder_px, "R_mean_x11") + (R_raw - interp(shoulder_px, "R_mean_mixed")) * (
            interp(shoulder_px, "R_std_x11") / interp(shoulder_px, "R_std_mixed"))

    corr_det = {}
    for v in CORRECTION_VARIANTS:
        d = detections[v].copy()
        d["R_corrected"] = correct_R(d["shoulder_px"].values, d["R"].values)
        d["miss_raw"] = d["R"] >= 0.4
        d["miss_corrected"] = d["R_corrected"] >= 0.4
        corr_det[v] = d

    print("\n-- miss rate (R raw vs fixed 0.4) by distance --")
    print(pd.DataFrame({LABELS[v]: corr_det[v].groupby("distance_cm")["miss_raw"].mean() for v in CORRECTION_VARIANTS}).round(3).to_string())
    print("\n-- miss rate (R corrected vs fixed 0.4) by distance --")
    print(pd.DataFrame({LABELS[v]: corr_det[v].groupby("distance_cm")["miss_corrected"].mean() for v in CORRECTION_VARIANTS}).round(3).to_string())

    # only 2 variants here -- role (raw/corrected) already carries marker
    # shape + linestyle, so give the 2nd variant hollow markers as a 3rd,
    # independent grayscale-safe cue on top of color.
    fill = {"fp16": None, "modelopt_int8_excl_cv4": "none"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for v in ["fp16", "modelopt_int8_excl_cv4"]:
        d = corr_det[v]
        agg_raw = d.groupby("distance_cm")["R"].mean()
        agg_corr = d.groupby("distance_cm")["R_corrected"].agg(["mean", "std"])
        ax.plot(agg_raw.index, agg_raw.values, color=VARIANT_COLOR[v], linestyle="--", marker="s", markersize=6,
                markerfacecolor=fill[v], label=f"{LABELS[v]} – raw")
        ax.errorbar(agg_corr.index, agg_corr["mean"], yerr=agg_corr["std"], color=VARIANT_COLOR[v],
                    linestyle="-", marker="o", markersize=6, markerfacecolor=fill[v], capsize=3, label=f"{LABELS[v]} – corrected")
    ax.plot(x11_stats["distance_cm"], x11_stats["R_mean"], color="tab:blue", linestyle="-", linewidth=2, alpha=0.5, label="yolo11x-pose FP32 target")
    threshold_curve = x11_stats["R_mean"] + K * x11_stats["R_std"]
    ax.plot(x11_stats["distance_cm"], threshold_curve, color="black", linestyle=":", linewidth=1.5, label=f"threshold (v11x mean+{K}×std)")
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("R (wrist_dist / shoulder_dist)")
    # extra headroom above the threshold curve so the legend has clear space
    # instead of sitting on top of it
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
    ax.legend(fontsize=9, loc="upper left", labelspacing=0.8, handlelength=2.5, handletextpad=0.8, borderpad=0.8)
    fig.tight_layout()
    savefig(fig, "1-3-6_R_value_raw_vs_corrected")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v in ["fp16", "modelopt_int8_excl_cv4"]:
        d = corr_det[v]
        agg = d.groupby("distance_cm").agg(miss_raw=("miss_raw", "mean"), miss_corr=("miss_corrected", "mean"))
        ax.plot(agg.index, agg["miss_raw"], color=VARIANT_COLOR[v], linestyle="--", marker="s", markersize=4,
                markerfacecolor=fill[v], label=f"{LABELS[v]} – raw R, fixed 0.4")
        ax.plot(agg.index, agg["miss_corr"], color=VARIANT_COLOR[v], linestyle="-", marker="o", markersize=4,
                markerfacecolor=fill[v], label=f"{LABELS[v]} – corrected R, fixed 0.4")
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("miss rate (fraction of individual person-detections missed)")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.tight_layout()
    savefig(fig, "1-3-7_miss_rate_raw_vs_corrected")

    # --- bonus: adaptive-threshold-ONLY comparison (moves the threshold,
    # doesn't rescale R) -- backs the 結論's "只換門檻" sentence, no current
    # Notion figure for this, text output only ---
    print("\n" + "=" * 100)
    print("3b. Adaptive-threshold-ONLY comparison (no R rescale) -- backs the 結論 “只換門檻” sentence")
    print("=" * 100)
    bin_stats = x11_stats_own_axis.sort_values("shoulder_mid").copy()
    bin_stats["threshold"] = bin_stats["R_mean"] + K * bin_stats["R_std"]

    def adaptive_threshold(shoulder_px):
        return np.interp(shoulder_px, bin_stats["shoulder_mid"], bin_stats["threshold"])

    adaptive_results = {}
    for v in CORRECTION_VARIANTS:
        d = detections[v].copy()
        d["miss_adaptive"] = d["R"] >= adaptive_threshold(d["shoulder_px"].values)
        adaptive_results[v] = d.groupby("distance_cm")["miss_adaptive"].mean()
    print(pd.DataFrame({LABELS[v]: adaptive_results[v] for v in CORRECTION_VARIANTS}).round(3).to_string())

    print("\nDONE")


if __name__ == "__main__":
    main()
