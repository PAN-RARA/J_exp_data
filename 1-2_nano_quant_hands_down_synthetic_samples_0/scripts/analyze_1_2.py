#!/usr/bin/env python3
"""
analyze_1_2.py
===============
Reproduces the Notion 1-2 numbers (per-keypoint OKS table + coefficient of
variation, shoulder/wrist OKS vs distance, shoulder/wrist OKS vs n_people)
directly from the per-detection CSVs in 1-2_csv.7z. Extract that archive
into a sibling folder named "1-2_csv" (i.e. next to this script) before
running. Self-contained -- no dependency on anything outside this repo.

Metric: OKS (Object Keypoint Similarity), the standard COCO keypoint metric,
computed self-referentially (each variant vs its OWN FP32 run, not an
external ground truth):
    OKS_i = exp( -d_i^2 / (2 * s^2 * sigma_i^2) )
d_i = pixel distance between this variant's keypoint i and the FP32
reference's keypoint i (same image, same person). s = object scale =
sqrt(box_w * box_h) of the FP32 reference detection. sigma_i = official
COCO per-keypoint constant (see COCO_SIGMA below). 1.0 = identical, lower =
more quantization damage. Aggregation order matters: OKS is computed per
detection per keypoint FIRST, then averaged (per-image, then across images)
-- never average raw pixel distance first and exponentiate after, since
exp() is nonlinear and that ordering gives different (and here, meaningless)
numbers.

Cross-variant matching: person_idx in the raw CSVs is NOT a stable identity
across precision runs (two runs on the same image aren't guaranteed to list
detections in the same order). Detections are matched to the FP32 reference
via Hungarian assignment on box IoU (scipy.optimize.linear_sum_assignment,
globally optimal per image, not greedy/rank-based) -- matches below
MIN_IOU are rejected rather than forced, since a low-IoU "best available"
pair likely isn't the same person.

Failure-mode filter: detections where the variant's shoulder width has
collapsed to <80% of the FP32 reference's shoulder width are excluded --
these are rare, INT8-specific keypoint-collapse failures (not typical
quantization noise), and including them would blow up the aggregate
statistics without reflecting general quantization behavior. The excluded
count is printed per variant so this is never silent.

Also reproduces the three Notion charts (Fig 1-2-1/2/3) into a "charts/"
folder next to this script, using the same layout as the original
analysis_scripts/oks_1_2_full.py (dot+errbar plot for the 17-keypoint
comparison, line plots for the distance/n_people sweeps).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

DATA_DIR = Path(__file__).resolve().parent.parent / "1-2_csv"
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
VARIANTS = ["fp16", "int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]
LABELS = {
    "fp16": "FP16", "int8": "INT8(legacy)", "modelopt_int8": "INT8(ModelOpt full)",
    "modelopt_int8_excl_cv4": "INT8(ModelOpt mixed→FP16)",
    "modelopt_int8_excl_cv4_fp32": "INT8(ModelOpt mixed→FP32)",
}
PLOT_VARIANTS = ["int8", "modelopt_int8", "modelopt_int8_excl_cv4", "modelopt_int8_excl_cv4_fp32"]  # excludes fp16, matches Fig 1-2-1
SHOULDER_RATIO_THRESHOLD = 0.80

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
VARIANT_COLOR = {"fp16": "#4C72B0", "int8": "#C44E52", "modelopt_int8": "#DD8452",
                  "modelopt_int8_excl_cv4": "#55A868", "modelopt_int8_excl_cv4_fp32": "#8172B2"}
BOX_COLS = ["box_x1", "box_y1", "box_x2", "box_y2"]
MIN_IOU = 0.3


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(ref_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """Per filename, Hungarian-match ref_df detections to other_df
    detections by box IoU. Groups other_df by filename ONCE up front
    (dict lookup) rather than re-filtering the whole frame per image --
    the naive "other_df[other_df.filename==f] inside a per-image loop"
    approach is O(n_files^2) and takes 10+ minutes at ~6500 files; this is
    seconds."""
    other_groups = dict(list(other_df.groupby("filename")))
    rows = []
    for filename, ref_g in ref_df.groupby("filename"):
        other_g = other_groups.get(filename)
        if other_g is None or len(ref_g) == 0:
            continue
        iou = box_iou_matrix(ref_g[BOX_COLS].to_numpy(), other_g[BOX_COLS].to_numpy())
        row_ind, col_ind = linear_sum_assignment(-iou)
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= MIN_IOU:
                rows.append({"filename": filename, "ref_idx": ref_g.index[r], "other_idx": other_g.index[c]})
    return pd.DataFrame(rows, columns=["filename", "ref_idx", "other_idx"])


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


def matched_with_shoulder_ratio(fp32_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    """One row per matched (fp32, variant) detection: filename, distance_cm,
    n_people, shoulder_ratio (for the failure-mode filter), and per-keypoint
    OKS for every keypoint in KP_ORDER."""
    m = match_detections(fp32_df, variant_df)
    ref = fp32_df.loc[m["ref_idx"]].reset_index(drop=True)
    oth = variant_df.loc[m["other_idx"]].reset_index(drop=True)
    s = obj_scale(ref)

    def pdist(a_x, a_y, b_x, b_y):
        return np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)

    shoulder_fp32 = pdist(ref.left_shoulder_x, ref.left_shoulder_y, ref.right_shoulder_x, ref.right_shoulder_y)
    shoulder_var = pdist(oth.left_shoulder_x, oth.left_shoulder_y, oth.right_shoulder_x, oth.right_shoulder_y)

    out = {
        "filename": m["filename"].values,
        "distance_cm": ref["distance_cm"].values,
        "n_people": ref["n_people_expected"].values,
        "shoulder_ratio": (shoulder_var / shoulder_fp32).values,
    }
    for kp in KP_ORDER:
        out[kp] = oks(ref, oth, kp, s).values
    return pd.DataFrame(out)


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {DATA_DIR} -- extract 1-2_csv.7z here first")

    print("loading csvs...", flush=True)
    dfs = load_csvs()
    print(f"  fp32: {len(dfs['fp32'])} rows, distances={sorted(dfs['fp32'].distance_cm.unique())}, "
          f"n_people={sorted(dfs['fp32'].n_people_expected.unique())}")

    detections = {}
    for v in VARIANTS:
        print(f"matching {v}...", flush=True)
        det = matched_with_shoulder_ratio(dfs["fp32"], dfs[v])
        excluded = (det["shoulder_ratio"] < SHOULDER_RATIO_THRESHOLD).sum()
        print(f"  {v}: excluded {excluded}/{len(det)} detections (shoulder_ratio < {SHOULDER_RATIO_THRESHOLD})")
        detections[v] = det[det["shoulder_ratio"] >= SHOULDER_RATIO_THRESHOLD].reset_index(drop=True)

    CHARTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 100)
    print("1. Per-keypoint mean OKS (per-image mean, then across images) + coefficient of variation across the 17 keypoints")
    print("=" * 100)
    kp_mean_by_variant, kp_sem_by_variant = {}, {}
    for v in PLOT_VARIANTS:
        det = detections[v]
        per_image = det.groupby("filename")[KP_ORDER].mean()
        n = len(per_image)
        kp_means = per_image.mean().reindex(KP_ORDER)
        kp_sems = (per_image.std(ddof=1) / np.sqrt(n)).reindex(KP_ORDER)
        kp_mean_by_variant[v] = kp_means
        kp_sem_by_variant[v] = kp_sems
        cv = kp_means.std(ddof=0) / kp_means.mean()
        print(f"-- {LABELS[v]} -- (17-keypoint CV = {cv:.4f})")
        for kp in KP_ORDER:
            print(f"  {kp:<16} {kp_means[kp]:.4f}")

    x = np.arange(len(KP_ORDER))
    offsets = np.linspace(-0.24, 0.24, len(PLOT_VARIANTS))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, v in enumerate(PLOT_VARIANTS):
        ax.errorbar(x + offsets[i], kp_mean_by_variant[v].values, yerr=kp_sem_by_variant[v].values,
                    fmt="o", markersize=6, capsize=3, color=VARIANT_COLOR[v], label=LABELS[v])
    ax.set_xticks(x)
    ax.set_xticklabels(KP_ORDER, rotation=45, ha="right")
    ax.set_ylabel("mean OKS vs FP32 (1.0 = identical), ±1 SEM")
    ax.set_title("1-2 (hands-down): per-keypoint OKS vs FP32, 4 variants")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.4)
    for name, start, end in GROUP_BOUNDS:
        if start > 0:
            ax.axvline(start - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.text((start + end - 1) / 2, 1.005, name, ha="center", fontsize=10, color="dimgray")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(str(CHARTS_DIR / "oks_1_2_errbar.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / "oks_1_2_errbar.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / "oks_1_2_errbar.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / 'oks_1_2_errbar.png'} (+.svg, +.pdf)")

    print("\n" + "=" * 100)
    print("2. Shoulder/wrist mean OKS by distance (all variants)")
    print("=" * 100)
    by_distance = {}
    for v in VARIANTS:
        det = detections[v]
        det = det.assign(shoulder=det[["left_shoulder", "right_shoulder"]].mean(axis=1),
                          wrist=det[["left_wrist", "right_wrist"]].mean(axis=1))
        per_image = det.groupby("filename").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"),
                                                  distance_cm=("distance_cm", "first"))
        by_distance[v] = per_image.groupby("distance_cm").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"))
    wrist_table = pd.DataFrame({LABELS[v]: by_distance[v]["wrist"] for v in VARIANTS})
    print("-- wrist --")
    print(wrist_table.round(4).to_string())
    shoulder_table = pd.DataFrame({LABELS[v]: by_distance[v]["shoulder"] for v in VARIANTS})
    print("-- shoulder --")
    print(shoulder_table.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v in VARIANTS:
        d = by_distance[v]
        ax.plot(d.index, d["shoulder"], color=VARIANT_COLOR[v], linestyle="-", marker="o", markersize=4,
                label=f"{LABELS[v]} – shoulder")
        ax.plot(d.index, d["wrist"], color=VARIANT_COLOR[v], linestyle="--", marker="s", markersize=4,
                label=f"{LABELS[v]} – wrist")
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("mean OKS vs FP32 (1.0 = identical)")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("1-2 (hands-down): shoulder & wrist OKS vs distance, 5 variants")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(str(CHARTS_DIR / "oks_1_2_shoulder_wrist_vs_distance.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / "oks_1_2_shoulder_wrist_vs_distance.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / "oks_1_2_shoulder_wrist_vs_distance.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / 'oks_1_2_shoulder_wrist_vs_distance.png'} (+.svg, +.pdf)")

    print("\n" + "=" * 100)
    print("3. Shoulder/wrist mean OKS by n_people (all variants)")
    print("=" * 100)
    by_npeople = {}
    for v in VARIANTS:
        det = detections[v]
        det = det.assign(shoulder=det[["left_shoulder", "right_shoulder"]].mean(axis=1),
                          wrist=det[["left_wrist", "right_wrist"]].mean(axis=1))
        per_image = det.groupby("filename").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"),
                                                  n_people=("n_people", "first"))
        by_npeople[v] = per_image.groupby("n_people").agg(shoulder=("shoulder", "mean"), wrist=("wrist", "mean"))
    wrist_table2 = pd.DataFrame({LABELS[v]: by_npeople[v]["wrist"] for v in VARIANTS})
    print("-- wrist --")
    print(wrist_table2.round(4).to_string())
    shoulder_table2 = pd.DataFrame({LABELS[v]: by_npeople[v]["shoulder"] for v in VARIANTS})
    print("-- shoulder --")
    print(shoulder_table2.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v in VARIANTS:
        d = by_npeople[v]
        ax.plot(d.index, d["shoulder"], color=VARIANT_COLOR[v], linestyle="-", marker="o", markersize=4,
                label=f"{LABELS[v]} – shoulder")
        ax.plot(d.index, d["wrist"], color=VARIANT_COLOR[v], linestyle="--", marker="s", markersize=4,
                label=f"{LABELS[v]} – wrist")
    ax.set_xlabel("n_people")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylabel("mean OKS vs FP32 (1.0 = identical)")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("1-2 (hands-down): shoulder & wrist OKS vs n_people, 5 variants")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(str(CHARTS_DIR / "oks_1_2_shoulder_wrist_vs_npeople.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / "oks_1_2_shoulder_wrist_vs_npeople.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / "oks_1_2_shoulder_wrist_vs_npeople.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / 'oks_1_2_shoulder_wrist_vs_npeople.png'} (+.svg, +.pdf)")

    print("\nDONE")


if __name__ == "__main__":
    main()
