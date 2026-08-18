#!/usr/bin/env python3
"""
analyze_1_4.py
===============
Reproduces the Notion 1-4 numbers and charts (Fig 1-4-1 through 1-4-5) from
the per-detection CSVs in 1-4_csv.7z. Extract that archive into a sibling
folder named "1-4_csv" (i.e. next to this script) before running -- it
extracts to one subfolder per model (e.g. "1-4_csv/yolov8n-pose/"), each
with synth_fp32.csv / synth_modelopt_int8_full.csv / synth_modelopt_int8_mixed.csv.

Same OKS metric, Hungarian box-IoU matching, and shoulder_ratio>=0.80
failure-mode filter as analyze_1_2.py/analyze_1_3.py -- see those scripts'
docstrings for the full rationale. The difference here is scope, not
method: 12 models (YOLOv8/YOLO11/YOLO26-pose x n/s/m/l) x 2 quantized
precisions (ModelOpt INT8 full, ModelOpt INT8 mixed-excluding-cv4), each
compared self-referentially against that SAME model's own FP32 -- this is
what lets "does quantization hurt on top of a model's own capacity" be
asked independently of "is this a bigger/better model" (a bigger model's
own FP32 baseline is already better; OKS-vs-own-FP32 factors that out).

Produces:
  - oks_1_4_master_table.csv: one row per (model, quant, keypoint) --
    mean_OKS + n_images. This is the master table everything else in this
    script (and every Notion 1-4 chart) is derived from.
  - Fig 1-4-1/2 (oks_1_4_hero_full/mixed): shoulder & wrist OKS across all
    12 models, one chart per precision.
  - Fig 1-4-3 (oks_1_4_size_trend): wrist OKS vs model tier (n/s/m/l), one
    line per architecture, full vs mixed side by side.
  - Fig 1-4-4 (oks_1_4_full_vs_mixed_delta): per-model wrist OKS
    improvement from switching full -> mixed.
  - Fig 1-4-5 (oks_1_4_head_anomaly_crossmodel): 17-keypoint OKS averaged
    ACROSS the 12 models (not across images -- each model contributes one
    number per keypoint, so the error bar here is model-to-model spread,
    N=12), full vs mixed overlaid. Checks whether the head-keypoint anomaly
    found in 1-2 (single model) generalizes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.labelsize"] = 13
matplotlib.rcParams["xtick.labelsize"] = 11
matplotlib.rcParams["ytick.labelsize"] = 11
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

DATA_DIR = Path(__file__).resolve().parent.parent / "1-4_csv"
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"

# (model_dir, arch, tier) -- arch/tier drive the size-trend and hero-chart
# ordering, matching the Notion figures' n->s->m->l, v8/v11/v26 grouping.
MODELS = [
    ("yolov8n-pose", "v8", "n"), ("yolov8s-pose", "v8", "s"), ("yolov8m-pose", "v8", "m"), ("yolov8l-pose", "v8", "l"),
    ("yolo11n-pose", "v11", "n"), ("yolo11s-pose", "v11", "s"), ("yolo11m-pose", "v11", "m"), ("yolo11l-pose", "v11", "l"),
    ("yolo26n-pose", "v26", "n"), ("yolo26s-pose", "v26", "s"), ("yolo26m-pose", "v26", "m"), ("yolo26l-pose", "v26", "l"),
]
TIER_ORDER = ["n", "s", "m", "l"]
ARCH_COLOR = {"v8": "#4C72B0", "v11": "#DD8452", "v26": "#55A868"}
# IEEE print figures fall back to grayscale -- pair each architecture with
# its own marker shape so the 3 series stay distinguishable by shape alone.
ARCH_MARKER = {"v8": "o", "v11": "s", "v26": "^"}
QUANT_FILES = {"full": "synth_modelopt_int8_full.csv", "mixed": "synth_modelopt_int8_mixed.csv"}
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
KP_GROUP = {**{k: "head" for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]},
            **{k: "trunk" for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]},
            **{k: "mid-limb" for k in ["left_elbow", "right_elbow", "left_knee", "right_knee"]},
            **{k: "extremity" for k in ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]}}
GROUP_BOUNDS_17 = [("head", 0, 5), ("trunk", 5, 9), ("mid-limb", 9, 13), ("extremity", 13, 17)]
BOX_COLS = ["box_x1", "box_y1", "box_x2", "box_y2"]
MIN_IOU = 0.3


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


def obj_scale(ref: pd.DataFrame) -> pd.Series:
    return np.sqrt((ref["box_x2"] - ref["box_x1"]) * (ref["box_y2"] - ref["box_y1"]))


def oks(ref: pd.DataFrame, oth: pd.DataFrame, kp: str, s: pd.Series) -> pd.Series:
    d2 = (ref[f"{kp}_x"] - oth[f"{kp}_x"]) ** 2 + (ref[f"{kp}_y"] - oth[f"{kp}_y"]) ** 2
    return np.exp(-d2 / (2 * (s ** 2) * (COCO_SIGMA[kp] ** 2)))


def pdist(ax, ay, bx, by):
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def per_keypoint_oks(fp32_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    """One row per matched detection: filename, shoulder_ratio (failure-mode
    filter), and per-keypoint OKS for every keypoint in KP_ORDER."""
    m = match_detections(fp32_df, variant_df)
    ref = fp32_df.loc[m["ref_idx"]].reset_index(drop=True)
    oth = variant_df.loc[m["other_idx"]].reset_index(drop=True)
    s = obj_scale(ref)
    shoulder_fp32 = pdist(ref.left_shoulder_x, ref.left_shoulder_y, ref.right_shoulder_x, ref.right_shoulder_y)
    shoulder_var = pdist(oth.left_shoulder_x, oth.left_shoulder_y, oth.right_shoulder_x, oth.right_shoulder_y)
    out = {"filename": m["filename"].values, "shoulder_ratio": (shoulder_var / shoulder_fp32).values}
    for kp in KP_ORDER:
        out[kp] = oks(ref, oth, kp, s).values
    return pd.DataFrame(out)


def savefig(fig, name):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg, +.pdf)")


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {DATA_DIR} -- extract 1-4_csv.7z here first")
    CHARTS_DIR.mkdir(exist_ok=True)

    # ============================================================
    # build the master table: mean OKS per (model, quant, keypoint)
    # ============================================================
    rows = []
    kp_mean = {}  # (model, quant) -> Series indexed by KP_ORDER
    for model, arch, tier in MODELS:
        model_dir = DATA_DIR / model
        fp32_df = pd.read_csv(model_dir / "synth_fp32.csv")
        for quant, fname in QUANT_FILES.items():
            print(f"matching {model} [{quant}]...", flush=True)
            variant_df = pd.read_csv(model_dir / fname)
            det = per_keypoint_oks(fp32_df, variant_df)
            excluded = (det["shoulder_ratio"] < SHOULDER_RATIO_THRESHOLD).sum()
            det = det[det["shoulder_ratio"] >= SHOULDER_RATIO_THRESHOLD].reset_index(drop=True)
            per_image = det.groupby("filename")[KP_ORDER].mean()
            means = per_image.mean().reindex(KP_ORDER)
            kp_mean[(model, quant)] = means
            n = len(per_image)
            print(f"  excluded {excluded}/{len(det)+excluded} (shoulder_ratio < {SHOULDER_RATIO_THRESHOLD}), n_images={n}")
            for kp in KP_ORDER:
                rows.append({"model": model, "arch": arch, "tier": tier, "quant": quant,
                             "keypoint": kp, "group": KP_GROUP[kp], "mean_OKS": means[kp], "n_images": n})

    master = pd.DataFrame(rows)
    master.to_csv(CHARTS_DIR / "oks_1_4_master_table.csv", index=False)
    print(f"\nsaved {CHARTS_DIR / 'oks_1_4_master_table.csv'} ({len(master)} rows)")

    # Tier-grouped x-axis ordering (n,n,n | s,s,s | m,m,m | l,l,l) used by
    # Fig 1-4-1/2/4 -- matches deployability framing (n/s run on Orin Nano,
    # m/l are reference-only), NOT the arch-grouped MODELS order used above
    # for matching/computation.
    ARCH_ORDER = ["v8", "v11", "v26"]
    ARCH_DISPLAY = {"v8": "YOLOv8", "v11": "YOLO11", "v26": "YOLO26"}
    TIER_MODELS = [next(mt for mt in MODELS if mt[1] == arch and mt[2] == tier)
                   for tier in TIER_ORDER for arch in ARCH_ORDER]
    tier_xlabels = [f"{a}{t}" for _, a, t in TIER_MODELS]
    divider_x = len(ARCH_ORDER) * 2 - 0.5  # between s-block and m-block

    # ============================================================
    # Fig 1-4-1/2: shoulder & wrist OKS across 12 models, one chart per quant
    # ============================================================
    QUANT_TITLE = {"full": "INT8(ModelOpt full)", "mixed": "INT8(ModelOpt mixed→excl.cv4)"}
    for quant, fname in [("full", "oks_1_4_hero_full"), ("mixed", "oks_1_4_hero_mixed")]:
        x = np.arange(12)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        for arch in ARCH_ORDER:
            shoulder = [kp_mean[(m, quant)][["left_shoulder", "right_shoulder"]].mean()
                        for m, a, t in TIER_MODELS if a == arch]
            wrist = [kp_mean[(m, quant)][["left_wrist", "right_wrist"]].mean()
                     for m, a, t in TIER_MODELS if a == arch]
            xs = [i for i, (m, a, t) in enumerate(TIER_MODELS) if a == arch]
            # linestyle carries the shoulder/wrist role, marker carries arch
            # identity, fill carries role too (redundant, grayscale-safe).
            ax.plot(xs, shoulder, linestyle="-", marker=ARCH_MARKER[arch], markersize=9, color=ARCH_COLOR[arch], label=f"{ARCH_DISPLAY[arch]} – shoulder")
            ax.plot(xs, wrist, linestyle="--", marker=ARCH_MARKER[arch], markersize=9, color=ARCH_COLOR[arch], markerfacecolor="white", label=f"{ARCH_DISPLAY[arch]} – wrist")
        ax.axvline(divider_x, color="black", linestyle=":", linewidth=1.5)
        ymin, ymax = ax.get_ylim()
        ytext = ymin + 0.30 * (ymax - ymin)
        ytext_low = ymin + 0.06 * (ymax - ymin)
        ax.text(divider_x - 1.7, ytext_low, "n/s (Nano-deployable)", ha="center", fontsize=12, style="italic", color="dimgray")
        ax.text(divider_x + 1 + (11 - divider_x) / 2, ytext, "m/l (reference only)", ha="center", fontsize=12, style="italic", color="dimgray")
        ax.set_xticks(x)
        ax.set_xticklabels(tier_xlabels)
        ax.set_ylabel("mean OKS vs own FP32")
        ax.legend(fontsize=11, loc="lower right", ncol=2, markerscale=1.1, labelspacing=0.7, handlelength=2.2, handletextpad=0.7, borderpad=0.7)
        fig.tight_layout()
        savefig(fig, fname)

    # ============================================================
    # Fig 1-4-3: wrist OKS vs model tier, one line per architecture, full vs mixed
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, quant, legend_loc in zip(axes, ["full", "mixed"], ["upper left", "lower left"]):
        for arch in ARCH_ORDER:
            y = []
            for tier in TIER_ORDER:
                model = next(m for m, a, t in MODELS if a == arch and t == tier)
                y.append(kp_mean[(model, quant)][["left_wrist", "right_wrist"]].mean())
            ax.plot(TIER_ORDER, y, "s-", color=ARCH_COLOR[arch], label=ARCH_DISPLAY[arch])
        ax.set_xlabel("model tier (capacity increasing →)")
        ax.set_title(f"wrist OKS vs tier, INT8({quant})")
        ax.legend(loc=legend_loc)
    axes[0].set_ylabel("mean wrist OKS vs own FP32")
    fig.suptitle("1-4 (hands-down): does quantization damage shrink monotonically with model capacity?")
    fig.tight_layout()
    savefig(fig, "oks_1_4_size_trend")

    # ============================================================
    # Fig 1-4-4: full -> mixed wrist OKS improvement per model
    # ============================================================
    delta = []
    for m, a, t in TIER_MODELS:
        full_wrist = kp_mean[(m, "full")][["left_wrist", "right_wrist"]].mean()
        mixed_wrist = kp_mean[(m, "mixed")][["left_wrist", "right_wrist"]].mean()
        delta.append(mixed_wrist - full_wrist)
    fig, ax = plt.subplots(figsize=(13, 5))
    bar_colors = [ARCH_COLOR[a] for _, a, _ in TIER_MODELS]
    ax.bar(range(12), delta, color=bar_colors)
    ax.axvline(divider_x, color="black", linestyle=":", linewidth=1.5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(tier_xlabels)
    ax.set_ylabel("wrist OKS improvement (mixed − full)")
    ax.set_title("1-4: does excluding cv4 from quantization help consistently across all 12 models?")
    fig.tight_layout()
    savefig(fig, "oks_1_4_full_vs_mixed_delta")
    print("\nfull->mixed wrist OKS delta per model:")
    for (m, _, _), d in zip(TIER_MODELS, delta):
        print(f"  {m:<16} {d:.4f}")

    # ============================================================
    # Fig 1-4-5: 17-keypoint OKS averaged ACROSS the 12 models (N=12 SEM),
    # full vs mixed overlaid -- checks the 1-2 head-anomaly finding generalizes
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 5.5))
    x = np.arange(len(KP_ORDER))
    for quant, color, offset in [("full", "tab:red", -0.1), ("mixed", "tab:green", 0.1)]:
        vals = np.array([[kp_mean[(m, quant)][kp] for m, _, _ in MODELS] for kp in KP_ORDER])  # (17, 12)
        means = vals.mean(axis=1)
        sems = vals.std(axis=1, ddof=1) / np.sqrt(vals.shape[1])
        ax.errorbar(x + offset, means, yerr=sems, fmt="o", markersize=6, capsize=3, color=color,
                    label=f"INT8({quant})")
    for name, start, end in GROUP_BOUNDS_17:
        if start > 0:
            ax.axvline(start - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.text((start + end - 1) / 2, 1.001, name, ha="center", fontsize=10, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(KP_ORDER, rotation=45, ha="right")
    ax.set_ylabel("mean OKS across 12 models, ±1 SEM (across models)")
    ax.set_title("1-4: is the head-keypoint anomaly (1-2 finding) consistent across 12 models?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "oks_1_4_head_anomaly_crossmodel")

    print("\nDONE")


if __name__ == "__main__":
    main()
