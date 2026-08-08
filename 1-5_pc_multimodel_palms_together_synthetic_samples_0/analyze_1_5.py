#!/usr/bin/env python3
"""
analyze_1_5.py
===============
Reproduces the Notion 1-5 numbers and charts (Fig 1-5-1 through 1-5-7) from
the per-detection CSVs in 1-5_csv.7z + 1-5_fp16_csv.7z. Extract both archives
into sibling folders "1-5_csv" (per-model subfolders, each with
synth_fp32.csv / synth_modelopt_int8_full.csv / synth_modelopt_int8_mixed.csv)
and "1-5_fp16_csv" (flat, "{model}.synth_fp16.csv" per file) next to this
script before running.

Same OKS metric, Hungarian box-IoU matching, and shoulder_ratio>=0.80
failure-mode filter as analyze_1_2.py/analyze_1_3.py/analyze_1_4.py -- see
those scripts' docstrings for the full rationale. 1-5 is 1-4's palms-together
counterpart: same 12 models (YOLOv8/YOLO11/YOLO26-pose x n/s/m/l) x 2
quantized precisions, but ALSO extends into 1-3's R-value/correction
methodology (Fig 1-5-6/7) to check whether that methodology generalizes
beyond the single v8n model it was designed on.

Produces:
  - oks_1_5_master_table.csv, r_value_1_5_summary_table.csv
  - Fig 1-5-1/2 (1-5-1_hero_full / 1-5-2_hero_mixed): shoulder & wrist OKS
    across all 12 models, one chart per precision.
  - Fig 1-5-3 (1-5-3_size_trend): wrist OKS vs model tier (n/s/m/l).
  - Fig 1-5-4 (1-5-4_full_vs_mixed_delta): per-model wrist OKS improvement
    from switching full -> mixed.
  - Fig 1-5-5 (1-5-5_head_anomaly_crossmodel): 17-keypoint OKS averaged
    ACROSS the 12 models (N=12 SEM), full vs mixed overlaid.
  - Fig 1-5-6 (1-5-6_r_vs_distance_crossmodel): R value (wrist_dist/
    shoulder_dist) vs distance, 12 models, full vs mixed panels, n/s
    (deployable on Orin Nano) emphasized over m/l (reference only).
  - Fig 1-5-7 (1-5-7_miss_rate_fixed_vs_adaptive): does 1-3's full
    correction methodology (yolo11x-calibrated threshold + per-variant
    ratio correction) generalize across 12 models? FP32/FP16/INT8(mixed),
    raw vs corrected R vs the yolo11x threshold curve.

External shared dependency (documented, not duplicated -- see analyze_1_3.py
for the same pattern): Fig 1-5-6 doesn't need it, but Fig 1-5-7's threshold
curve needs yolo11x-pose's own FP32 self-referential R curve, computed once
and reused across 1-3/1-5/1-6:
    YOLO11X_FP32_CSV = Path(r"C:\\Users\\user\\pose_quant_env\\xtier_fp32_pray_results\\yolo11x-pose.synth_fp32.csv")
If unavailable, everything except Fig 1-5-7 still runs (guarded below).
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

DATA_DIR = Path(__file__).parent / "1-5_csv"
FP16_DIR = Path(__file__).parent / "1-5_fp16_csv"
CHARTS_DIR = Path(__file__).parent / "charts"
YOLO11X_FP32_CSV = Path(r"C:\Users\user\pose_quant_env\xtier_fp32_pray_results\yolo11x-pose.synth_fp32.csv")

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
K = 1.645

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


def box_iou_matrix(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(ref_df, other_df):
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


def obj_scale(ref):
    return np.sqrt((ref["box_x2"] - ref["box_x1"]) * (ref["box_y2"] - ref["box_y1"]))


def oks(ref, oth, kp, s):
    d2 = (ref[f"{kp}_x"] - oth[f"{kp}_x"]) ** 2 + (ref[f"{kp}_y"] - oth[f"{kp}_y"]) ** 2
    return np.exp(-d2 / (2 * (s ** 2) * (COCO_SIGMA[kp] ** 2)))


def pdist(ax, ay, bx, by):
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def per_keypoint_oks(fp32_df, variant_df):
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


def _dist(df, kp):
    return np.sqrt((df[f"left_{kp}_x"] - df[f"right_{kp}_x"]) ** 2 + (df[f"left_{kp}_y"] - df[f"right_{kp}_y"]) ** 2)


def r_self(df):
    shoulder_px = _dist(df, "shoulder")
    wrist_px = _dist(df, "wrist")
    return pd.DataFrame({"distance_cm": df["distance_cm"].values, "R": (wrist_px / shoulder_px).values})


def curve_from_self(df):
    r = r_self(df)
    return r.groupby("distance_cm").agg(R_mean=("R", "mean"), R_std=("R", "std")).reset_index()


def per_detection_R(fp32_df, variant_df):
    """Match variant to its own-model FP32, R from the variant's own keypoints."""
    m = match_detections(fp32_df, variant_df)
    ref = fp32_df.loc[m["ref_idx"]].reset_index(drop=True)
    oth = variant_df.loc[m["other_idx"]].reset_index(drop=True)
    shoulder_ref = _dist(ref, "shoulder")
    shoulder_oth = _dist(oth, "shoulder")
    wrist_oth = _dist(oth, "wrist")
    out = pd.DataFrame({
        "distance_cm": ref["distance_cm"].values,
        "R": (wrist_oth / shoulder_oth).values,
        "shoulder_ratio": (shoulder_oth / shoulder_ref).values,
    })
    return out[out["shoulder_ratio"] >= SHOULDER_RATIO_THRESHOLD].reset_index(drop=True)


def savefig(fig, name):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg, +.pdf)")


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {DATA_DIR} -- extract 1-5_csv.7z here first")
    CHARTS_DIR.mkdir(exist_ok=True)

    # ============================================================
    # 1. build the OKS master table: mean OKS per (model, quant, keypoint)
    # ============================================================
    rows = []
    kp_mean = {}
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
    master.to_csv(CHARTS_DIR / "oks_1_5_master_table.csv", index=False)
    print(f"\nsaved {CHARTS_DIR / 'oks_1_5_master_table.csv'} ({len(master)} rows)")

    # Tier-grouped x-axis ordering (n,n,n | s,s,s | m,m,m | l,l,l) used by
    # Fig 1-5-1/2/4 -- matches deployability framing (n/s run on Orin Nano,
    # m/l are reference-only), NOT the arch-grouped MODELS order used above
    # for matching/computation.
    ARCH_ORDER = ["v8", "v11", "v26"]
    ARCH_DISPLAY = {"v8": "YOLOv8", "v11": "YOLO11", "v26": "YOLO26"}
    TIER_MODELS = [next(mt for mt in MODELS if mt[1] == arch and mt[2] == tier)
                   for tier in TIER_ORDER for arch in ARCH_ORDER]
    tier_xlabels = [f"{a}{t}" for _, a, t in TIER_MODELS]
    divider_x = len(ARCH_ORDER) * 2 - 0.5  # between s-block and m-block

    # ============================================================
    # Fig 1-5-1/2: shoulder & wrist OKS across 12 models, one chart per quant
    # ============================================================
    QUANT_TITLE = {"full": "INT8(ModelOpt full)", "mixed": "INT8(ModelOpt mixed→excl.cv4)"}
    for quant, fname in [("full", "1-5-1_hero_full"), ("mixed", "1-5-2_hero_mixed")]:
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
        # "n/s" sits low, near the divider -- clear of data in both panels
        # (full's only low point is the single YOLO26n anomaly at x=2, long
        # resolved by x=5; mixed never dips this close to its own ymin).
        # "m/l" stays at its original higher spot -- moving it down would
        # run it into the legend box in the bottom-right corner instead.
        ax.text(divider_x - 1.7, ytext_low, "n/s (Nano-deployable)", ha="center", fontsize=12, style="italic", color="dimgray")
        ax.text(divider_x + 1 + (11 - divider_x) / 2, ytext, "m/l (reference only)", ha="center", fontsize=12, style="italic", color="dimgray")
        ax.set_xticks(x)
        ax.set_xticklabels(tier_xlabels)
        ax.set_ylabel("mean OKS vs own FP32")
        ax.legend(fontsize=11, loc="lower right", ncol=2, markerscale=1.1, labelspacing=0.7, handlelength=2.2, handletextpad=0.7, borderpad=0.7)
        fig.tight_layout()
        savefig(fig, fname)

    # ============================================================
    # Fig 1-5-3: wrist OKS vs model tier, one line per architecture, full vs mixed
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, quant, legend_loc in zip(axes, ["full", "mixed"], ["upper left", "upper left"]):
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
    fig.suptitle("1-5 (palms-together): does quantization damage shrink monotonically with model capacity?")
    fig.tight_layout()
    savefig(fig, "1-5-3_size_trend")

    # ============================================================
    # Fig 1-5-4: full -> mixed wrist OKS improvement per model
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
    ax.set_title("1-5: does excluding cv4 from quantization help consistently across all 12 models?")
    fig.tight_layout()
    savefig(fig, "1-5-4_full_vs_mixed_delta")
    print("\nfull->mixed wrist OKS delta per model:")
    for (m, _, _), d in zip(TIER_MODELS, delta):
        print(f"  {m:<16} {d:.4f}")

    # ============================================================
    # Fig 1-5-5: 17-keypoint OKS averaged ACROSS the 12 models (N=12 SEM)
    # ============================================================
    fig, ax = plt.subplots(figsize=(16, 5.5))
    x = np.arange(len(KP_ORDER))
    for quant, color, marker, offset in [("full", "tab:red", "o", -0.1), ("mixed", "tab:green", "s", 0.1)]:
        vals = np.array([[kp_mean[(m, quant)][kp] for m, _, _ in MODELS] for kp in KP_ORDER])
        means = vals.mean(axis=1)
        sems = vals.std(axis=1, ddof=1) / np.sqrt(vals.shape[1])
        ax.errorbar(x + offset, means, yerr=sems, fmt=marker, markersize=6, capsize=3, color=color,
                    label=f"INT8({quant})")
    ymin, ymax = ax.get_ylim()
    ytext_group = ymax - 0.05 * (ymax - ymin)
    for name, start, end in GROUP_BOUNDS_17:
        if start > 0:
            ax.axvline(start - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.text((start + end - 1) / 2, ytext_group, name, ha="center", fontsize=13, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(KP_ORDER, rotation=45, ha="right")
    ax.set_ylabel("mean OKS across 12 models, ±1 SEM (across models)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "1-5-5_head_anomaly_crossmodel")

    # ============================================================
    # 2. R value vs distance (needed for Fig 1-5-6 and Fig 1-5-7)
    # ============================================================
    print("\n" + "=" * 100)
    print("2. R value vs distance per model/quant (Fig 1-5-6)")
    print("=" * 100)
    own_curve_full_mixed = {}  # (model, quant) -> curve_from_self df
    for model, arch, tier in MODELS:
        model_dir = DATA_DIR / model
        for quant, fname in QUANT_FILES.items():
            df = pd.read_csv(model_dir / fname)
            own_curve_full_mixed[(model, quant)] = curve_from_self(df)

    # v26 excluded here: its known cv4-related instability (see Fig 1-5-5)
    # would read as "v26 has a problem" in a chart that isn't about that --
    # that story is already told where it belongs. m/l tiers dropped too
    # (reference-only, not Nano-deployable) to keep this to the 8 series
    # (2 arch x 2 tier x 2 quant) that are actually the point.
    TIER_STYLE = {"n": ("o", "-"), "s": ("s", "--")}
    QUANT_FILL = {"full": True, "mixed": False}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for model, arch, tier in MODELS:
        if arch == "v26" or tier not in ("n", "s"):
            continue
        marker, ls = TIER_STYLE[tier]
        for quant in ("full", "mixed"):
            c = own_curve_full_mixed[(model, quant)]
            filled = QUANT_FILL[quant]
            ax.plot(c["distance_cm"], c["R_mean"], color=ARCH_COLOR[arch], marker=marker, linestyle=ls,
                     markersize=6, linewidth=2.0,
                     markerfacecolor=ARCH_COLOR[arch] if filled else "white",
                     label=f"{ARCH_DISPLAY[arch]}-{tier} ({quant})")
    ax.axhline(0.4, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("mean R (wrist_dist / shoulder_dist)")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    savefig(fig, "1-5-6_r_vs_distance_crossmodel")

    # ============================================================
    # 3. Fig 1-5-7: full correction methodology across 12 models
    # (FP32 / FP16 / INT8(mixed), needs yolo11x FP32 target curve + FP16 CSVs)
    # ============================================================
    if not YOLO11X_FP32_CSV.exists():
        print(f"\n[SKIP Fig 1-5-7] yolo11x FP32 target curve not found at {YOLO11X_FP32_CSV}")
        print("DONE (1-5-1 through 1-5-6 only)")
        return
    if not FP16_DIR.is_dir():
        print(f"\n[SKIP Fig 1-5-7] FP16 CSVs not found at {FP16_DIR} -- extract fp16_csv.7z here first")
        print("DONE (1-5-1 through 1-5-6 only)")
        return

    print("\n" + "=" * 100)
    print("3. Full correction methodology across 12 models (Fig 1-5-7)")
    print("=" * 100)
    yolo11x_curve = curve_from_self(pd.read_csv(YOLO11X_FP32_CSV))
    yolo11x_curve["threshold"] = yolo11x_curve["R_mean"] + K * yolo11x_curve["R_std"]
    x11_mean = dict(zip(yolo11x_curve["distance_cm"], yolo11x_curve["R_mean"]))
    x11_std = dict(zip(yolo11x_curve["distance_cm"], yolo11x_curve["R_std"]))
    x11_thr = dict(zip(yolo11x_curve["distance_cm"], yolo11x_curve["threshold"]))
    print("yolo11x-pose FP32 threshold curve:")
    print(yolo11x_curve.round(4).to_string(index=False))

    def correct(R_raw, distances, own_mean_map, own_std_map):
        return np.array([
            x11_mean[d] + (r - own_mean_map[d]) * (x11_std[d] / own_std_map[d])
            for r, d in zip(R_raw, distances)
        ])

    VARIANT_LABEL = {"fp32": "FP32", "fp16": "FP16", "int8_mixed": "INT8(mixed)"}
    VARIANT_COLOR = {"fp32": "#555555", "fp16": "#DD8452", "int8_mixed": "#55A868"}
    ARCH_LABEL = {"v8": "v8", "v11": "v11", "v26": "v26"}

    r5_rows = []
    for model, arch, tier in MODELS:
        model_dir = DATA_DIR / model
        fp32_df = pd.read_csv(model_dir / "synth_fp32.csv")
        fp16_df = pd.read_csv(FP16_DIR / f"{model}.synth_fp16.csv")
        mixed_df = pd.read_csv(model_dir / "synth_modelopt_int8_mixed.csv")

        own_curve = {"fp32": curve_from_self(fp32_df), "fp16": curve_from_self(fp16_df),
                     "int8_mixed": curve_from_self(mixed_df)}
        raw_R = {"fp32": r_self(fp32_df),
                  "fp16": per_detection_R(fp32_df, fp16_df)[["distance_cm", "R"]],
                  "int8_mixed": per_detection_R(fp32_df, mixed_df)[["distance_cm", "R"]]}

        for variant in ["fp32", "fp16", "int8_mixed"]:
            oc = own_curve[variant]
            own_mean_map = dict(zip(oc["distance_cm"], oc["R_mean"]))
            own_std_map = dict(zip(oc["distance_cm"], oc["R_std"]))
            d = raw_R[variant]
            thr = d["distance_cm"].map(x11_thr)
            R_corr = correct(d["R"].values, d["distance_cm"].values, own_mean_map, own_std_map)
            r5_rows.append({"model": model, "arch": arch, "tier": tier, "variant": variant,
                             "miss_raw": (d["R"].values >= thr.values).mean(),
                             "miss_corrected": (R_corr >= thr.values).mean(), "n": len(d)})
        print(f"done: {model}")

    summary = pd.DataFrame(r5_rows)
    summary.to_csv(CHARTS_DIR / "r_value_1_5_summary_table.csv", index=False)
    print("\n" + summary.round(4).to_string(index=False))

    # v26 excluded from R-value-related comparisons (same reasoning as Fig
    # 1-5-6): its cv4-related instability is already established there, so
    # it doesn't belong in a chart that isn't about that. Dropping it also
    # cuts 12 model-groups down to 8, letting this fit a single column
    # instead of needing double-column width.
    summary = summary[summary["arch"] != "v26"].copy()
    MODEL_ORDER = [f"{arch}{size}" for size in TIER_ORDER for arch in ["v8", "v11"]]
    summary["order"] = summary.apply(lambda r: MODEL_ORDER.index(f"{r['arch']}{r['tier']}"), axis=1)
    xlabels8 = [f"{ARCH_LABEL[a]}{t}" for t in TIER_ORDER for a in ["v8", "v11"]]

    # Grouped bar chart: 6 bars per model (3 variants x raw/corrected).
    # raw vs corrected: fill (white+hatch = raw/baseline, solid = corrected/
    # the paper's result). Variant (fp32/fp16/int8_mixed) is color -- but
    # since position alone isn't obvious to a first-time reader, the first
    # model group is additionally labeled a-f above its bars, matching the
    # a-f prefixes in the legend; the reader decodes bar order once from
    # that group and reads the other 7 by position afterward.
    fig, ax = plt.subplots(figsize=(9, 6.5))
    x = np.arange(8)
    n_bars = 6
    bar_w = 0.8 / n_bars
    variants = ["fp32", "fp16", "int8_mixed"]
    slots = [(v, role) for v in variants for role in ("raw", "corrected")]
    bar_handles = {}
    for j, (variant, role) in enumerate(slots):
        letter = chr(ord("a") + j)
        sub = summary[summary["variant"] == variant].sort_values("order")
        color = VARIANT_COLOR[variant]
        vals = sub["miss_raw"].values if role == "raw" else sub["miss_corrected"].values
        offset = (j - (n_bars - 1) / 2) * bar_w
        label = f"({letter}) {VARIANT_LABEL[variant]} – {role}"
        if role == "raw":
            bars = ax.bar(x + offset, vals, width=bar_w, facecolor="white", edgecolor=color,
                    hatch="///", linewidth=1.0, label=label)
        else:
            bars = ax.bar(x + offset, vals, width=bar_w, facecolor=color, edgecolor="black",
                    linewidth=0.6, label=label)
        ax.text(x[0] + offset, vals[0] + 0.004, letter, ha="center", fontsize=11)
        bar_handles[(variant, role)] = (bars, label)
    ax.axvline(3.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels8, fontsize=10)
    ax.set_ylabel("miss rate (fraction of person-detections missed)")
    # 3 rows x 2 cols, each row = one variant's (raw, corrected) pair --
    # column-major fill with ncol=2 puts the first half of the handle list
    # in column 1 and the second half in column 2.
    legend_order = [(v, role) for role in ("raw", "corrected") for v in variants]
    handles = [bar_handles[k][0] for k in legend_order]
    labels = [bar_handles[k][1] for k in legend_order]
    ax.legend(handles, labels, fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    savefig(fig, "1-5-7_miss_rate_fixed_vs_adaptive")

    print("\nDONE")


if __name__ == "__main__":
    main()
