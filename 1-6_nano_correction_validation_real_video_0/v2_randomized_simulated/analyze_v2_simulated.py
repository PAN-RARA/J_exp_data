import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\user\pose_quant_env\J_exp_data\1-6_nano_correction_validation_real_video_0")
import analyze_1_6 as m

NEW_SIM_DIR = r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\real_video_results_simulated_v2"

# same yolo11x real-video reference curve/threshold used for the original analysis
x11_raw = pd.read_csv(m.YOLO11X_REAL_VIDEO_CSV)
x11_filtered = m.top_n_filter(x11_raw)
x11_curve = m.curve_from_self(x11_filtered)
x11_curve["threshold"] = x11_curve["R_p95"]
x11_mean = dict(zip(x11_curve["distance_cm"], x11_curve["R_mean"]))
x11_std = dict(zip(x11_curve["distance_cm"], x11_curve["R_std"]))
x11_thr = dict(zip(x11_curve["distance_cm"], x11_curve["threshold"]))

CHART_VARIANTS = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
SIM_DISTANCES = [275, 325, 375, 425, 450, 475, 500, 525, 550]

all_rows = []
for variant in CHART_VARIANTS:
    raw = pd.read_csv(f"{NEW_SIM_DIR}/sim_{variant}.csv")
    print(f"{variant}: {len(raw)} raw detections before filtering")
    filtered = m.top_n_filter(raw)
    print(f"{variant}: {len(filtered)} after top_n_filter")

    synth_df = pd.read_csv(m.SYNTH_DIR / f"synth_{variant}.csv")
    own_curve = m.curve_from_self(synth_df)
    own_mean = dict(zip(own_curve["distance_cm"], own_curve["R_mean"]))
    own_std = dict(zip(own_curve["distance_cm"], own_curve["R_std"]))

    r = m.r_self(filtered)
    r = r[r["distance_cm"].isin(SIM_DISTANCES)].copy()
    thr = r["distance_cm"].map(x11_thr)
    R_corrected = r["distance_cm"].map(x11_mean) + (r["R"] - r["distance_cm"].map(own_mean)) * (
        r["distance_cm"].map(x11_std) / r["distance_cm"].map(own_std))
    r["R_corrected"] = R_corrected
    r["threshold"] = thr
    r["miss_corrected"] = r["R_corrected"] >= r["threshold"]
    r["miss_fixed_0.4"] = r["R"] >= 0.4
    r["variant"] = variant
    all_rows.append(r)

full = pd.concat(all_rows)
full.to_csv(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\v2_corrected_results.csv", index=False)

print()
print("=" * 90)
print("NEW (v2, randomized multi-source) corrected miss rate by distance, mean across 3 CHART_VARIANTS")
print("=" * 90)
summary = full.groupby("distance_cm").agg(
    miss_corrected_mean=("miss_corrected", "mean"),
    miss_fixed_mean=("miss_fixed_0.4", "mean"),
    n=("R", "count"),
).reset_index()
print(summary.round(4).to_string(index=False))

print()
print("=" * 90)
print("per-variant breakdown")
print("=" * 90)
print(full.groupby(["variant", "distance_cm"])["miss_corrected"].mean().unstack().round(4).to_string())
