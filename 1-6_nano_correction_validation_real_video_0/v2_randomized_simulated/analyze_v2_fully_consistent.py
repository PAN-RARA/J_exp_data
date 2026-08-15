import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\user\pose_quant_env\J_exp_data\1-6_nano_correction_validation_real_video_0")
import analyze_1_6 as m

NEW_SIM_DIR = r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\real_video_results_simulated_v2"
NEW_X11_CSV = r"C:\Users\user\pose_quant_env\xtier_fp32_pray_results\yolo11x-pose.real_video_v2.csv"

# NEW yolo11x reference curve (v2: native unchanged, simulated now randomized multi-source,
# same frame files the deployed model was evaluated on)
x11_raw = pd.read_csv(NEW_X11_CSV)
x11_filtered = m.top_n_filter(x11_raw)
x11_curve = m.curve_from_self(x11_filtered)
x11_curve["threshold"] = x11_curve["R_p95"]
x11_mean = dict(zip(x11_curve["distance_cm"], x11_curve["R_mean"]))
x11_std = dict(zip(x11_curve["distance_cm"], x11_curve["R_std"]))
x11_thr = dict(zip(x11_curve["distance_cm"], x11_curve["threshold"]))

print("=== NEW (v2) yolo11x reference curve ===")
print(x11_curve.round(4).to_string(index=False))

CHART_VARIANTS = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
SIM_DISTANCES = [275, 325, 375, 425, 450, 475, 500, 525, 550]
REAL_DISTANCES = [250, 300, 350, 400]

all_rows = []
for variant in CHART_VARIANTS:
    # real (unchanged, native captures)
    real_raw = pd.read_csv(rf"C:\Users\user\pose_quant_env\J_exp_data\1-6_nano_correction_validation_real_video_0\exported_charts\1-6\real_video_results_filtered\{variant}.csv")
    real_filtered = real_raw  # already filtered per original pipeline
    # simulated (new v2)
    sim_raw = pd.read_csv(f"{NEW_SIM_DIR}/sim_{variant}.csv")
    sim_filtered = m.top_n_filter(sim_raw)

    synth_df = pd.read_csv(m.SYNTH_DIR / f"synth_{variant}.csv")
    own_curve = m.curve_from_self(synth_df)
    own_mean = dict(zip(own_curve["distance_cm"], own_curve["R_mean"]))
    own_std = dict(zip(own_curve["distance_cm"], own_curve["R_std"]))

    for source, df in [("real", real_filtered), ("simulated", sim_filtered)]:
        r = m.r_self(df)
        dlist = REAL_DISTANCES if source == "real" else SIM_DISTANCES
        r = r[r["distance_cm"].isin(dlist)].copy()
        thr = r["distance_cm"].map(x11_thr)
        R_corrected = r["distance_cm"].map(x11_mean) + (r["R"] - r["distance_cm"].map(own_mean)) * (
            r["distance_cm"].map(x11_std) / r["distance_cm"].map(own_std))
        r["R_corrected"] = R_corrected
        r["threshold"] = thr
        r["miss_corrected"] = r["R_corrected"] >= r["threshold"]
        r["miss_fixed_0.4"] = r["R"] >= 0.4
        r["variant"] = variant
        r["source"] = source
        all_rows.append(r)

full = pd.concat(all_rows)
full.to_csv(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\v2_fully_consistent_results.csv", index=False)

print()
print("=" * 90)
print("FULLY CONSISTENT v2 corrected miss rate by distance (mean across 3 CHART_VARIANTS)")
print("=" * 90)
summary = full.groupby(["source", "distance_cm"]).agg(
    miss_corrected_mean=("miss_corrected", "mean"),
    miss_fixed_mean=("miss_fixed_0.4", "mean"),
    R_raw_mean=("R", "mean"),
    R_corrected_mean=("R_corrected", "mean"),
    n=("R", "count"),
).reset_index()
print(summary.round(4).to_string(index=False))
