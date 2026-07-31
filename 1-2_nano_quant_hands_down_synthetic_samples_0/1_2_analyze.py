#!/usr/bin/env python3
"""
1_2_analyze.py -- 1-2 (hands-down synthetic dataset) corrected analysis.

Reproduces every number in the Notion "1-2 ... (新)" section's 詳細數據:
  (a) per-image-aggregated wrist/shoulder MAE by distance, all n_people
      -- INT8(legacy) table, plus ModelOpt full vs mixed comparison
  (b) R-value distribution (pooled within-character variance), 4 variants
  (c) merged dose-response + mitigation table
  (d) cross-n_people robustness (ModelOpt mixed)
  (e) R-value NSR vs distance, 4 variants

Shared matching/statistics logic lives in ../common/ -- see
pose_quant_analysis.py for the methodology writeup (per-image aggregation,
character-identity recovery, pooled within-character R statistics, the
0.80 shoulder-ratio exclusion filter).

Run: python 1_2_analyze.py
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "1-2_csv"
REPO_ROOT = Path(__file__).resolve().parent.parent
ARRANGEMENT_PATH = REPO_ROOT / "blender_file" / "arrangement.json"
sys.path.insert(0, str(REPO_ROOT / "common"))

from pose_quant_analysis import (  # noqa: E402
    load_variant_csvs, build_character_map, build_all_variant_detections,
    per_image_then_across_images, overall_shoulder_wrist, r_stats,
)

VARIANTS = ["fp16", "int8", "modelopt_int8", "modelopt_int8_excl_cv4"]
LABELS = {
    "fp16": "FP16",
    "int8": "INT8(legacy)",
    "modelopt_int8": "INT8(ModelOpt full)",
    "modelopt_int8_excl_cv4": "INT8(ModelOpt mixed)",
}


def main():
    print("Loading CSVs...")
    dfs = load_variant_csvs(DATA_DIR, VARIANTS)
    print("Building character map from arrangement.json...")
    char_map = build_character_map(dfs["fp32"], ARRANGEMENT_PATH)

    print("Matching each variant against FP32 (Hungarian box-IoU) + applying shoulder-ratio filter...")
    filtered = build_all_variant_detections(dfs, VARIANTS, char_map)

    print("\n=== (a) INT8(legacy): wrist/shoulder MAE by distance, all n_people, per-image aggregated ===")
    print(per_image_then_across_images(filtered["int8"], ["distance_cm"]).sort_values("distance_cm").round(3).to_string(index=False))

    print("\n=== (a) ModelOpt full vs mixed: wrist/shoulder MAE by distance ===")
    print("-- ModelOpt full --")
    print(per_image_then_across_images(filtered["modelopt_int8"], ["distance_cm"]).sort_values("distance_cm").round(3).to_string(index=False))
    print("-- ModelOpt mixed --")
    print(per_image_then_across_images(filtered["modelopt_int8_excl_cv4"], ["distance_cm"]).sort_values("distance_cm").round(3).to_string(index=False))

    print("\n=== (b) R-value distribution (pooled within-character), 4 variants ===")
    b_rows = [{"variant": LABELS[v], **r_stats(filtered[v])} for v in VARIANTS]
    print(pd.DataFrame(b_rows).round(4).to_string(index=False))

    print("\n=== (c) merged dose-response + mitigation table (all distances collapsed) ===")
    c_rows = []
    for v in VARIANTS:
        shoulder, wrist = overall_shoulder_wrist(filtered[v])
        stats = r_stats(filtered[v])
        c_rows.append({
            "variant": LABELS[v], "shoulder_MAE": shoulder, "wrist_MAE": wrist,
            "ratio": wrist / shoulder, "R_quant_noise_std": stats["quant_noise_std"], "NSR": stats["NSR"],
        })
    print(pd.DataFrame(c_rows).round(3).to_string(index=False))

    print("\n=== (d) cross-n_people robustness (ModelOpt mixed) ===")
    print(per_image_then_across_images(filtered["modelopt_int8_excl_cv4"], ["n_people"]).sort_values("n_people").round(3).to_string(index=False))

    print("\n=== (e) R-value NSR vs distance, 4 variants ===")
    distances = sorted(filtered["fp16"]["distance_cm"].unique())
    e_rows = []
    for d in distances:
        row = {"distance_cm": d}
        for v in VARIANTS:
            sub = filtered[v][filtered[v]["distance_cm"] == d]
            row[LABELS[v]] = r_stats(sub)["NSR"] if len(sub) else float("nan")
        e_rows.append(row)
    print(pd.DataFrame(e_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
