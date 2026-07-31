#!/usr/bin/env python3
"""
pose_quant_analysis.py -- shared analysis routines for the 1-2 / 1-3
synthetic-dataset quantization experiments (per-image aggregation,
character-identity recovery via arrangement.json, pooled within-character
R-value statistics). See each experiment's own analyze.py for the specific
tables it produces; this module just holds the reusable pieces so the two
scripts don't duplicate ~150 lines of near-identical logic.

Methodology summary (see the Notion page's narrative for the "why"):
  - Multi-person aggregation: average all matched people within an image
    FIRST, then average across images (equal weight per image, not per
    person) -- avoids over-weighting higher-n_people images.
  - Character identity: `arrangement.json` gives, per image, each person's
    ground-truth x_m (world position) and character (body model). We
    recover which detection belongs to which character by rank-matching
    ground-truth people (sorted by x_m) against FP32 detections (sorted by
    box center x).
  - R = wrist_dist / shoulder_dist (a body-proportion ratio, NOT a
    cross-variant error). natural_variation_std and quant_noise_std use
    POOLED WITHIN-CHARACTER variance (like pooled-variance ANOVA) so that
    between-character body-type differences don't contaminate the "how
    noisy is quantization" measurement.
  - Filter: exclude detections where shoulder_dist_variant /
    shoulder_dist_fp32 < SHOULDER_RATIO_THRESHOLD -- guards against narrow,
    INT8-specific keypoint-collapse failure modes that would otherwise blow
    up the R-based statistics without reflecting general quantization
    behavior. Always report how many detections got excluded per variant.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from match_detections import match_detections

SHOULDER_RATIO_THRESHOLD = 0.80


def load_variant_csvs(data_dir: Path, variants: list[str]) -> dict[str, pd.DataFrame]:
    dfs = {"fp32": pd.read_csv(data_dir / "synth_fp32.csv")}
    for v in variants:
        dfs[v] = pd.read_csv(data_dir / f"synth_{v}.csv")
    return dfs


def build_character_map(fp32_df: pd.DataFrame, arrangement_path: Path) -> dict:
    """Returns dict: fp32_df.index -> character, via rank-matching
    ground-truth x_m order against FP32 detection box-center-x order."""
    with open(arrangement_path, encoding="utf-8") as f:
        arrangement = json.load(f)
    arr_by_id = {a["sample_id"]: a for a in arrangement}

    char_map = {}
    mismatches = 0
    for filename, g in fp32_df.groupby("filename"):
        sample_id = filename.rsplit(".", 1)[0]
        arr = arr_by_id.get(sample_id)
        if arr is None:
            continue
        people = sorted(arr["people"], key=lambda p: p["x_m"])
        if len(people) != len(g):
            mismatches += 1
            continue
        box_xc = (g["box_x1"] + g["box_x2"]) / 2
        order = box_xc.sort_values().index
        for rank, idx in enumerate(order):
            char_map[idx] = people[rank]["character"]
    if mismatches:
        print(f"  [warn] {mismatches} images had a people-count mismatch vs arrangement.json (skipped)")
    return char_map


def _pixel_dist(a_x, a_y, b_x, b_y):
    return np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)


def per_detection_table(fp32_df: pd.DataFrame, variant_df: pd.DataFrame, char_map: dict) -> pd.DataFrame:
    """One row per matched (fp32, variant) detection pair: shoulder/wrist
    absolute pixel error (averaged left+right), each side's own
    R = wrist_dist/shoulder_dist, the shoulder-size ratio (for the
    failure-mode filter), character, distance, and n_people."""
    m = match_detections(fp32_df, variant_df)
    ref = fp32_df.loc[m["ref_idx"]].reset_index(drop=True)
    oth = variant_df.loc[m["other_idx"]].reset_index(drop=True)

    shoulder_err = 0.5 * (
        _pixel_dist(ref.left_shoulder_x, ref.left_shoulder_y, oth.left_shoulder_x, oth.left_shoulder_y)
        + _pixel_dist(ref.right_shoulder_x, ref.right_shoulder_y, oth.right_shoulder_x, oth.right_shoulder_y)
    )
    wrist_err = 0.5 * (
        _pixel_dist(ref.left_wrist_x, ref.left_wrist_y, oth.left_wrist_x, oth.left_wrist_y)
        + _pixel_dist(ref.right_wrist_x, ref.right_wrist_y, oth.right_wrist_x, oth.right_wrist_y)
    )

    shoulder_dist_fp32 = _pixel_dist(ref.left_shoulder_x, ref.left_shoulder_y, ref.right_shoulder_x, ref.right_shoulder_y)
    wrist_dist_fp32 = _pixel_dist(ref.left_wrist_x, ref.left_wrist_y, ref.right_wrist_x, ref.right_wrist_y)
    shoulder_dist_var = _pixel_dist(oth.left_shoulder_x, oth.left_shoulder_y, oth.right_shoulder_x, oth.right_shoulder_y)
    wrist_dist_var = _pixel_dist(oth.left_wrist_x, oth.left_wrist_y, oth.right_wrist_x, oth.right_wrist_y)

    return pd.DataFrame({
        "filename": m["filename"].values,
        "distance_cm": ref["distance_cm"].values,
        "n_people": ref["n_people_expected"].values,
        "character": m["ref_idx"].map(char_map).values,
        "shoulder_err": shoulder_err.values,
        "wrist_err": wrist_err.values,
        "R_fp32": (wrist_dist_fp32 / shoulder_dist_fp32).values,
        "R_var": (wrist_dist_var / shoulder_dist_var).values,
        "shoulder_ratio": (shoulder_dist_var / shoulder_dist_fp32).values,
    })


def build_all_variant_detections(dfs: dict, variants: list[str], char_map: dict,
                                  threshold: float = SHOULDER_RATIO_THRESHOLD) -> dict:
    """Runs per_detection_table for every variant vs fp32, applies the
    shoulder-ratio filter, prints an exclusion-count summary, and returns
    {variant: filtered_df}."""
    per_detection = {v: per_detection_table(dfs["fp32"], dfs[v], char_map) for v in variants}
    filtered = {
        v: df[df["shoulder_ratio"] >= threshold].reset_index(drop=True)
        for v, df in per_detection.items()
    }
    for v in variants:
        excluded = len(per_detection[v]) - len(filtered[v])
        print(f"  {v}: excluded {excluded}/{len(per_detection[v])} detections (shoulder_ratio < {threshold})")
    return filtered


def per_image_then_across_images(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Average within each image (filename) first, then average across
    images grouped by group_cols (equal weight per image, not per person)."""
    per_image = df.groupby("filename").agg(
        shoulder_err=("shoulder_err", "mean"),
        wrist_err=("wrist_err", "mean"),
        **{c: (c, "first") for c in group_cols},
    ).reset_index()
    agg = per_image.groupby(group_cols).agg(
        shoulder=("shoulder_err", "mean"),
        wrist=("wrist_err", "mean"),
    ).reset_index()
    agg["ratio"] = agg["wrist"] / agg["shoulder"]
    return agg


def overall_shoulder_wrist(df: pd.DataFrame) -> tuple[float, float]:
    """Per-image-then-overall mean shoulder/wrist error (all distances/n_people collapsed)."""
    per_image = df.groupby("filename").agg(shoulder_err=("shoulder_err", "mean"), wrist_err=("wrist_err", "mean"))
    return per_image["shoulder_err"].mean(), per_image["wrist_err"].mean()


def pooled_within_character_std(values: pd.Series, characters: pd.Series) -> float:
    """Pooled (ANOVA-style) within-group std across characters:
    sqrt( sum((n_i-1)*var_i) / sum(n_i-1) )."""
    df = pd.DataFrame({"v": np.asarray(values), "c": np.asarray(characters)})
    num, den = 0.0, 0
    for _, g in df.groupby("c")["v"]:
        n = len(g)
        if n > 1:
            num += (n - 1) * g.var(ddof=1)
            den += (n - 1)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def r_stats(det_df: pd.DataFrame) -> dict:
    """Given an already shoulder_ratio-filtered per-detection table for one
    variant, compute R_mean, R_std, natural_variation_std, quant_noise_std,
    NSR -- all using pooled within-character variance."""
    delta_R = det_df["R_var"] - det_df["R_fp32"]
    natural_std = pooled_within_character_std(det_df["R_fp32"], det_df["character"])
    quant_std = pooled_within_character_std(delta_R, det_df["character"])
    r_std = pooled_within_character_std(det_df["R_var"], det_df["character"])
    return {
        "R_mean": det_df["R_var"].mean(),
        "R_std": r_std,
        "natural_variation_std": natural_std,
        "quant_noise_std": quant_std,
        "NSR": quant_std / natural_std,
    }
