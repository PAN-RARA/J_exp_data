#!/usr/bin/env python3
"""
match_detections.py
=======================
Robust cross-variant detection matching for the multi-person synthetic pose
data (1-2_hands_down/, 1-3_palms_together/).

THE PROBLEM this replaces: the CSVs' `person_idx` column is NOT a stable
cross-run identity. It's just "the Nth row this particular inference run
happened to output for this image" -- two different precision runs on the
SAME image are not guaranteed to list detections in the same order (e.g. if
internal ordering is confidence-rank based and confidence shifts slightly
under quantization). Joining on person_idx directly (what an early version
of the analysis did) can silently pair keypoints from TWO DIFFERENT
PHYSICAL PEOPLE in multi-person scenes. That contamination produces large,
keypoint-type-agnostic pseudo-error that swamps the real, smaller,
quantization-specific wrist-vs-shoulder asymmetry signal -- this was
confirmed empirically: single-person scenes (n_people_expected==1, zero
matching ambiguity by construction) show a clean 1.4-2.5x wrist/shoulder
asymmetry matching the original Jetson project's finding, while the
person_idx-joined multi-person scenes showed ~1.0 (no asymmetry) with
absolute errors 10-30x larger than the single-person subset.

WHY HUNGARIAN ON BOX IoU, NOT 1D X-SORT: an earlier alternative (sort each
image's detections by shoulder-midpoint x, pair by rank) is better than
trusting person_idx but has its own blind spot -- two people close together
in x can still have their order flipped by quantization-induced box shift
(measured up to 100-200px in the worst legacy-INT8 case), which can exceed
the actual x-gap between two people in a crowded synthetic scene. Box IoU
uses 2D position AND size/shape, not one coordinate, and Hungarian
assignment (scipy.optimize.linear_sum_assignment) finds the GLOBALLY
optimal one-to-one pairing across all detections in the image at once,
rather than a greedy or rank-based pairing.

Matches below MIN_IOU are REJECTED, not forced. A low-IoU "best available"
pairing likely means the two detections aren't the same person (a missed
detection, an extra false detection, or a box shift so large it's no longer
trustworthy as "the same detection") -- forcing it would reintroduce
exactly the contamination this module exists to prevent. Unmatched
detections are reported as a diagnostic, not silently dropped without a
trace: callers should look at how many ref/other detections had no
acceptable match, since that's itself informative (e.g. quantization
causing a person to be missed entirely).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN_IOU = 0.3
BOX_COLS = ["box_x1", "box_y1", "box_x2", "box_y2"]


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_* are (N,4) arrays of [x1,y1,x2,y2]. Returns (N_a, N_b) IoU matrix."""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


def match_one_image(ref_g: pd.DataFrame, other_g: pd.DataFrame):
    """Returns list of (ref_index, other_index, iou) for accepted matches
    (IoU >= MIN_IOU), using Hungarian assignment on the full IoU matrix
    (maximize total IoU across the whole image at once, not greedy)."""
    if len(ref_g) == 0 or len(other_g) == 0:
        return []

    ref_boxes = ref_g[BOX_COLS].to_numpy()
    other_boxes = other_g[BOX_COLS].to_numpy()
    iou = box_iou_matrix(ref_boxes, other_boxes)

    row_ind, col_ind = linear_sum_assignment(-iou)  # maximize IoU == minimize -IoU

    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= MIN_IOU:
            matches.append((ref_g.index[r], other_g.index[c], iou[r, c]))
    return matches


def match_detections(ref_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """Per filename, matches ref_df detections to other_df detections via
    Hungarian box-IoU assignment. Returns columns [filename, ref_idx,
    other_idx, iou] -- one row per ACCEPTED match only. Unmatched detections
    on either side are simply absent (not paired to anything) -- compare
    len(result) against len(ref_df)/len(other_df) to see how many were
    dropped."""
    all_matches = []
    for filename, ref_g in ref_df.groupby("filename"):
        other_g = other_df[other_df["filename"] == filename]
        for ref_idx, other_idx, iou in match_one_image(ref_g, other_g):
            all_matches.append({
                "filename": filename, "ref_idx": ref_idx, "other_idx": other_idx, "iou": iou,
            })
    return pd.DataFrame(all_matches, columns=["filename", "ref_idx", "other_idx", "iou"])
