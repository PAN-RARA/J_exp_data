#!/usr/bin/env python3
"""
Runs deployed-model TensorRT inference (all precision variants) on the v3
calib/eval split frames. Run this ON THE NANO, with the frames/ folder from
this same directory copied over first (3.1GB -- transfer however you
normally move files to the Nano: scp, USB, etc.).

Why this exists: the offline validation pipeline for the real-video
correction (Section IV-C's distance-adaptive threshold) had two problems,
found and fixed in this order:
  1. own_mean/own_std (the deployed model's own per-distance normalization
     used in the R_corrected formula) was computed from SYNTHETIC data, not
     real video -- real own_std turns out to be ~2-3x smaller than the
     synthetic-derived value, distorting the correction. Fix: compute
     own_mean/own_std from real video instead.
  2. Doing that naively (same real-video sample for both calibration and
     evaluation) reintroduces a same-sample-reuse concern. Fix: split the
     real-footage pool into two DISJOINT halves up front (see
     generate_calib_eval_split_frames.py) -- one for calibration, one for
     evaluation -- so they never share a source frame.
  3. Along the way, a THIRD issue surfaced empirically: a handful of
     low-keypoint-confidence detections (the model locking onto background
     clutter instead of the person -- confirmed by pulling the actual
     frames) were producing wildly out-of-range R values (>1, sometimes >3,
     vs a normal ~0.2-0.45) and skewing the miss-rate at specific distances.
     Fix: filter detections by keypoint confidence (min of the 4 keypoints R
     depends on) before computing any R statistics -- matches the confidence
     filtering the deployed beta_S6.py pipeline already does, which this
     offline analysis script had been missing. See README.md for the numbers
     each fix produced.

This script produces the RAW per-detection CSVs (unfiltered, all confidence
levels kept) -- the keypoint-confidence filter and the calib/eval
re-partition both happen downstream in analyze_v3_real_calibration.py, not
here, so nothing about the filtering threshold is baked into these frozen
per-variant result files.

Frame folder layout (already split at generation time, but the analysis
script re-partitions its own way regardless -- see that script's docstring):
  frames/{calib,eval}/{n_people}{target_distance_cm}/frame_NNNN_src{sd}_orig{fi}.png
"""
import csv
import re
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
try:
    from batch_infer_synthetic import (  # noqa: E402
        TRTEngine, decode_pose_output, detection_to_original_space, COCO_KEYPOINT_NAMES,
    )
except ImportError:
    raise SystemExit(
        "batch_infer_synthetic.py not found next to this script -- copy it over from the "
        "Nano's existing quant_test scripts (same helper used by batch_infer_simulated_far_v2.py)."
    )

FRAMES_DIR = Path.home() / "project/quant_test/real_video_frames_calib_eval_split"
ENGINES_DIR = Path.home() / "project/quant_test/engines"
OUT_DIR = Path.home() / "project/quant_test/real_video_results_calib_eval_split"

CONF_THRESHOLD, IOU_THRESHOLD = 0.25, 0.7

ENGINES = {
    "fp32": ENGINES_DIR / "yolov8n-pose.fp32.engine",
    "fp16": ENGINES_DIR / "yolov8n-pose.fp16.engine",
    "int8": ENGINES_DIR / "yolov8n-pose.int8.engine",
    "modelopt_int8": ENGINES_DIR / "yolov8n-pose.modelopt_int8.engine",
    "modelopt_int8_excl_cv4": ENGINES_DIR / "yolov8n-pose.modelopt_int8_excl_cv4.engine",
    "modelopt_int8_excl_cv4_fp32": ENGINES_DIR / "yolov8n-pose.modelopt_int8_excl_cv4_fp32.engine",
}

DIR_RE = re.compile(r"^(\d)(\d{3})$")


def detection_to_row(det_orig: dict, split: str, n_people: int, target_d: int, frame_file: str, person_idx: int) -> dict:
    kpts = det_orig["keypoints"]
    box = det_orig["box_xyxy"]
    row = {
        "split": split, "n_people": n_people, "target_distance_cm": target_d,
        "frame_file": frame_file, "person_idx": person_idx,
        "box_conf": det_orig["conf"],
        "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
    }
    for kp_idx, kp_name in enumerate(COCO_KEYPOINT_NAMES):
        row[f"{kp_name}_x"] = kpts[kp_idx, 0]
        row[f"{kp_name}_y"] = kpts[kp_idx, 1]
        row[f"{kp_name}_conf"] = kpts[kp_idx, 2]
    return row


def run_one_engine(name: str, engine_path: Path, out_csv: Path):
    print(f"\n{'='*70}\nEngine: {name} ({engine_path})\n{'='*70}", flush=True)
    engine = TRTEngine(str(engine_path))

    all_frames = []
    for split in ["calib", "eval"]:
        split_dir = FRAMES_DIR / split
        for td_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            m = DIR_RE.match(td_dir.name)
            n_people, target_d = int(m.group(1)), int(m.group(2))
            for fp in sorted(td_dir.glob("frame_*.png")):
                all_frames.append((split, n_people, target_d, fp))
    total_frames = len(all_frames)
    print(f"{total_frames} total frames to process", flush=True)

    rows = []
    n_no_detection = 0
    t0 = time.time()
    for i, (split, n_people, target_d, fp) in enumerate(all_frames, start=1):
        with Image.open(fp) as im:
            orig_w, orig_h = im.size

        raw = engine.infer(str(fp))
        dets = decode_pose_output(raw, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)

        if not dets:
            n_no_detection += 1
        else:
            for person_idx, det in enumerate(dets):
                det_orig = detection_to_original_space(det, orig_w, orig_h)
                rows.append(detection_to_row(det_orig, split, n_people, target_d, fp.name, person_idx))

        if i % 500 == 0 or i == total_frames:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{name}] {i}/{total_frames} frames ({rate:.1f} frame/s, {len(rows)} person-detections so far)", flush=True)

    print(f"[{name}] Done: {len(rows)} person-detections, {n_no_detection} frames with zero detections", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] Wrote {out_csv}", flush=True)


def main():
    if not FRAMES_DIR.exists():
        raise SystemExit(f"expected {FRAMES_DIR} -- copy the frames/ folder over from the PC first")
    for name, engine_path in ENGINES.items():
        if not engine_path.exists():
            print(f"[SKIP] engine not found: {engine_path}", flush=True)
            continue
        out_csv = OUT_DIR / f"{name}.csv"
        if out_csv.exists():
            print(f"[SKIP] already exists: {out_csv}", flush=True)
            continue
        run_one_engine(name, engine_path, out_csv)
    print("\nALL ENGINES DONE", flush=True)


if __name__ == "__main__":
    main()
