#!/usr/bin/env python3
"""Same as batch_infer_simulated_far.py, pointed at the v2 randomized-source
simulated frames (real_video_frames_simulated_v2) -- each target simulated
distance now draws from a random mix of ALL 7 real source distances
(100-400cm), individually rescaled per-frame based on its own true capture
distance, rather than always rescaling from one fixed nearest-real-distance
source. See generate_randomized_simulated_frames.py for how these frames
were built and the 475cm keypoint-mislocalization investigation that
motivated the change (a single source video's blind spot was dominating
one entire simulated-distance bin under the old method).

Run with the SYSTEM python3:
    /usr/bin/python3 batch_infer_simulated_far_v2.py
"""
import csv
import re
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from batch_infer_synthetic import (  # noqa: E402
    TRTEngine, decode_pose_output, detection_to_original_space, COCO_KEYPOINT_NAMES,
)

FRAMES_DIR = Path.home() / "project/quant_test/real_video_frames_simulated_v2"
ENGINES_DIR = Path.home() / "project/quant_test/engines"
OUT_DIR = Path.home() / "project/quant_test/real_video_results_simulated_v2"
CONF_THRESHOLD, IOU_THRESHOLD = 0.25, 0.7

ENGINES = {
    "fp32": ENGINES_DIR / "yolov8n-pose.fp32.engine",
    "fp16": ENGINES_DIR / "yolov8n-pose.fp16.engine",
    "int8": ENGINES_DIR / "yolov8n-pose.int8.engine",
    "modelopt_int8": ENGINES_DIR / "yolov8n-pose.modelopt_int8.engine",
    "modelopt_int8_excl_cv4": ENGINES_DIR / "yolov8n-pose.modelopt_int8_excl_cv4.engine",
    "modelopt_int8_excl_cv4_fp32": ENGINES_DIR / "yolov8n-pose.modelopt_int8_excl_cv4_fp32.engine",
}

VIDEO_RE = re.compile(r"^(\d)(\d{3})$")


def video_meta(stem: str) -> dict:
    m = VIDEO_RE.match(stem)
    assert m, f"unexpected video stem: {stem!r}"
    return {"n_people": int(m.group(1)), "distance_cm": int(m.group(2))}


def detection_to_row(det_orig: dict, source_video: str, frame_idx: int, meta: dict, person_idx: int) -> dict:
    kpts = det_orig["keypoints"]
    box = det_orig["box_xyxy"]
    row = {
        "source_video": source_video,
        "frame_idx": frame_idx,
        "distance_cm": meta["distance_cm"],
        "n_people_expected": meta["n_people"],
        "person_idx": person_idx,
        "box_conf": det_orig["conf"],
        "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
    }
    for kp_idx, kp_name in enumerate(COCO_KEYPOINT_NAMES):
        row[f"{kp_name}_x"] = kpts[kp_idx, 0]
        row[f"{kp_name}_y"] = kpts[kp_idx, 1]
        row[f"{kp_name}_conf"] = kpts[kp_idx, 2]
    return row


def run_one_engine(name: str, engine_path: Path, video_dirs: list, out_csv: Path):
    print(f"\n{'='*70}\nEngine: {name} ({engine_path})\n{'='*70}", flush=True)
    engine = TRTEngine(str(engine_path))

    rows = []
    n_no_detection = 0
    t0 = time.time()
    frame_counter = 0

    all_frames = []
    for video_dir in video_dirs:
        meta = video_meta(video_dir.name)
        frame_paths = sorted(video_dir.glob("frame_*.png"))
        for fp in frame_paths:
            frame_idx = int(fp.stem.split("_")[1])
            all_frames.append((video_dir.name, frame_idx, meta, fp))
    total_frames = len(all_frames)
    print(f"{len(video_dirs)} sim-videos, {total_frames} total frames to process", flush=True)

    for source_video, frame_idx, meta, fp in all_frames:
        frame_counter += 1
        with Image.open(fp) as im:
            orig_w, orig_h = im.size

        raw = engine.infer(str(fp))
        dets = decode_pose_output(raw, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)

        if not dets:
            n_no_detection += 1
        else:
            for person_idx, det in enumerate(dets):
                det_orig = detection_to_original_space(det, orig_w, orig_h)
                rows.append(detection_to_row(det_orig, source_video, frame_idx, meta, person_idx))

        if frame_counter % 500 == 0 or frame_counter == total_frames:
            elapsed = time.time() - t0
            rate = frame_counter / elapsed if elapsed > 0 else 0
            print(f"  [{name}] {frame_counter}/{total_frames} frames ({rate:.1f} frame/s, "
                  f"{len(rows)} person-detections so far)", flush=True)

    print(f"[{name}] Done: {len(rows)} person-detections, {n_no_detection} frames with zero detections", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] Wrote {out_csv}", flush=True)


def main():
    video_dirs = sorted([d for d in FRAMES_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} simulated-frame folders in {FRAMES_DIR}", flush=True)
    for name, engine_path in ENGINES.items():
        if not engine_path.exists():
            print(f"[SKIP] engine not found: {engine_path}", flush=True)
            continue
        out_csv = OUT_DIR / f"sim_{name}.csv"
        if out_csv.exists():
            print(f"[SKIP] already exists: {out_csv}", flush=True)
            continue
        run_one_engine(name, engine_path, video_dirs, out_csv)
    print("\nALL ENGINES DONE", flush=True)


if __name__ == "__main__":
    main()
