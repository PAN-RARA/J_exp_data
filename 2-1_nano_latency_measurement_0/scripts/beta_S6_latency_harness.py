#!/usr/bin/env python3
"""
2-1 end-to-end latency harness (v2: full speed, per-stage segmented timing).

Replays a single continuous concatenated video per people-count scenario
(~/project/quant_test/concat_videos/concat_{n}p.mp4 -- the 7 distance clips
for that people-count spliced back to back) through beta_S6.py's EXACT
business/decision logic -- tracking, confidence/boundary/overlap filters,
frontal check, CONFIRM_FRAMES debounce, map_value(), RESEND_INTERVAL throttle
-- copied verbatim from ~/project/src/beta_S6.py.

v1 paced playback to 30fps to mimic the real camera's cap.set(CAP_PROP_FPS,
30). Dropped: capping at 30fps just reproduces a number already known in
advance (the system comfortably keeps up at 30fps for all 3 precisions/people
counts) and hides the actual processing ceiling. This version runs flat out
-- no pacing -- so fps_mean/fps_1pct_low reflect genuine throughput headroom,
and a long unpaced run also gives a chance to catch sustained-load thermal
throttling (checked at the end: first-half vs second-half mean latency).

v1 also bundled everything into two cumulative checkpoints (yolo_ms, cmd_ms),
which doesn't answer "slow at which stage". This version stamps 6 boundaries
per frame: t_capture (loop start, mirrors beta_S6.py's t_capture semantics --
set BEFORE cap.read(), so decode cost is counted, not silently skipped like
v1's generator-based prefetch did) -> t_read (cap.read() done: real video-
decode/"camera capture" cost) -> t_preprocess (letterbox+normalize done) ->
t_gpu (TensorRT H2D+execute+D2H done) -> t_decode (NMS+keypoint-extract+
coord-remap done) -> t_logic (tracking/filters/frontal/confirm/decision done,
every frame) -> t_cmd_sent (throttled by RESEND_INTERVAL, same as production).

The only thing swapped out from real beta_S6.py is the inference call:
ultralytics' YOLO() can't load our modelopt-built engines (confirmed
empirically -- crashes in NMS postprocessing on their output-tensor
metadata), so this uses the raw TensorRT pipeline (TRTEngine +
decode_pose_output) from batch_infer_synthetic.py instead, exactly as
1-2~1-6 already do. No physical serial device is needed: beta_S6.py stamps
t_cmd_sent and updates video_to_cmd_ms unconditionally, whether or not the
actual write succeeds, so this harness does the same and just skips the
write itself.

Needs BOTH cv2 and tensorrt/pycuda in the same process; the Nano's two
existing venvs each have only one half (YOLOvenv: cv2/flask/pyserial, no
pycuda; system python3: tensorrt/pycuda, no cv2), so this runs under a venv
built with `python3 -m venv --system-site-packages ~/latency_venv` (inherits
system python3's tensorrt/pycuda) + `pip install opencv-python-headless`:
    ~/latency_venv/bin/python3 beta_S6_latency_harness.py --engine <path> --n-people 1
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / "project/quant_test/scripts"))
from batch_infer_synthetic import TRTEngine, decode_pose_output, detection_to_original_space  # noqa: E402
import pycuda.driver as cuda  # noqa: E402

# ==============================
# beta_S6.py constants, copied verbatim
# ==============================
BOUNDARY_MARGIN          = 30
CONF_SHOULDER_THRESHOLD  = 0.5
CONF_WRIST_THRESHOLD     = 0.4
IOU_OVERLAP_THRESHOLD    = 0.1
MARGIN_RATIO             = 0.15
VERTICAL_OFFSET_RATIO    = 1.0
MATCH_DISTANCE_THRESHOLD = 80
THRESHOLD_RATIO          = 0.5
CONFIRM_SECONDS          = 0.2
FAN_THRESHOLD_DEFAULT    = 3
RESEND_INTERVAL          = 0.3
ZERO_TIMEOUT             = 10

FPS_NOMINAL_FOR_CONFIRM = 30.0  # CONFIRM_FRAMES is derived from the real camera's
                                 # nominal fps in production (fps_from_cap), not from
                                 # however fast this harness happens to run -- keep it
                                 # fixed so CONFIRM_SECONDS means the same real-world
                                 # debounce window regardless of test throughput.
CONFIRM_FRAMES = max(1, int(CONFIRM_SECONDS * FPS_NOMINAL_FOR_CONFIRM))
TARGET_SIZE    = 640

CONCAT_VIDEOS_DIR = Path.home() / "project/quant_test/concat_videos"

STAGES = ["capture_ms", "preprocess_ms", "gpu_ms", "decode_ms", "logic_ms"]


# ==============================
# Fast in-memory preprocessing (cv2-based) -- mirrors build_int8_engine.py's
# letterbox()+preprocess_image() math exactly, but operates on an in-memory
# BGR frame from cv2.VideoCapture instead of re-opening a file with PIL.
# ==============================
def preprocess_frame(frame_bgr, target_size=TARGET_SIZE, pad_value=114):
    h, w = frame_bgr.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr)


def gpu_infer(engine: TRTEngine, batch: np.ndarray) -> np.ndarray:
    """Just the H2D copy + execute + D2H copy, given an already-preprocessed
    batch -- split out from preprocessing so the two can be timed separately."""
    batch = batch[None, ...]
    assert batch.shape == engine.input_shape, f"{batch.shape} != {engine.input_shape}"
    cuda.memcpy_htod_async(engine.d_input, batch, engine.stream)
    engine.context.execute_async_v3(stream_handle=engine.stream.handle)
    output = np.empty(engine.output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, engine.d_output, engine.stream)
    engine.stream.synchronize()
    return output[0]


# ==============================
# beta_S6.py helper functions, copied verbatim
# ==============================
def is_within_boundary(x, y, w, h, margin):
    return margin < x < (w - margin) and margin < y < (h - margin)


def shoulder_center(lsx, lsy, rsx, rsy):
    return (lsx + rsx) / 2, (lsy + rsy) / 2


def match_to_track(cx, cy, prev_pos, threshold):
    best_id, best_dist = None, float("inf")
    for tid, (px, py) in prev_pos.items():
        dist = math.hypot(cx - px, cy - py)
        if dist < best_dist:
            best_dist = dist
            best_id = tid
    return best_id if best_dist <= threshold else None


def is_frontal(lsx, rsx, lwx, rwx, lsy, rsy, lwy, rwy, shoulder_dist):
    if shoulder_dist <= 0:
        return False
    margin = MARGIN_RATIO * shoulder_dist
    left_turn = lwx > lsx + margin
    right_turn = rwx < rsx - margin
    horizontal_ok = not (left_turn or right_turn)
    shoulder_mid_y = (lsy + rsy) / 2
    wrist_mid_y = (lwy + rwy) / 2
    vertical_ok = abs(wrist_mid_y - shoulder_mid_y) <= VERTICAL_OFFSET_RATIO * shoulder_dist
    return horizontal_ok and vertical_ok


def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def get_overlapping_indices(boxes, iou_threshold):
    overlapping = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if compute_iou(boxes[i], boxes[j]) >= iou_threshold:
                overlapping.add(i)
                overlapping.add(j)
    return overlapping


def map_value(hands_together, total_people, fan_threshold):
    H = max(0, int(hands_together))
    N = max(1, int(total_people))
    T = max(1, int(fan_threshold))
    if N <= T:
        value = round(1535 * H / N)
    else:
        if H <= N - 3:
            value = round(1535 * H / (N - 3))
        elif H == N - 2:
            value = 1536
        elif H == N - 1:
            value = 1537
        else:
            value = 1538
    return value


def run_scenario(engine_path: Path, n_people: int, loops: int, out_dir: Path):
    print(f"\n{'='*70}\nengine={engine_path.name} n_people={n_people} loops={loops} (full speed, no pacing)\n{'='*70}")
    engine = TRTEngine(str(engine_path))
    video_path = CONCAT_VIDEOS_DIR / f"concat_{n_people}p.mp4"
    if not video_path.exists():
        raise SystemExit(f"missing concatenated video: {video_path}")

    prev_positions = {}
    consecutive_counts = {}
    confirmed_tracker = {}
    next_track_id = [0]
    zero_start_time = None
    last_send_time = 0.0

    stage_samples = {k: [] for k in STAGES}
    total_ms_samples = []   # t_logic - t_capture: full per-frame pipeline, every frame
    cmd_ms_samples = []     # t_cmd_sent - t_capture: throttled by RESEND_INTERVAL, like production
    frame_idx = 0

    for loop_i in range(loops):
        cap = cv2.VideoCapture(str(video_path))
        try:
            while True:
                t_capture = time.time()
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                t_read = time.time()

                orig_h, orig_w = frame_bgr.shape[:2]

                batch = preprocess_frame(frame_bgr)
                t_preprocess = time.time()

                raw = gpu_infer(engine, batch)
                t_gpu = time.time()

                dets = decode_pose_output(raw, conf_threshold=0.25, iou_threshold=0.7)
                dets_orig = [detection_to_original_space(d, orig_w, orig_h) for d in dets]
                t_decode = time.time()

                total_people = 0
                hands_together = 0
                current_frame_positions = {}
                used_track_ids = set()

                boxes = [d["box_xyxy"] for d in dets_orig]
                overlapping_indices = get_overlapping_indices(boxes, IOU_OVERLAP_THRESHOLD)

                for p_i, det in enumerate(dets_orig):
                    kpts = det["keypoints"]  # (17,3) x,y,conf
                    lsx, lsy, conf_ls = kpts[5]
                    rsx, rsy, conf_rs = kpts[6]
                    lwx, lwy, conf_lw = kpts[9]
                    rwx, rwy, conf_rw = kpts[10]
                    cx, cy = shoulder_center(lsx, lsy, rsx, rsy)

                    matched_id = match_to_track(cx, cy, prev_positions, MATCH_DISTANCE_THRESHOLD)
                    if matched_id in used_track_ids:
                        matched_id = None
                    if matched_id is not None:
                        track_id = matched_id
                    else:
                        track_id = next_track_id[0]
                        next_track_id[0] += 1
                        consecutive_counts[track_id] = 0
                        confirmed_tracker[track_id] = False
                    used_track_ids.add(track_id)
                    current_frame_positions[track_id] = (cx, cy)

                    shoulder_conf_ok = conf_ls >= CONF_SHOULDER_THRESHOLD and conf_rs >= CONF_SHOULDER_THRESHOLD
                    wrist_conf_ok = conf_lw >= CONF_WRIST_THRESHOLD and conf_rw >= CONF_WRIST_THRESHOLD
                    conf_ok = shoulder_conf_ok and wrist_conf_ok

                    boundary_ok = (is_within_boundary(lsx, lsy, orig_w, orig_h, BOUNDARY_MARGIN) and
                                   is_within_boundary(rsx, rsy, orig_w, orig_h, BOUNDARY_MARGIN) and
                                   is_within_boundary(lwx, lwy, orig_w, orig_h, BOUNDARY_MARGIN) and
                                   is_within_boundary(rwx, rwy, orig_w, orig_h, BOUNDARY_MARGIN))
                    overlap_ok = p_i not in overlapping_indices

                    shoulder_dist = math.hypot(lsx - rsx, lsy - rsy)
                    wrist_dist = math.hypot(lwx - rwx, lwy - rwy)

                    if not overlap_ok or not conf_ok or not boundary_ok:
                        consecutive_counts[track_id] = 0
                        continue

                    ratio = wrist_dist / shoulder_dist if shoulder_dist > 0 else 999
                    frontal = is_frontal(lsx, rsx, lwx, rwx, lsy, rsy, lwy, rwy, shoulder_dist)
                    hands_now = (ratio < THRESHOLD_RATIO) and frontal
                    if hands_now:
                        consecutive_counts[track_id] += 1
                    else:
                        consecutive_counts[track_id] = 0

                    confirmed = consecutive_counts[track_id] >= CONFIRM_FRAMES
                    if confirmed and not confirmed_tracker[track_id]:
                        confirmed_tracker[track_id] = True

                    total_people += 1
                    if confirmed:
                        hands_together += 1

                prev_positions = current_frame_positions

                now2 = time.time()
                if total_people == 0:
                    if zero_start_time is None:
                        zero_start_time = now2
                    desired_value = 2000 if (now2 - zero_start_time >= ZERO_TIMEOUT) else \
                        map_value(hands_together, total_people, FAN_THRESHOLD_DEFAULT)
                else:
                    zero_start_time = None
                    desired_value = 0 if hands_together == 0 else \
                        map_value(hands_together, total_people, FAN_THRESHOLD_DEFAULT)
                t_logic = time.time()

                stage_samples["capture_ms"].append((t_read - t_capture) * 1000)
                stage_samples["preprocess_ms"].append((t_preprocess - t_read) * 1000)
                stage_samples["gpu_ms"].append((t_gpu - t_preprocess) * 1000)
                stage_samples["decode_ms"].append((t_decode - t_gpu) * 1000)
                stage_samples["logic_ms"].append((t_logic - t_decode) * 1000)
                total_ms_samples.append((t_logic - t_capture) * 1000)

                if desired_value is not None and (now2 - last_send_time) >= RESEND_INTERVAL:
                    t_cmd_sent = time.time()
                    # send_serial() write itself skipped (no MCU attached) -- beta_S6.py
                    # stamps this timing unconditionally regardless of serial state
                    last_send_time = now2
                    cmd_ms_samples.append((t_cmd_sent - t_capture) * 1000)

                frame_idx += 1
                if frame_idx % 500 == 0:
                    print(f"  {frame_idx} frames, running total_ms mean={np.mean(total_ms_samples):.1f}ms "
                          f"(capture={np.mean(stage_samples['capture_ms']):.1f} "
                          f"preprocess={np.mean(stage_samples['preprocess_ms']):.1f} "
                          f"gpu={np.mean(stage_samples['gpu_ms']):.1f} "
                          f"decode={np.mean(stage_samples['decode_ms']):.1f} "
                          f"logic={np.mean(stage_samples['logic_ms']):.1f})")
        finally:
            cap.release()

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else None

    n = len(total_ms_samples)
    half = n // 2
    first_half_mean = float(np.mean(total_ms_samples[:half])) if half > 0 else None
    second_half_mean = float(np.mean(total_ms_samples[half:])) if (n - half) > 0 else None
    throttle_drift_pct = (100.0 * (second_half_mean - first_half_mean) / first_half_mean
                           if first_half_mean else None)

    total_time_s = sum(total_ms_samples) / 1000.0
    fps_mean = n / total_time_s if total_time_s > 0 else None
    slowest_1pct = sorted(total_ms_samples, reverse=True)[:max(1, n // 100)]
    fps_1pct_low = 1000.0 / (sum(slowest_1pct) / len(slowest_1pct))

    result = {
        "engine": engine_path.name,
        "n_people": n_people,
        "n_frames": n,
        "n_cmd_samples": len(cmd_ms_samples),
        "fps_mean": fps_mean,
        "fps_1pct_low": fps_1pct_low,
        "total_ms_mean": float(np.mean(total_ms_samples)),
        "total_ms_p50": pct(total_ms_samples, 50),
        "total_ms_p95": pct(total_ms_samples, 95),
        "total_ms_p99": pct(total_ms_samples, 99),
        "cmd_ms_mean": float(np.mean(cmd_ms_samples)) if cmd_ms_samples else None,
        "cmd_ms_p50": pct(cmd_ms_samples, 50),
        "cmd_ms_p95": pct(cmd_ms_samples, 95),
        "cmd_ms_p99": pct(cmd_ms_samples, 99),
        "throttle_check_first_half_ms": first_half_mean,
        "throttle_check_second_half_ms": second_half_mean,
        "throttle_check_drift_pct": throttle_drift_pct,
    }
    for stage in STAGES:
        arr = stage_samples[stage]
        result[f"{stage}_mean"] = float(np.mean(arr))
        result[f"{stage}_p50"] = pct(arr, 50)
        result[f"{stage}_p95"] = pct(arr, 95)

    print(json.dumps(result, indent=2))
    if throttle_drift_pct is not None and throttle_drift_pct > 10:
        print(f"  [WARNING] second-half mean is {throttle_drift_pct:.1f}% slower than first-half -- "
              f"possible thermal throttling under sustained load")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame_csv = out_dir / f"{engine_path.stem}_n{n_people}_frames.csv"
    with open(per_frame_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx"] + STAGES + ["total_ms"])
        rows = zip(range(n), *[stage_samples[s] for s in STAGES], total_ms_samples)
        w.writerows(rows)

    with open(out_dir / "summary.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--n-people", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--out-dir", default=str(Path.home() / "project/quant_test/latency_results"))
    args = ap.parse_args()

    run_scenario(Path(args.engine), args.n_people, args.loops, Path(args.out_dir))


if __name__ == "__main__":
    main()
