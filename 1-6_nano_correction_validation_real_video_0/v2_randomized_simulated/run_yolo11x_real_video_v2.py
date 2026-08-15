#!/usr/bin/env python3
"""v2: same as run_yolo11x_real_video.py for the native 250/300/350/400cm
distances (read directly from Test_21/*.mp4, unchanged), but for the
275-550cm simulated range, reads from the pre-generated, randomized
multi-source frames in real_video_frames_simulated_v2/ (see
generate_randomized_simulated_frames.py) instead of doing an in-memory
single-nearest-source rescale. This makes yolo11x-pose (the correction
target) and v8n-pose+variants (the deployed model) get evaluated on the
EXACT SAME simulated images for the far-distance range, and removes the
same single-source-contamination issue from the target curve that was
just fixed for the deployed model.
"""
import re
import time
from pathlib import Path

import torch
import _cuda_dll_fix  # noqa: F401 -- must be imported after torch, before onnxruntime
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from preprocess import preprocess_array
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "Test_21"
ONNX_PATH = ROOT / "onnx_exports" / "yolo11x-pose.onnx"
OUT_CSV = ROOT / "xtier_fp32_pray_results" / "yolo11x-pose.real_video_v2.csv"
SIM_FRAMES_DIR = Path(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\real_video_frames_simulated_v2")

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CSV_HEADER = (
    ["source_video", "frame_idx", "distance_cm", "n_people_expected", "person_idx",
     "box_conf", "box_x1", "box_y1", "box_x2", "box_y2"]
    + [f"{kp}_{suffix}" for kp in KEYPOINT_NAMES for suffix in ("x", "y", "conf")]
)
CONF_THRES, IOU_THRES = 0.25, 0.45
MAX_SESSION_RETRIES = 5

DISTANCE_VIDEO_RE = re.compile(r"^(\d)(\d{3})\.mp4$")
NATIVE_DISTANCES = [250, 300, 350, 400]
SIM_DISTANCES = [275, 325, 375, 425, 450, 475, 500, 525, 550]
N_PEOPLE_GROUPS = [1, 2, 3]


def make_session(onnx_path: Path) -> ort.InferenceSession:
    last_err = None
    for attempt in range(1, MAX_SESSION_RETRIES + 1):
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
            input_meta = sess.get_inputs()[0]
            dtype = np.float16 if "float16" in input_meta.type else np.float32
            dummy = np.zeros((1, 3, 640, 640), dtype=dtype)
            sess.run(None, {input_meta.name: dummy})
            if attempt > 1:
                print(f"    [make_session] recovered on attempt {attempt}", flush=True)
            return sess
        except Exception as e:
            last_err = e
            print(f"    [make_session] attempt {attempt}/{MAX_SESSION_RETRIES} failed "
                  f"({type(e).__name__}), retrying with a fresh session...", flush=True)
    raise RuntimeError(
        f"CUDA session for {onnx_path.name} failed {MAX_SESSION_RETRIES} times in a row. Last error: {last_err}"
    )


def run_frame(sess, input_name, input_dtype, frame_bgr):
    orig_shape = frame_bgr.shape
    batch = preprocess_array(frame_bgr, 640)[None, ...].astype(input_dtype)
    raw = sess.run(None, {input_name: batch})[0]
    pred = torch.from_numpy(raw.astype(np.float32))

    dets = non_max_suppression(
        pred, conf_thres=CONF_THRES, iou_thres=IOU_THRES, nc=1, end2end=False, max_det=300
    )[0]
    if len(dets) == 0:
        return []

    boxes = ops.scale_boxes((640, 640), dets[:, :4].clone(), orig_shape)
    kpts_xy = dets[:, 6:].view(len(dets), 17, 3)[..., :2].clone()
    kpts_xy = ops.scale_coords((640, 640), kpts_xy, orig_shape)
    kpts_conf = dets[:, 6:].view(len(dets), 17, 3)[..., 2]

    results = []
    for i in range(len(dets)):
        results.append({
            "box_conf": dets[i, 4].item(),
            "box": boxes[i].tolist(),
            "kpts_xy": kpts_xy[i].tolist(),
            "kpts_conf": kpts_conf[i].tolist(),
        })
    return results


def write_row(f, source_video, frame_idx, distance_cm, n_people, person_idx, det):
    row = [source_video, frame_idx, distance_cm, n_people, person_idx, f"{det['box_conf']:.6f}"]
    row += [f"{v:.4f}" for v in det["box"]]
    for (x, y), c in zip(det["kpts_xy"], det["kpts_conf"]):
        row += [f"{x:.4f}", f"{y:.4f}", f"{c:.6f}"]
    f.write(",".join(str(v) for v in row) + "\n")


def main():
    sess = make_session(ONNX_PATH)
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_dtype = np.float16 if "float16" in input_meta.type else np.float32

    OUT_CSV.parent.mkdir(exist_ok=True)
    t0 = time.time()
    n_written = 0
    n_frames_processed = 0

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(CSV_HEADER) + "\n")

        # --- native distances: unchanged, direct from Test_21 videos ---
        videos = sorted(VIDEO_DIR.glob("*.mp4"))
        targets = []
        for v in videos:
            m = DISTANCE_VIDEO_RE.match(v.name)
            if m and int(m.group(2)) in NATIVE_DISTANCES:
                targets.append((v, int(m.group(1)), int(m.group(2))))
        print(f"Found {len(targets)} native-distance videos", flush=True)

        for video_path, n_people, source_d in targets:
            source_video = video_path.stem
            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = run_frame(sess, input_name, input_dtype, frame)
                for person_idx, det in enumerate(dets):
                    write_row(f, source_video, frame_idx, source_d, n_people, person_idx, det)
                n_written += len(dets)
                n_frames_processed += 1
                frame_idx += 1
                if n_frames_processed % 500 == 0:
                    elapsed = time.time() - t0
                    rate = n_frames_processed / elapsed if elapsed > 0 else 0
                    print(f"  [native] {n_frames_processed} frames ({rate:.1f}/s, {n_written} dets)", flush=True)
            cap.release()
            print(f"  done {source_video}: {frame_idx} frames", flush=True)

        # --- simulated distances: read from the pre-generated randomized-source PNGs ---
        for n_people in N_PEOPLE_GROUPS:
            for target_d in SIM_DISTANCES:
                frame_dir = SIM_FRAMES_DIR / f"{n_people}{target_d}"
                frame_paths = sorted(frame_dir.glob("frame_*.png"))
                source_video = f"{n_people}{target_d}_simv2"
                for fp in frame_paths:
                    frame_idx = int(fp.stem.split("_")[1])
                    frame_bgr = cv2.imread(str(fp))
                    dets = run_frame(sess, input_name, input_dtype, frame_bgr)
                    for person_idx, det in enumerate(dets):
                        write_row(f, source_video, frame_idx, target_d, n_people, person_idx, det)
                    n_written += len(dets)
                    n_frames_processed += 1
                    if n_frames_processed % 500 == 0:
                        elapsed = time.time() - t0
                        rate = n_frames_processed / elapsed if elapsed > 0 else 0
                        print(f"  [sim] {n_frames_processed} frame-instances ({rate:.1f}/s, {n_written} dets)", flush=True)
                print(f"  done n={n_people} target={target_d}cm: {len(frame_paths)} frames", flush=True)

    elapsed = time.time() - t0
    print(f"\nDONE {OUT_CSV.name}: {n_frames_processed} frame-instances, {n_written} detections, "
          f"{elapsed:.1f}s ({elapsed/max(n_frames_processed,1)*1000:.1f}ms/frame-instance)", flush=True)


if __name__ == "__main__":
    main()
