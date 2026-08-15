"""
Runs ONNX inference (onnxruntime CUDA) on the v3 calib/eval split frames for
yolo11x-pose FP32 (the reference/target model only). This is meant to run on
this PC -- yolo11x is never deployed/quantized, it's only ever used as the
FP32 correction target, so plain onnxruntime is a faithful reproduction, no
TensorRT engine needed. The deployed-model precision variants (fp32, fp16,
legacy int8, modelopt int8/mixed) need the actual TensorRT engines and must
be run on the Nano instead -- see batch_infer_calib_eval_split.py in this
same folder for that half.

Earlier validation runs (yolov8n-pose FP32 via plain onnxruntime, methodology
check only) confirmed the v3 approach -- see README.md -- but that model's
inference doesn't belong in this script going forward.
"""
import re
import time
from pathlib import Path

import torch
import _cuda_dll_fix  # noqa: F401
import cv2
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_array
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(r"C:\Users\user\pose_quant_env")
SPLIT_DIR = Path(__file__).parent / "frames"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "yolo11x_fp32": ROOT / "onnx_exports" / "yolo11x-pose.onnx",
}

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CSV_HEADER = (
    ["split", "n_people", "target_distance_cm", "frame_file", "person_idx",
     "box_conf", "box_x1", "box_y1", "box_x2", "box_y2"]
    + [f"{kp}_{suffix}" for kp in KEYPOINT_NAMES for suffix in ("x", "y", "conf")]
)
CONF_THRES, IOU_THRES = 0.25, 0.45
MAX_SESSION_RETRIES = 5


def make_session(onnx_path):
    last_err = None
    for attempt in range(1, MAX_SESSION_RETRIES + 1):
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
            input_meta = sess.get_inputs()[0]
            dtype = np.float16 if "float16" in input_meta.type else np.float32
            dummy = np.zeros((1, 3, 640, 640), dtype=dtype)
            sess.run(None, {input_meta.name: dummy})
            return sess
        except Exception as e:
            last_err = e
            print(f"    [make_session] attempt {attempt}/{MAX_SESSION_RETRIES} failed ({type(e).__name__}), retrying...", flush=True)
    raise RuntimeError(f"CUDA session failed {MAX_SESSION_RETRIES} times. Last error: {last_err}")


def run_frame(sess, input_name, input_dtype, frame_bgr):
    orig_shape = frame_bgr.shape
    batch = preprocess_array(frame_bgr, 640)[None, ...].astype(input_dtype)
    raw = sess.run(None, {input_name: batch})[0]
    pred = torch.from_numpy(raw.astype(np.float32))
    dets = non_max_suppression(pred, conf_thres=CONF_THRES, iou_thres=IOU_THRES, nc=1, end2end=False, max_det=300)[0]
    if len(dets) == 0:
        return []
    boxes = ops.scale_boxes((640, 640), dets[:, :4].clone(), orig_shape)
    kpts_xy = dets[:, 6:].view(len(dets), 17, 3)[..., :2].clone()
    kpts_xy = ops.scale_coords((640, 640), kpts_xy, orig_shape)
    kpts_conf = dets[:, 6:].view(len(dets), 17, 3)[..., 2]
    results = []
    for i in range(len(dets)):
        results.append({
            "box_conf": dets[i, 4].item(), "box": boxes[i].tolist(),
            "kpts_xy": kpts_xy[i].tolist(), "kpts_conf": kpts_conf[i].tolist(),
        })
    return results


def write_row(f, split, n_people, target_d, frame_file, person_idx, det):
    row = [split, n_people, target_d, frame_file, person_idx, f"{det['box_conf']:.6f}"]
    row += [f"{v:.4f}" for v in det["box"]]
    for (x, y), c in zip(det["kpts_xy"], det["kpts_conf"]):
        row += [f"{x:.4f}", f"{y:.4f}", f"{c:.6f}"]
    f.write(",".join(str(v) for v in row) + "\n")


DIR_RE = re.compile(r"^(\d)(\d{3})$")


def main():
    for model_name, onnx_path in MODELS.items():
        out_csv = OUT_DIR / f"{model_name}.csv"
        if out_csv.exists():
            print(f"[SKIP] {out_csv} already exists")
            continue
        print(f"\n=== {model_name} ({onnx_path}) ===", flush=True)
        sess = make_session(onnx_path)
        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        input_dtype = np.float16 if "float16" in input_meta.type else np.float32

        t0 = time.time()
        n_written = 0
        n_frames = 0
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(CSV_HEADER) + "\n")
            for split in ["calib", "eval"]:
                split_dir = SPLIT_DIR / split
                target_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
                for td_dir in target_dirs:
                    m = DIR_RE.match(td_dir.name)
                    n_people, target_d = int(m.group(1)), int(m.group(2))
                    frame_paths = sorted(td_dir.glob("frame_*.png"))
                    for fp in frame_paths:
                        frame_bgr = cv2.imread(str(fp))
                        dets = run_frame(sess, input_name, input_dtype, frame_bgr)
                        for person_idx, det in enumerate(dets):
                            write_row(f, split, n_people, target_d, fp.name, person_idx, det)
                        n_written += len(dets)
                        n_frames += 1
                        if n_frames % 500 == 0:
                            elapsed = time.time() - t0
                            rate = n_frames / elapsed if elapsed > 0 else 0
                            print(f"  [{model_name}] {n_frames} frames ({rate:.1f}/s, {n_written} dets)", flush=True)
        elapsed = time.time() - t0
        print(f"DONE {model_name}: {n_frames} frames, {n_written} detections, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
