"""
v3: fixes two problems found in the v2 pipeline:
  1. own_mean/own_std (the deployed model's own per-distance normalization used
     in the R_corrected formula) was computed from SYNTHETIC data, not real
     video -- real-video own_std turns out to be ~2-3x smaller than the
     synthetic-derived value at matching distances (checked directly against
     real_video_results_filtered), which distorts the correction.
  2. v2's 9 simulated distances independently drew 300 samples each (with
     replacement ACROSS distances) from the same 2100-frame pool, so any two
     distance bins shared ~14% of their source frames -- not fully independent.

Fix: split the 2100-frame pool (per n_people group) into two DISJOINT halves
up front -- a calibration half and an evaluation half -- so calibration
(own_mean/own_std and the yolo11x target curve) and evaluation (the R values
whose miss rate actually gets reported) never share a single source frame.
Within each half, every frame is independently assigned to exactly one of the
13 target distances (250..550, 25cm steps -- this now includes the 4 distances
that v1/v2 treated as "native, unscaled" real footage; here they go through
the same pool-draw + rescale mechanism, with scale=1.0 when target==source by
chance) and rescaled from its own true capture distance, matching the same
shoulder_px(d) = K/(d+offset) model used in the v2 pipeline.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

VIDEO_DIR = Path(r"C:\Users\user\pose_quant_env\Test_21")
OUT_DIR = Path(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\real_video_frames_calib_eval_split")
OUT_DIR.mkdir(exist_ok=True)

with open(r"C:\Users\user\pose_quant_env\J_exp_data\1-6_nano_correction_validation_real_video_0\v2_randomized_simulated\shoulder_px_model.json") as f:
    model_params = json.load(f)
K, OFFSET = model_params["K"], model_params["offset"]


def shoulder_px(d):
    return K / (d + OFFSET)


REAL_DISTANCES = [100, 150, 200, 250, 300, 350, 400]
TARGET_DISTANCES = [250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550]
N_PEOPLE_GROUPS = [1, 2, 3]
FRAMES_PER_SOURCE = 300

manifest_rows = []

for n in N_PEOPLE_GROUPS:
    pool = [(sd, fi) for sd in REAL_DISTANCES for fi in range(FRAMES_PER_SOURCE)]
    random.shuffle(pool)
    half = len(pool) // 2
    splits = {"calib": pool[:half], "eval": pool[half:]}

    print(f"n_people={n}: pool={len(pool)}, calib={len(splits['calib'])}, eval={len(splits['eval'])}")

    for split_name, split_pool in splits.items():
        # independent scale assignment: each sample independently draws one target
        assignments = [(sd, fi, random.choice(TARGET_DISTANCES)) for sd, fi in split_pool]

        # group by (source_distance) to minimize video open/seek overhead
        by_source = {}
        for sd, fi, td in assignments:
            by_source.setdefault(sd, []).append((fi, td))

        out_idx_counters = {}
        for sd, items in by_source.items():
            video_path = VIDEO_DIR / f"{n}{sd}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            for fi, td in sorted(items):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    print(f"WARNING: failed to read frame {fi} from {video_path}")
                    continue
                scale = shoulder_px(td) / shoulder_px(sd)
                h, w = frame.shape[:2]
                if scale < 1.0:
                    new_w, new_h = round(w * scale), round(h * scale)
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
                    pad_x, pad_y = (w - new_w) // 2, (h - new_h) // 2
                    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
                elif scale > 1.0:
                    # target closer than source -- upscale (can happen now since
                    # source/target are independently drawn, unlike v2 which only
                    # ever went farther)
                    canvas = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    new_w, new_h = round(w * scale), round(h * scale)
                    upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    # center-crop back to original canvas size
                    x0 = (new_w - w) // 2
                    y0 = (new_h - h) // 2
                    canvas = upscaled[max(y0, 0):max(y0, 0) + h, max(x0, 0):max(x0, 0) + w]
                    if canvas.shape[:2] != (h, w):
                        # edge case: extreme upscale ratio, pad if crop came up short
                        tmp = np.full((h, w, 3), 114, dtype=np.uint8)
                        ch, cw = canvas.shape[:2]
                        tmp[:ch, :cw] = canvas[:h, :w]
                        canvas = tmp
                else:
                    canvas = frame

                out_sub = OUT_DIR / split_name / f"{n}{td}"
                out_sub.mkdir(parents=True, exist_ok=True)
                out_idx = out_idx_counters.get(td, 0)
                out_name = f"frame_{out_idx:04d}_src{sd}_orig{fi}.png"
                cv2.imwrite(str(out_sub / out_name), canvas)
                manifest_rows.append({
                    "split": split_name, "n_people": n, "target_distance_cm": td, "out_file": out_name,
                    "source_distance_cm": sd, "source_frame_idx": fi, "scale_factor": scale,
                })
                out_idx_counters[td] = out_idx + 1
            cap.release()

        for td in TARGET_DISTANCES:
            cnt = out_idx_counters.get(td, 0)
            print(f"  [{split_name}] n={n} target={td}cm: {cnt} frames")

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
print(f"\nDONE. {len(manifest)} total frames written. Manifest saved to {OUT_DIR / 'manifest.csv'}")

# sanity check: confirm calib and eval never share a (n_people, source_distance, source_frame_idx)
calib_keys = set(zip(manifest[manifest.split == "calib"].n_people, manifest[manifest.split == "calib"].source_distance_cm, manifest[manifest.split == "calib"].source_frame_idx))
eval_keys = set(zip(manifest[manifest.split == "eval"].n_people, manifest[manifest.split == "eval"].source_distance_cm, manifest[manifest.split == "eval"].source_frame_idx))
overlap = calib_keys & eval_keys
print(f"calib/eval source-frame overlap: {len(overlap)} (should be 0)")
