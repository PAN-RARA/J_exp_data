"""
Regenerate the simulated far-distance frames (275-550cm) using a randomized,
multi-source approach: instead of always rescaling from ONE fixed nearest
real distance (the original simulate_far_distance_frames.py's method, which
let a single source video's blind spot contaminate an entire simulated-
distance bin -- see the 475cm keypoint-mislocalization investigation this
session), each target simulated distance now draws a RANDOM sample from ALL
7 real source distances (100-400cm), and each sampled frame is individually
rescaled based on its OWN true capture distance.

Scale factor model: shoulder_px = K / (d + offset), fit on REAL 100-400cm
data (not synthetic) -- R^2=0.99946, see fit_shoulder_px_model.py. Using
real data avoids relying on synthetic-to-real transfer for the geometry
model itself.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

random.seed(42)
np.random.seed(42)

VIDEO_DIR = Path(r"C:\Users\user\pose_quant_env\Test_21")
OUT_DIR = Path(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\real_video_frames_simulated_v2")
OUT_DIR.mkdir(exist_ok=True)

with open(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\shoulder_px_model.json") as f:
    model_params = json.load(f)
K, OFFSET = model_params["K"], model_params["offset"]


def shoulder_px(d):
    return K / (d + OFFSET)


REAL_DISTANCES = [100, 150, 200, 250, 300, 350, 400]
TARGET_DISTANCES = [275, 325, 375, 425, 450, 475, 500, 525, 550]
N_PEOPLE_GROUPS = [1, 2, 3]
FRAMES_PER_SOURCE = 300
SAMPLES_PER_TARGET = 300  # matches original per-bin sample size

manifest_rows = []

for n in N_PEOPLE_GROUPS:
    # pool: every (source_distance, frame_idx) combination for this n_people group
    pool = [(sd, fi) for sd in REAL_DISTANCES for fi in range(FRAMES_PER_SOURCE)]

    for target_d in TARGET_DISTANCES:
        out_sub = OUT_DIR / f"{n}{target_d}"
        out_sub.mkdir(exist_ok=True)

        # random sample WITHOUT replacement from the full 7-distance pool
        sampled = random.sample(pool, SAMPLES_PER_TARGET)
        # group by source video to minimize video-open/seek overhead
        by_source = {}
        for sd, fi in sampled:
            by_source.setdefault(sd, []).append(fi)

        out_idx = 0
        for sd, frame_indices in by_source.items():
            video_path = VIDEO_DIR / f"{n}{sd}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            scale = shoulder_px(target_d) / shoulder_px(sd)
            for fi in sorted(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    print(f"WARNING: failed to read frame {fi} from {video_path}")
                    continue
                h, w = frame.shape[:2]
                new_w, new_h = round(w * scale), round(h * scale)
                if scale < 1.0:
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
                    pad_x, pad_y = (w - new_w) // 2, (h - new_h) // 2
                    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
                else:
                    # target closer than source (shouldn't happen here since all
                    # targets are farther than all sources, but guard anyway)
                    canvas = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)

                out_name = f"frame_{out_idx:04d}_src{sd}_orig{fi}.png"
                cv2.imwrite(str(out_sub / out_name), canvas)
                manifest_rows.append({
                    "n_people": n, "target_distance_cm": target_d, "out_file": out_name,
                    "source_distance_cm": sd, "source_frame_idx": fi, "scale_factor": scale,
                })
                out_idx += 1
            cap.release()

        print(f"n={n} target={target_d}cm: wrote {out_idx} frames, "
              f"sourced from {len(by_source)} distances ({dict((k, len(v)) for k, v in by_source.items())})")

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
print(f"\nDONE. {len(manifest)} total frames written. Manifest saved to {OUT_DIR / 'manifest.csv'}")
