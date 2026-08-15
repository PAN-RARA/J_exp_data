import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REAL_DIR = r"C:\Users\user\pose_quant_env\J_exp_data\1-6_nano_correction_validation_real_video_0\exported_charts\1-6\real_video_results_filtered"

df = pd.read_csv(f"{REAL_DIR}/fp32.csv")
df["shoulder_px"] = np.sqrt((df["left_shoulder_x"] - df["right_shoulder_x"]) ** 2 +
                             (df["left_shoulder_y"] - df["right_shoulder_y"]) ** 2)

curve = df.groupby("distance_cm")["shoulder_px"].agg(["mean", "std", "count"]).reset_index()
print("empirical shoulder_px by real distance:")
print(curve.round(2).to_string(index=False))


def model(d, K, offset):
    return K / (d + offset)


popt, pcov = curve_fit(model, curve["distance_cm"], curve["mean"], p0=[30000, 0])
K, offset = popt
pred = model(curve["distance_cm"], K, offset)
resid = curve["mean"] - pred
r2 = 1 - np.sum(resid ** 2) / np.sum((curve["mean"] - curve["mean"].mean()) ** 2)
print(f"\nfit: shoulder_px = {K:.1f} / (d + {offset:.1f}), R^2={r2:.5f}")
print("fitted vs actual:")
for d, actual, p in zip(curve["distance_cm"], curve["mean"], pred):
    print(f"  d={d:>4}  actual={actual:>7.2f}  fitted={p:>7.2f}  resid={actual-p:>6.2f} ({100*(actual-p)/actual:>5.2f}%)")

print("\nextrapolated shoulder_px at simulated target distances:")
for d in [275, 325, 375, 425, 450, 475, 500, 525, 550]:
    print(f"  d={d}: {model(d, K, offset):.2f}")

import json
with open(r"C:\Users\user\AppData\Local\Temp\claude\C--Users-user\9ac3c4f3-cab3-4585-9dda-eb94953ba32c\scratchpad\shoulder_px_model.json", "w") as f:
    json.dump({"K": K, "offset": offset, "r2": r2}, f)
print("\nsaved model params to shoulder_px_model.json")
