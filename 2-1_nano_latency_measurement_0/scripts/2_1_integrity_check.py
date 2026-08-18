#!/usr/bin/env python3
"""2-1 latency battery: completeness/integrity check across the 3 power-mode
runs (MAXN_SUPER / 15W / 25W), each 9 scenarios (3 engines x 3 people)."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\user\pose_quant_env\exported_charts\2-1")
MODES = {
    "MAXN_SUPER": ROOT / "latency_results_maxn",
    "15W": ROOT / "latency_results_powermodes" / "15W",
    "25W": ROOT / "latency_results_powermodes" / "25W",
}
ENGINES = ["fp32", "fp16", "modelopt_int8_excl_cv4"]
N_PEOPLE = [1, 2, 3]
EXPECTED_FRAMES = 2100 * 5  # loops=5

STAGE_COLS = ["capture_ms", "preprocess_ms", "gpu_ms", "decode_ms", "logic_ms", "total_ms"]

problems = []
all_summary_rows = []

for mode, d in MODES.items():
    jsonl = d / "summary.jsonl"
    if not jsonl.exists():
        problems.append(f"[{mode}] MISSING summary.jsonl")
        continue
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if len(rows) != 9:
        problems.append(f"[{mode}] summary.jsonl has {len(rows)} rows, expected 9")

    seen = set()
    for r in rows:
        eng_short = r["engine"].replace("yolov8n-pose.", "").replace(".engine", "")
        key = (eng_short, r["n_people"])
        if key in seen:
            problems.append(f"[{mode}] DUPLICATE entry for {key}")
        seen.add(key)
        r["mode"] = mode
        r["engine_short"] = eng_short
        all_summary_rows.append(r)

        if r["n_frames"] != EXPECTED_FRAMES:
            problems.append(f"[{mode}] {key}: n_frames={r['n_frames']}, expected {EXPECTED_FRAMES}")

        for k, v in r.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                problems.append(f"[{mode}] {key}: field '{k}' is {v}")
            if v is None and k not in ("cmd_ms_mean", "cmd_ms_p50", "cmd_ms_p95", "cmd_ms_p99"):
                problems.append(f"[{mode}] {key}: field '{k}' is None")

        # sanity: stage means should sum close to total_ms mean (they're all per-frame
        # means so this holds approximately, not exactly, due to Jensen's-inequality-style
        # nonlinearity across frames -- flag only if grossly off)
        stage_sum = sum(r[f"{s}_mean"] for s in ["capture_ms", "preprocess_ms", "gpu_ms", "decode_ms", "logic_ms"])
        if abs(stage_sum - r["total_ms_mean"]) > 0.5:
            problems.append(f"[{mode}] {key}: stage sum ({stage_sum:.2f}ms) vs total_ms_mean "
                             f"({r['total_ms_mean']:.2f}ms) differ by {abs(stage_sum - r['total_ms_mean']):.2f}ms")

    expected_keys = {(e, n) for e in ENGINES for n in N_PEOPLE}
    missing = expected_keys - seen
    if missing:
        problems.append(f"[{mode}] MISSING scenarios: {sorted(missing)}")

    # per-frame CSVs
    for eng in ENGINES:
        for n in N_PEOPLE:
            csv_path = d / f"yolov8n-pose.{eng}_n{n}_frames.csv"
            if not csv_path.exists():
                problems.append(f"[{mode}] MISSING csv: {csv_path.name}")
                continue
            df = pd.read_csv(csv_path)
            if len(df) != EXPECTED_FRAMES:
                problems.append(f"[{mode}] {csv_path.name}: {len(df)} rows, expected {EXPECTED_FRAMES}")
            if df.isnull().values.any():
                n_null = df.isnull().sum().sum()
                problems.append(f"[{mode}] {csv_path.name}: {n_null} null values")
            for col in STAGE_COLS:
                if col not in df.columns:
                    problems.append(f"[{mode}] {csv_path.name}: missing column {col}")
                    continue
                if (df[col] < 0).any():
                    problems.append(f"[{mode}] {csv_path.name}: negative values in {col}")
                # cross-check csv mean vs summary.jsonl mean for this stage
                summary_row = next((r for r in rows if r["engine_short"] == eng and r["n_people"] == n), None)
                if summary_row:
                    csv_mean = df[col].mean()
                    json_mean = summary_row.get(f"{col}_mean")
                    if json_mean is not None and abs(csv_mean - json_mean) > 0.01:
                        problems.append(f"[{mode}] {csv_path.name}: csv {col} mean={csv_mean:.4f} != "
                                         f"summary.jsonl mean={json_mean:.4f}")

    # tegrastats log
    tegra_path = d / "tegrastats.log"
    if not tegra_path.exists():
        problems.append(f"[{mode}] MISSING tegrastats.log")
    else:
        lines = tegra_path.read_text(errors="replace").splitlines()
        if len(lines) < 50:
            problems.append(f"[{mode}] tegrastats.log suspiciously short: {len(lines)} lines")
        # check GPU temp field present (absence = GPU wasn't actually up during logging)
        gpu_missing = sum(1 for l in lines if "gpu@" not in l)
        if gpu_missing > 0:
            problems.append(f"[{mode}] tegrastats.log: {gpu_missing}/{len(lines)} lines missing 'gpu@' field "
                             f"(possible GPU-down period during logging)")
        # power field sanity
        vdd_missing = sum(1 for l in lines if "VDD_IN" not in l)
        if vdd_missing > 0:
            problems.append(f"[{mode}] tegrastats.log: {vdd_missing}/{len(lines)} lines missing VDD_IN")

print(f"\n{'='*70}\nTotal problems found: {len(problems)}\n{'='*70}")
for p in problems:
    print(" -", p)

# cross-mode sanity: gpu_ms should be roughly stable across modes for fp16/int8-mixed
# (per earlier finding), fp32 may drift a bit -- just print the comparison table
df_all = pd.DataFrame(all_summary_rows)
print(f"\n{'='*70}\ngpu_ms_mean by engine x n_people x mode\n{'='*70}")
pivot = df_all.pivot_table(index=["engine_short", "n_people"], columns="mode", values="gpu_ms_mean")
pivot = pivot[["MAXN_SUPER", "15W", "25W"]]
print(pivot.round(3).to_string())

print(f"\n{'='*70}\nn_cmd_samples by engine x n_people x mode (sanity: should be roughly RESEND_INTERVAL-throttled, similar order across modes)\n{'='*70}")
pivot2 = df_all.pivot_table(index=["engine_short", "n_people"], columns="mode", values="n_cmd_samples")
pivot2 = pivot2[["MAXN_SUPER", "15W", "25W"]]
print(pivot2.astype(int).to_string())

df_all.to_csv(ROOT / "2_1_all_27_scenarios_combined.csv", index=False)
print(f"\nSaved combined table: {ROOT / '2_1_all_27_scenarios_combined.csv'}")
