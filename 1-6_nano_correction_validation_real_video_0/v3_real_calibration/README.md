# v3: real-video calibration for the distance-adaptive threshold (2026-08-15)

Supersedes `v2_randomized_simulated/` for the calibration methodology (v2's
randomized-multi-source frame generation was correct and is reused here
unchanged -- see `generate_calib_eval_split_frames.py`, which extends v2's
approach to also cover the 4 native distances 250/300/350/400 through the
same pool-draw + rescale mechanism, and to a disjoint calib/eval split).

## What was wrong with v1/v2 (found and fixed in this order)

1. **own_mean/own_std computed from synthetic data, not real video.** The
   deployed model's own per-distance R normalization (used in the
   `R_corrected = x11_mean + (R_raw - own_mean) * (x11_std/own_std)`
   formula) was fit on the 1-3 experiment's synthetic CSVs in both v1 and
   v2 -- inherited unchanged from the original 1-3/1-5 methodology, never
   revisited when v2 fixed the *target* curve (x11) to use real video
   instead of synthetic. Checked directly: real own_std is ~2-3x SMALLER
   than the synthetic-derived value at matching native distances (e.g.
   250cm: synthetic std=0.132 vs real std=0.040 for fp32). Fixed: own_mean/
   own_std now computed from real video too (the calib split, below).

2. **Same-sample-reuse risk.** Naively computing own_mean/own_std from the
   SAME real-video sample used to evaluate miss rate would reintroduce a
   leakage-adjacent concern. Fixed: `generate_calib_eval_split_frames.py`
   splits the 2100-frame real-footage pool (per n_people group) into two
   DISJOINT halves before any target-distance assignment -- confirmed via
   the manifest that no single source frame (n_people, source_distance,
   source_frame_idx) appears more than once anywhere in the 6300-frame set.
   `analyze_v3_real_calibration.py` re-partitions on top of this (see #4)
   but always from disjoint pools, so leakage-freedom holds regardless of
   the exact ratio used.

3. **Low-confidence garbage detections contaminating R.** Found by pulling
   actual frames for the most extreme R outliers (R>1, sometimes >3, vs a
   normal ~0.2-0.45): the model was locking onto background clutter (a
   wardrobe, consistently) instead of the person, at box_conf ~0.26-0.34
   and wrist_conf as low as 0.07, while genuine detections run box_conf
   0.90+ and wrist_conf 0.76+. These aren't real pose measurements and were
   skewing miss rate at whichever distance bin they happened to land in.
   Fixed: `analyze_v3_real_calibration.py` filters with the SAME confidence
   thresholds the deployed `beta_S6_latency_harness.py` already uses
   (`CONF_SHOULDER_THRESHOLD=0.5`, `CONF_WRIST_THRESHOLD=0.4`, both L/R) --
   this offline analysis pipeline had simply never applied that filter, the
   deployed system was never missing it.

4. **50/50 calib/eval split wastes eval precision for no calibration
   benefit.** Bootstrap-checked: own_mean/own_std estimates are already
   stable at n≈250 per distance (barely moves vs n≈1000). Meanwhile miss
   rate is a rare-event proportion that benefits directly from more eval
   data regardless of how uniform the gesture is. `analyze_v3_real_
   calibration.py` re-splits at `CALIB_FRACTION=0.2` (still from disjoint
   pools) instead of 50/50, so evaluation gets 4x the data for the same
   total footage.

## Methodology check on this PC (FP32 only, before committing to a Nano run)

Ran yolov8n-pose FP32 via plain onnxruntime (not the actual deployed
TensorRT precision -- methodology validation only) through fixes #1-4 in
sequence, on the eval split, mean-across-distance corrected miss rate:

| Stage | Peak miss rate | Where | Std across distances |
|---|---|---|---|
| Paper's current number (v1/v2, synthetic own_std) | 9.1% | 475cm | -- |
| Real own_mean/own_std, no split (same-sample reuse) | 11.26% | 250cm | erratic |
| + disjoint 50/50 split | 19.49% | 525cm | erratic |
| + rebalanced 20/80 split (more eval data) | 27.60% | 525cm | erratic (worse -- proved it wasn't noise) |
| + confidence filter (fix #3) | **5.31%** | **550cm** | **1.68** (down from ~8.5) |

The confidence filter was the one that actually mattered -- the 450/525cm
spikes were background-misdetection outliers, not a real distance effect,
and not fixed by more data (more data made the spike worse, which is what
pointed at "real outliers, not sampling noise" rather than "underpowered
split"). Final pattern (0.4%-5.3%, peaking at the farthest distance) is the
first one that actually looks like a sane distance-difficulty curve instead
of noise.

**This was FP32-only, on this PC, to validate the methodology before
spending Nano time.** It does NOT tell us what the actual deployed
precision variants (fp16, int8, modelopt...) will show -- that's what the
Nano run is for.

## Pipeline order

1. `generate_calib_eval_split_frames.py` -- already run, produced `frames/`
   (6300 PNGs, 3.1GB) and `frame_manifest.csv`. Don't need to rerun this
   unless the source footage changes.
2. **Copy `frames/` to the Nano.** 3.1GB, however you normally transfer
   files there.
3. **On the Nano**: `batch_infer_calib_eval_split.py` -- needs
   `batch_infer_synthetic.py` (the `TRTEngine` helper) copied alongside it
   from wherever the Nano's existing quant_test scripts live, and the 6
   `.engine` files at `~/project/quant_test/engines/`. Produces
   `results/{fp32,fp16,int8,modelopt_int8,modelopt_int8_excl_cv4,
   modelopt_int8_excl_cv4_fp32}.csv`.
4. **Copy `results/*.csv` back to this PC** (into this same `results/`
   folder).
5. **On the PC**: `infer_yolo11x_reference.py` -- onnxruntime CUDA, no Nano
   needed. Produces `results/yolo11x_fp32.csv`.
6. **On the PC**: `analyze_v3_real_calibration.py` -- combines both,
   applies the confidence filter and calib/eval re-split, prints the final
   per-variant miss-rate tables, writes
   `results/v3_final_miss_rate_by_variant.csv`.

## After this

Once the real deployed-precision numbers are in, main.tex's IV-C paragraph
5 (currently "stays under 10% throughout the full 250--550cm range, peaking
at 9.1% near 475cm") needs updating to whatever this pipeline reports for
the FP16/mixed-INT8 variants actually cited there -- don't reuse the FP32
methodology-check numbers above, they're not the deployment precision.
Fig 12 and the real-video validation prose in reviewer_concerns_prep.md
(journal_paper/) will also need revisiting once the real numbers land.
