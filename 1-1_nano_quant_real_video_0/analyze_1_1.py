#!/usr/bin/env python3
"""
analyze_1_1.py
==============
Reproduces the Notion 1-1 "全量測試" numbers (誤觸率 / 事件判定達標率 / 骨架
異常率) directly from the per-frame CSVs in 1-1_csv.7z. Extract that archive
into a sibling folder named "1-1_csv" (i.e. next to this script) before
running.

Also produces Fig 1-1-1 (R value vs distance, 4 precisions, per-track points
+ within-track errorbar, overlaid on the per-distance mean). This is 1-1's
only chart -- everything else stays as tables (false-trigger rate is a
one-line text callout, not a figure; file-size/speed/accuracy are compact
enough as tables per journal convention). Fig 1-1-1 uses R directly rather
than the pass/fail achievement rate, since achievement rate's
CONFIRM_FRAMES smoothing hides R's underlying drift/noise until it finally
tips over the threshold -- within this pilot's limited 100-400cm range that
made FP16/ModelOpt mixed look "mostly fine," which undersells why the
systematic distance-adaptive investigation in 1-2 onward is needed. R is
also the same metric 1-3 onward standardizes on, so this makes a tighter
bridge to that section.

Each CSV is one (precision, scenario) pair, filename "{precision}_{scenario}.csv":
  precision in {"32","16","8","mix"} = FP32 / FP16 / legacy INT8 / ModelOpt mixed
  scenario  in:
    - "{n_people}{distance_cm}"  e.g. "3400" = 3 people at 400cm (distance matrix,
      n_people in 1-3, distance_cm in 100..400 step 50)
    - "1s" / "2s"                 side-walking, 1 or 2 people
    - "3hd" / "3au"                3 people, hands-down / arms-crossed (false-trigger tests)
    - "3w"                         3 people, chaotic movement

Each row is one tracked person in one frame. track_id is NOT stable across
precisions/runs -- it's just this run's local track counter, and includes
spurious short-lived "ghost" tracks from detection noise. A track is treated
as a REAL person only if it appears in >=90% of the scenario's frames (real
tracks are always exactly n_people_expected, this threshold just filters out
the ghosts, it's not doing any delicate selection). This matches the
Hungarian/consecutive-frame-confirmation logic beta_S6.py already applies on
the Jetson -- we're not re-deriving "confirmed", just aggregating a column
that logic already computed per frame.

Two different aggregations, matched to what each scenario type answers:
  - False-trigger scenarios (1s/2s/3hd/3au): person is NOT doing the gesture,
    so any confirmed==1 is a false positive. Aggregated as "does ANY real
    track in this frame show confirmed==1" (frame-level OR, not per-track
    mean) -- a false alarm on any one person in frame is still one false
    alarm for that frame, matching how the deployed system would actually
    trigger the atomizer once per frame regardless of how many people are in
    view.
  - Distance-matrix scenarios (n_people+distance_cm): person IS doing the
    gesture throughout, so we want the per-person hit rate -- aggregated as
    mean(confirmed) per real track, kept separate (not pooled across people
    in the same scene), since the whole point of this experiment is that
    quantization damage is NOT uniform across people in the same frame.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.labelsize"] = 13
matplotlib.rcParams["xtick.labelsize"] = 11
matplotlib.rcParams["ytick.labelsize"] = 11
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "1-1_csv"
CHARTS_DIR = Path(__file__).parent / "charts"
PRECISION_LABELS = {"32": "FP32", "16": "FP16", "8": "legacy INT8", "mix": "ModelOpt mixed"}
PRECISION_ORDER = ["32", "16", "8", "mix"]
PRECISION_COLOR = {"32": "#555555", "16": "#4C72B0", "8": "#C44E52", "mix": "#55A868"}
# IEEE print figures fall back to grayscale -- pair each precision with its
# own marker shape so the 4 series stay distinguishable by shape alone.
PRECISION_MARKER = {"32": "o", "16": "s", "8": "^", "mix": "D"}
PRECISION_LINESTYLE = {"32": "-", "16": "--", "8": "-.", "mix": ":"}
# draw-order (z-order) only -- front to back: int8 > fp32 > fp16 > mixed.
# Kept separate from PRECISION_ORDER so the legend order is unaffected.
PRECISION_ZORDER = {"8": 5, "32": 4, "16": 3, "mix": 2}
REAL_TRACK_FRAME_FRACTION = 0.90

FALSE_TRIGGER_SCENARIOS = ["1s", "2s", "3hd", "3au"]
DISTANCE_CM = [100, 150, 200, 250, 300, 350, 400]
DISTANCE_SCENARIOS = [f"{n}{d}" for n in (1, 2, 3) for d in DISTANCE_CM]


def scenario_n_people(scenario: str) -> int:
    m = re.match(r"^(\d)", scenario)
    if not m:
        raise ValueError(f"can't parse n_people from scenario {scenario!r}")
    return int(m.group(1))


def load(precision: str, scenario: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{precision}_{scenario}.csv")


def real_tracks(df: pd.DataFrame) -> list[int]:
    total_frames = df["frame"].nunique()
    counts = df["track_id"].value_counts()
    real = counts[counts >= REAL_TRACK_FRAME_FRACTION * total_frames]
    return sorted(real.index.tolist(), key=lambda t: -counts[t])


def false_trigger_rate(precision: str, scenario: str) -> tuple[int, int]:
    """Returns (n_frames_with_any_confirmed, n_total_frames). Uses ALL
    tracks, not just "real" ones -- under heavy quantization damage a
    person's track can fragment into many short-lived track_ids (seen on
    legacy INT8 "2s", where the longest fragment covers only ~69% of
    frames), so no single track would pass the real_tracks() persistence
    filter even though a person is genuinely there throughout. A false
    confirm on a fragment is still a real false alarm, so it must count."""
    df = load(precision, scenario)
    per_frame_any = df.groupby("frame")["confirmed"].max()
    return int(per_frame_any.sum()), len(per_frame_any)


def ghost_skeleton_rate(precision: str, scenario: str) -> tuple[int, int, int]:
    """Returns (n_frames_with_extra_skeletons, n_total_frames, max_raw_num_people)."""
    df = load(precision, scenario)
    n_expected = scenario_n_people(scenario)
    per_frame_raw = df.groupby("frame")["raw_num_people"].first()
    ghost = (per_frame_raw > n_expected).sum()
    return int(ghost), len(per_frame_raw), int(per_frame_raw.max())


def per_track_hit_rate(precision: str, scenario: str) -> list[float]:
    """Returns per-real-track mean(confirmed), ordered by track persistence
    (most-persistent track first) -- NOT pooled across people."""
    df = load(precision, scenario)
    real = real_tracks(df)
    return [df[df["track_id"] == t]["confirmed"].mean() for t in real]


def savefig(fig, name):
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(CHARTS_DIR / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg, +.pdf)")


def per_track_ratio_stats(precision: str, scenario: str) -> list[tuple[float, float]]:
    """Returns per-real-track (mean_R, std_R) across that track's own frames
    -- this within-track std is legitimate (repeated measurement of ONE
    person's held pose, not a population summary), unlike averaging across
    the few distinct tracks/people themselves."""
    df = load(precision, scenario)
    real = real_tracks(df)
    out = []
    for t in real:
        r = df[df["track_id"] == t]["ratio"]
        out.append((r.mean(), r.std()))
    return out


def make_distance_degradation_chart():
    """Fig 1-1-1: R value (wrist_dist/shoulder_dist) vs distance, 4
    precisions, 1-3 people pooled, 250-400cm real video.

    Uses R directly instead of the pass/fail achievement rate. Achievement
    rate is a CONFIRM_FRAMES-smoothed binary outcome -- it absorbs R's
    underlying drift/noise and only shows a problem once that drift finally
    tips it over the 0.4 threshold for long enough. Within the real video
    pilot's limited 100-400cm range, that means FP16/ModelOpt mixed still
    read as "mostly fine" on achievement rate even though R itself is
    already visibly drifting and getting noisier under quantization -- R is
    the earlier, more sensitive signal, and it's also the same metric 1-3
    onward standardizes on, so this makes a tighter bridge to that section
    than achievement rate did. Each track's own ~300 correlated frames are
    first collapsed to that track's mean R (a legitimate within-track
    summary), and those per-track means are then pooled into one mean +/-
    SD per distance/precision (settled: no independence concern once
    collapsed to per-track means -- see discussion) rather than plotted as
    separate per-track points."""
    CHARTS_DIR.mkdir(exist_ok=True)
    CHART_DISTANCE_CM = [d for d in DISTANCE_CM if d >= 250]
    rows = []
    for scenario in DISTANCE_SCENARIOS:
        n = scenario_n_people(scenario)
        d = int(scenario[1:])
        if d not in CHART_DISTANCE_CM:
            continue
        for p in PRECISION_ORDER:
            for track_idx, (r_mean, r_std) in enumerate(per_track_ratio_stats(p, scenario)):
                rows.append({"distance_cm": d, "n_people": n, "precision": p,
                             "track_idx": track_idx, "r_mean": r_mean})
    df = pd.DataFrame(rows)

    # Marker shape + linestyle carry series identity (grayscale-safe, no
    # reliance on color); error bars are thin/capless/low-alpha so they don't
    # visually collide at the 4 sparse ticks -- they're background context,
    # not something that itself needs to be individually distinguishable.
    fig, ax = plt.subplots(figsize=(11, 6.0))
    # Embeds at 3.4in single-column width; fonts are back-calculated so they
    # land at the intended final size once LaTeX shrinks it down (see Fig 5's
    # _SHRINK5 in analyze_1_3.py for the same pattern).
    _SHRINK9 = 3.4 / 10.90
    # Only 4 distinct x-values (250/300/350/400); 0.15 margin pulled the
    # first/last points too far in from the axes edges, so tightened to 0.06.
    # y tightened too -- default ~5% autoscale margin plus the headroom added
    # below was leaving too much dead space above/below the data.
    ax.margins(x=0.06, y=0.06)
    for p in PRECISION_ORDER:
        sub = df[df["precision"] == p]
        agg = sub.groupby("distance_cm")["r_mean"].agg(["mean", "std"])
        agg["std"] = agg["std"].fillna(0)
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt="none",
                    ecolor=PRECISION_COLOR[p], elinewidth=1, capsize=3, capthick=1, alpha=0.35, zorder=1)
        ax.plot(agg.index, agg["mean"], color=PRECISION_COLOR[p], linestyle=PRECISION_LINESTYLE[p],
                marker=PRECISION_MARKER[p], markersize=14, linewidth=2.2, zorder=PRECISION_ZORDER[p], label=PRECISION_LABELS[p])
    # Not labeled in the legend -- explained in the caption instead, one less
    # entry to fit alongside the 4 precision levels.
    ax.axhline(0.4, color="black", linestyle="--", linewidth=1, alpha=0.6)
    # Headroom above the threshold line for the legend -- just enough to
    # clear the tallest error bar (legacy INT8 at x=400), not more, now that
    # margins() above handles the rest of the spacing.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.08)
    ax.set_ylim(top=0.48)
    ax.set_xlabel("distance (cm)", fontsize=9 / _SHRINK9)
    ax.set_ylabel("R (wrist_dist / shoulder_dist)", fontsize=9 / _SHRINK9)
    ax.set_xticks(CHART_DISTANCE_CM)
    ax.tick_params(labelsize=8 / _SHRINK9)
    # 2x2 grid now that "threshold" moved out of the legend (4 precision
    # levels left) -- plenty of headroom here (only 4 sparse x-ticks, no
    # other in-plot text), so the legend can stay at a generous,
    # body-text-adjacent size.
    ax.legend(fontsize=8 / _SHRINK9, loc="upper left", ncol=2,
              columnspacing=1.0, handletextpad=0.5, handlelength=1.8, labelspacing=0.4)
    fig.tight_layout()
    savefig(fig, "1-1-1_r_value_vs_distance")


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"expected extracted CSVs at {DATA_DIR} -- extract 1-1_csv.7z here first")

    make_distance_degradation_chart()

    print("=" * 100)
    print("1. False-trigger rate (1s/2s/3hd/3au) -- frame-level, ANY real track confirmed")
    print("=" * 100)
    for scenario in FALSE_TRIGGER_SCENARIOS:
        print(f"-- {scenario} --")
        for p in PRECISION_ORDER:
            hits, total = false_trigger_rate(p, scenario)
            print(f"  {PRECISION_LABELS[p]:<16} {hits}/{total} = {100*hits/total:.4f}%")

    print()
    print("=" * 100)
    print("2. Ghost-skeleton rate for 3hd (raw_num_people > expected n_people)")
    print("=" * 100)
    for p in PRECISION_ORDER:
        ghost, total, max_raw = ghost_skeleton_rate(p, "3hd")
        print(f"  {PRECISION_LABELS[p]:<16} {ghost}/{total} = {100*ghost/total:.1f}%  (max raw_num_people={max_raw})")

    print()
    print("=" * 100)
    print("3. Per-track hit rate, distance matrix (400cm shown -- edit DISTANCE_CM filter for others)")
    print("=" * 100)
    for scenario in [s for s in DISTANCE_SCENARIOS if s.endswith("400")]:
        n = scenario_n_people(scenario)
        print(f"-- {scenario} ({n} people, 400cm) --")
        for p in PRECISION_ORDER:
            rates = per_track_hit_rate(p, scenario)
            rates_str = " / ".join(f"{100*r:.1f}%" for r in rates)
            print(f"  {PRECISION_LABELS[p]:<16} {rates_str}")

    print()
    print("=" * 100)
    print("4. Per-track hit rate, ALL distances (close-range sanity check, all should be high)")
    print("=" * 100)
    for scenario in DISTANCE_SCENARIOS:
        n = scenario_n_people(scenario)
        row = [scenario]
        for p in PRECISION_ORDER:
            rates = per_track_hit_rate(p, scenario)
            row.append(" / ".join(f"{100*r:.1f}%" for r in rates))
        print(f"  {row[0]:<8} " + "  |  ".join(f"{PRECISION_LABELS[p]}: {row[i+1]}" for i, p in enumerate(PRECISION_ORDER)))

    print("\nDONE")


if __name__ == "__main__":
    main()
