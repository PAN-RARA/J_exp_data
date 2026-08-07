#!/usr/bin/env python3
"""
analyze_1_1.py
==============
Reproduces the Notion 1-1 "全量測試" numbers (誤觸率 / 事件判定達標率 / 骨架
異常率) directly from the per-frame CSVs in 1-1_csv.7z. Extract that archive
into a sibling folder named "1-1_csv" (i.e. next to this script) before
running.

Also produces Fig 1-1-1 (per-track achievement rate vs distance, 4
precisions, individual points overlaid on the per-distance mean to keep the
individual-variance finding visible). This is 1-1's only chart -- everything
else stays as tables (false-trigger rate is a one-line text callout, not a
figure; file-size/speed/accuracy are compact enough as tables per journal
convention). Fig 1-1-1 is the real-video pilot's distance-degradation
observation that motivates 1-2 onward's systematic investigation, not a
restatement of what 1-3/1-5/1-6 show in more depth later.

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

import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "1-1_csv"
CHARTS_DIR = Path(__file__).parent / "charts"
PRECISION_LABELS = {"32": "FP32", "16": "FP16", "8": "legacy INT8", "mix": "ModelOpt mixed"}
PRECISION_ORDER = ["32", "16", "8", "mix"]
PRECISION_COLOR = {"32": "#555555", "16": "#4C72B0", "8": "#C44E52", "mix": "#55A868"}
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
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}.png (+.svg)")


def make_distance_degradation_chart():
    """Fig 1-1-1: per-track event-detection achievement rate vs distance,
    4 precisions, 1-3 people pooled. Plotted as raw per-track scatter (not
    mean+/-SD) -- each distance/precision cell has only 1-3 tracks (n_people
    of that scenario), too few for SD to be a meaningful summary, and the
    metric itself is a bounded percentage that mostly sits near 100% with
    occasional individual crashes rather than a symmetric spread, so an
    errorbar mischaracterizes it (and can even poke past 100%). Showing the
    individual points directly is the honest representation and keeps the
    individual-variance finding visible (legacy INT8 at 400cm: one track
    near 100%, another near 4%). This is the real-video pilot observation
    that motivates the systematic distance-adaptive investigation carried
    out in 1-2 onward, not a restatement of it."""
    CHARTS_DIR.mkdir(exist_ok=True)
    CHART_DISTANCE_CM = [d for d in DISTANCE_CM if d >= 250]  # 100-200cm is flat/uninteresting, cut it
    rows = []
    for scenario in DISTANCE_SCENARIOS:
        n = scenario_n_people(scenario)
        d = int(scenario[1:])
        if d not in CHART_DISTANCE_CM:
            continue
        for p in PRECISION_ORDER:
            for track_idx, r in enumerate(per_track_hit_rate(p, scenario)):
                rows.append({"distance_cm": d, "n_people": n, "precision": p,
                             "track_idx": track_idx, "hit_rate": r})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p in PRECISION_ORDER:
        sub = df[df["precision"] == p]
        ax.scatter(sub["distance_cm"], sub["hit_rate"] * 100, color=PRECISION_COLOR[p],
                   alpha=0.55, s=28, linewidths=0, zorder=3)
        mean_by_d = sub.groupby("distance_cm")["hit_rate"].mean() * 100
        ax.plot(mean_by_d.index, mean_by_d.values, color=PRECISION_COLOR[p], marker="o",
                markersize=5, linewidth=2, label=PRECISION_LABELS[p], zorder=2)
    ax.set_xlabel("distance (cm)")
    ax.set_ylabel("achievement rate (%) — individual tracks + mean")
    ax.set_title("1-1 (real video, 1-3 people pooled): per-track achievement rate vs distance (250-400cm)")
    ax.set_xticks(CHART_DISTANCE_CM)
    ax.set_ylim(0, 101)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    savefig(fig, "1-1-1_achievement_rate_vs_distance")


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
