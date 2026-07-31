"""
arrangement_generator.py
==========================
Pure Python, no bpy needed. Reads pose_library.json (exported from the
Blender pose-library tool) and produces arrangement.json: for every
(distance, people_count, sample_index) combination, which pose-library
entries go at which X position.

Responsibility (settled 2026-07): this script only computes WHAT goes
WHERE -- it never touches Blender, never places a camera, never renders
anything. A separate, later piece reads this file's output and calls
call_pose(entry, location) (from the pose library tool) once per person,
per sample, using distance_cm to set up the camera for that sample.

DESIGN
-------
  distances:      250cm to 550cm, 25cm steps (13 points). 25cm chosen so
                   a smoother FP32 floor-finding curve is possible later;
                   the 50cm-only subset is just every other point in this
                   same file if that ends up being all that's needed --
                   filter on distance_cm % 50 == 0.
  people_counts:  1..5
  samples/cell:   100 (see pose library: 5 characters x 20 poses = 100
                   unique combos for the 1-person case, more than enough
                   combinatorial room for 2-5 people too)
  spacing:        0.70m between adjacent people, NOT scaled per distance
                   -- validated safe at 250cm (the tightest case, least
                   horizontal FOV available); farther distances only have
                   MORE horizontal room, so the same spacing stays safe.
  character reuse: within one sample, characters are chosen WITHOUT
                   replacement (a person can't be in two places in the
                   same frame); which characters appear, their left-to-
                   right order, and their individual pose are all drawn
                   independently per sample.
"""

import argparse
import json
import random

DISTANCES_CM = list(range(250, 551, 25))  # 250,275,...,550 -- 13 points
PEOPLE_COUNTS = [1, 2, 3, 4, 5]
SAMPLES_PER_CELL = 100
SPACING_M = 0.70
RANDOM_SEED = 7  # fixed on purpose -- reproducible, same output every run


def load_pose_library(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data["entries"]
    by_character = {}
    for e in entries:
        by_character.setdefault(e["armature_name"], []).append(e)
    return entries, by_character


def x_positions(n, spacing=SPACING_M):
    """Symmetric line centered on X=0. n=3 -> [-spacing, 0, +spacing]."""
    return [round((i - (n - 1) / 2.0) * spacing, 4) for i in range(n)]


def generate_arrangements(by_character, seed=RANDOM_SEED):
    rng = random.Random(seed)
    characters = sorted(by_character.keys())
    if len(characters) < max(PEOPLE_COUNTS):
        raise ValueError(
            f"pose_library.json only has {len(characters)} characters, "
            f"need at least {max(PEOPLE_COUNTS)} for the largest "
            f"people-count condition ({max(PEOPLE_COUNTS)} people)."
        )

    samples = []
    for distance_cm in DISTANCES_CM:
        for n in PEOPLE_COUNTS:
            xs = x_positions(n)
            for sample_idx in range(SAMPLES_PER_CELL):
                chosen_chars = rng.sample(characters, n)  # no repeats within a sample
                people = []
                for slot, char_name in enumerate(chosen_chars):
                    pose_entry = rng.choice(by_character[char_name])
                    people.append({
                        "slot": slot,
                        "x_m": xs[slot],
                        "character": char_name,
                        "pose_entry_id": pose_entry["id"],
                    })
                samples.append({
                    "sample_id": f"d{distance_cm}_n{n}_s{sample_idx:03d}",
                    "distance_cm": distance_cm,
                    "people_count": n,
                    "sample_index": sample_idx,
                    "people": people,
                })
    return samples


def summarize(samples):
    n_distances = len(DISTANCES_CM)
    n_counts = len(PEOPLE_COUNTS)
    expected = n_distances * n_counts * SAMPLES_PER_CELL
    ids = set(s["sample_id"] for s in samples)
    print(f"design: {n_distances} distances x {n_counts} people-counts x "
          f"{SAMPLES_PER_CELL} samples/cell = {expected} expected")
    print(f"generated: {len(samples)} samples, {len(ids)} unique sample_id")
    # sanity: no repeated character within any single sample
    bad = [s["sample_id"] for s in samples
           if len(set(p["character"] for p in s["people"])) != len(s["people"])]
    print(f"samples with a repeated character in the same frame: {len(bad)} (should be 0)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pose-library", default="pose_library.json")
    p.add_argument("--out", default="arrangement.json")
    args = p.parse_args()

    _, by_character = load_pose_library(args.pose_library)
    samples = generate_arrangements(by_character)
    summarize(samples)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")
