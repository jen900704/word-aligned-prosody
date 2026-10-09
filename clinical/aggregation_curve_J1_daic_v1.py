#!/usr/bin/env python3
"""Amendment J1 on DAIC-WOZ: Spearman-Brown prediction against observed
split-half reliability as a function of tokens per half.

Reuses the frozen DAIC preprocessing by importing the frozen reliability
script. Adds no outcome variable; this file never touches a label.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daic_aggregate_reliability_blind_v1 as frozen

K_GRID = (5, 10, 20, 50, 100, 200)
DRAWS = 200
SEED = 13


def one_way_icc(groups: list[np.ndarray]) -> dict:
    """ICC(1) for unbalanced one-way random effects on token residuals."""
    sizes = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups], dtype=float)
    total = sizes.sum()
    grand = float((sizes * means).sum() / total)
    n_groups = len(groups)
    ms_between = float((sizes * (means - grand) ** 2).sum() / (n_groups - 1))
    within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    ms_within = within / (total - n_groups)
    n_zero = float((total - (sizes ** 2).sum() / total) / (n_groups - 1))
    denominator = ms_between + (n_zero - 1.0) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else float("nan")
    return {
        "icc1": float(icc),
        "ms_between": ms_between,
        "ms_within": ms_within,
        "n_zero": n_zero,
        "groups": n_groups,
        "tokens": int(total),
    }


def spearman_brown_prediction(icc: float, k: int) -> float:
    return k * icc / (1.0 + (k - 1) * icc)


def observed_curve(halves: dict[str, dict[int, np.ndarray]], k: int, rng) -> dict:
    eligible = [
        participant
        for participant, sides in halves.items()
        if len(sides.get(0, ())) >= k and len(sides.get(1, ())) >= k
    ]
    if len(eligible) < frozen.MIN_PAIRED_PARTICIPANTS:
        return {"tokens_per_half": k, "eligible_participants": len(eligible), "status": "support_below_50"}
    means, medians = [], []
    raw_means, raw_medians = [], []
    for _ in range(DRAWS):
        left_mean, right_mean, left_med, right_med = [], [], [], []
        for participant in eligible:
            sides = halves[participant]
            a = rng.choice(sides[0], size=k, replace=False)
            b = rng.choice(sides[1], size=k, replace=False)
            left_mean.append(a.mean())
            right_mean.append(b.mean())
            left_med.append(np.median(a))
            right_med.append(np.median(b))
        for store, raw_store, left, right in (
            (means, raw_means, left_mean, right_mean),
            (medians, raw_medians, left_med, right_med),
        ):
            left_array = np.asarray(left, dtype=float)
            right_array = np.asarray(right, dtype=float)
            if left_array.std() <= 0 or right_array.std() <= 0:
                continue
            raw = float(np.corrcoef(left_array, right_array)[0, 1])
            raw_store.append(raw)
            store.append(frozen.spearman_brown(raw))
    def summarise(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        return {
            "draws": int(array.size),
            "median": float(np.median(array)),
            "p05": float(np.percentile(array, 5)),
            "p95": float(np.percentile(array, 95)),
        }
    return {
        "tokens_per_half": k,
        "eligible_participants": len(eligible),
        "reliability_of_a_k_token_mean": summarise(raw_means),
        "reliability_of_a_k_token_median": summarise(raw_medians),
        "spearman_brown_to_2k_mean": summarise(means),
        "spearman_brown_to_2k_median": summarise(medians),
        "status": "complete",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = frozen.load_tokens(args.token_dir)
    assignment = frozen.chronological_assignment(frame)
    rng = np.random.Generator(np.random.PCG64(SEED))

    features = []
    for feature in frozen.FEATURES:
        work = frozen.feature_frame(frame, feature, assignment)
        residuals = frozen.residualize_by_half(work, feature)
        halves: dict[str, dict[int, np.ndarray]] = {}
        for (participant, half), group in residuals.groupby(["participant_id", "half"], sort=True):
            halves.setdefault(str(participant), {})[int(half)] = group["residual"].to_numpy(float)
        pooled = [
            np.concatenate([sides[0], sides[1]])
            for sides in halves.values()
            if 0 in sides and 1 in sides
        ]
        icc = one_way_icc(pooled)
        curve = []
        for k in K_GRID:
            record = observed_curve(halves, k, rng)
            if record["status"] == "complete":
                record["prediction_at_k"] = spearman_brown_prediction(icc["icc1"], k)
                record["prediction_at_2k"] = spearman_brown_prediction(icc["icc1"], 2 * k)
                record["gap_at_k_prediction_minus_observed"] = (
                    record["prediction_at_k"]
                    - record["reliability_of_a_k_token_mean"]["median"]
                )
                record["gap_at_2k_prediction_minus_observed"] = (
                    record["prediction_at_2k"]
                    - record["spearman_brown_to_2k_mean"]["median"]
                )
            curve.append(record)
        tokens_per_half = [len(s[0]) for s in halves.values() if 0 in s]
        features.append(
            {
                "feature": feature,
                "token_level_icc": icc,
                "median_tokens_per_participant_half": float(np.median(tokens_per_half)),
                "curve": curve,
            }
        )

    payload = {
        "analysis": "aggregation_curve_J1_daic_v1",
        "amendment": "MEASUREMENT_MODEL_amendment_J_20260904.md",
        "corpus": "DAIC-WOZ",
        "labels_loaded": False,
        "outcome_analysis_performed": False,
        "split_unit": "chronologically ordered odd/even participant utterances",
        "draws_per_k": DRAWS,
        "seed": SEED,
        "generator": "PCG64",
        "prediction": "R(k) = k rho / (1 + (k - 1) rho), rho = token-level ICC(1)",
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"features": len(features)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
