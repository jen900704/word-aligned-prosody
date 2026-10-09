#!/usr/bin/env python3
"""Amendment J1 on MSP-Podcast: Spearman-Brown prediction against observed
split-half reliability as a function of tokens per half.

Preprocessing is the frozen one: the merged acoustics are loaded through the
frozen hash-chain loader and residualized by the frozen primary model, so the
tokens here are the tokens behind the Gate M numbers. Nothing is refitted and
no outcome is read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.msp_final_analysis_v1 import load_merged, prepare_feature
from tools.msp_reliability_v2 import MIN_HALF_TOKENS, MIN_PAIRED_SPEAKERS, corrected_pearson

FEATURE_ORDER = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "duration_log_syllable_residual",
)
K_GRID = (5, 10, 20, 50, 100, 200)
DRAWS = 200
SEED = 13
HALVES = ("A", "B")


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


def prediction(icc: float, k: int) -> float:
    return k * icc / (1.0 + (k - 1) * icc)


def tokens_for(icc: float, target: float) -> float:
    return math.inf if icc <= 0 else target / ((1.0 - target) * icc)


def summarise(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"draws": 0}
    return {
        "draws": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def observed_curve(halves: dict, base_speakers: list, k: int, rng) -> dict:
    eligible = [
        speaker
        for speaker in base_speakers
        if len(halves[speaker]["A"]) >= k and len(halves[speaker]["B"]) >= k
    ]
    if len(eligible) < MIN_PAIRED_SPEAKERS:
        return {"tokens_per_half": k, "eligible_speakers": len(eligible), "status": "support_below_50"}
    raw_mean, raw_median, sb_mean, sb_median = [], [], [], []
    for _ in range(DRAWS):
        columns = {("mean", h): [] for h in HALVES}
        columns.update({("median", h): [] for h in HALVES})
        for speaker in eligible:
            for h in HALVES:
                sample = rng.choice(halves[speaker][h], size=k, replace=False)
                columns[("mean", h)].append(float(sample.mean()))
                columns[("median", h)].append(float(np.median(sample)))
        for kind, raw_store, sb_store in (
            ("mean", raw_mean, sb_mean),
            ("median", raw_median, sb_median),
        ):
            raw, corrected = corrected_pearson(columns[(kind, "A")], columns[(kind, "B")])
            if raw is None or corrected is None:
                continue
            raw_store.append(float(raw))
            sb_store.append(float(corrected))
    return {
        "tokens_per_half": k,
        "eligible_speakers": len(eligible),
        "reliability_of_a_k_token_mean": summarise(raw_mean),
        "reliability_of_a_k_token_median": summarise(raw_median),
        "spearman_brown_to_2k_mean": summarise(sb_mean),
        "spearman_brown_to_2k_median": summarise(sb_median),
        "status": "complete",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-acoustics", required=True, type=Path)
    parser.add_argument("--merge-safe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rows, merge_receipt = load_merged(args.merged_acoustics, args.merge_safe)
    rng = np.random.Generator(np.random.PCG64(SEED))

    features = []
    for feature in FEATURE_ORDER:
        residual_rows, fit = prepare_feature(rows, feature)
        halves: dict[str, dict[str, list]] = {}
        for record in residual_rows:
            speaker = str(record["speaker_id"])
            half = str(record["primary_half"])
            if half not in HALVES:
                raise ValueError("unexpected half label")
            halves.setdefault(speaker, {"A": [], "B": []})[half].append(float(record["residual"]))
        del residual_rows, fit
        halves = {s: {h: np.asarray(v, dtype=float) for h, v in d.items()} for s, d in halves.items()}
        base_speakers = [
            s
            for s, d in halves.items()
            if len(d["A"]) >= MIN_HALF_TOKENS and len(d["B"]) >= MIN_HALF_TOKENS
        ]
        base_speakers.sort()
        pooled = [np.concatenate([halves[s]["A"], halves[s]["B"]]) for s in base_speakers]
        icc = one_way_icc(pooled)
        curve = []
        for k in K_GRID:
            record = observed_curve(halves, base_speakers, k, rng)
            if record["status"] == "complete":
                record["prediction_at_k"] = prediction(icc["icc1"], k)
                record["prediction_at_2k"] = prediction(icc["icc1"], 2 * k)
                record["gap_at_k_prediction_minus_observed"] = (
                    record["prediction_at_k"] - record["reliability_of_a_k_token_mean"]["median"]
                )
            curve.append(record)

        features.append(
            {
                "feature": feature,
                "token_level_icc": icc,
                "base_speakers": len(base_speakers),
                "median_tokens_per_speaker_half": float(
                    np.median([len(halves[s]["A"]) for s in base_speakers])
                ),
                "tokens_for_0.80": tokens_for(icc["icc1"], 0.80),
                "tokens_for_0.90": tokens_for(icc["icc1"], 0.90),
                "curve": curve,
            }
        )
        del halves, pooled

    payload = {
        "analysis": "msp_aggregation_curve_J1_v1",
        "amendment": "MEASUREMENT_MODEL_amendment_J_20260904.md",
        "corpus": "MSP-Podcast",
        "models_refit": 0,
        "clinical_outcomes_accessed": False,
        "merged_token_rows": merge_receipt.get("merged_token_rows"),
        "merged_sha256": merge_receipt.get("output_sha256"),
        "draws_per_k": DRAWS,
        "seed": SEED,
        "generator": "PCG64",
        "prediction": "R(k) = k rho / (1 + (k - 1) rho), rho = token-level ICC(1)",
        "eligibility": "frozen speaker support rule: at least 50 tokens in each half",
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"features": len(features)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
