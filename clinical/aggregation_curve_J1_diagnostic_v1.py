#!/usr/bin/env python3
"""Diagnostic for J1: why the observed DAIC curve sits above the ICC(1)
Spearman-Brown prediction.

Hypothesis under test: within-participant token variance is heavily right
skewed across participants, so the ANOVA mean-square-within (an arithmetic
mean of per-participant variances) overstates the noise that actually limits
the across-participant correlation of half means.

Reports the skew and a median-based prediction beside the frozen-style one.
No label is touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daic_aggregate_reliability_blind_v1 as frozen
from aggregation_curve_J1_daic_v1 import K_GRID, one_way_icc, spearman_brown_prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = frozen.load_tokens(args.token_dir)
    assignment = frozen.chronological_assignment(frame)

    features = []
    for feature in frozen.FEATURES:
        work = frozen.feature_frame(frame, feature, assignment)
        residuals = frozen.residualize_by_half(work, feature)
        pooled, within_variances, participant_means = [], [], []
        for _, group in residuals.groupby("participant_id", sort=True):
            values = group["residual"].to_numpy(float)
            pooled.append(values)
            within_variances.append(float(values.var(ddof=1)))
            participant_means.append(float(values.mean()))
        icc = one_way_icc(pooled)
        within = np.asarray(within_variances, dtype=float)
        between = float(np.var(np.asarray(participant_means, dtype=float), ddof=1))
        arithmetic = float(within.mean())
        median_within = float(np.median(within))
        features.append(
            {
                "feature": feature,
                "icc1": icc["icc1"],
                "within_variance_arithmetic_mean": arithmetic,
                "within_variance_median": median_within,
                "within_variance_p90": float(np.percentile(within, 90)),
                "within_variance_max": float(within.max()),
                "skew_ratio_mean_over_median": arithmetic / median_within,
                "between_participant_variance_of_means": between,
                "prediction_from_ms_within": {
                    str(k): spearman_brown_prediction(icc["icc1"], k) for k in K_GRID
                },
                "prediction_from_median_within": {
                    str(k): between / (between + median_within / k) for k in K_GRID
                },
            }
        )

    payload = {
        "analysis": "aggregation_curve_J1_diagnostic_v1",
        "amendment": "MEASUREMENT_MODEL_amendment_J_20260904.md",
        "corpus": "DAIC-WOZ",
        "labels_loaded": False,
        "note": "diagnostic only; it revises no estimate and no gate",
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"features": len(features)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
