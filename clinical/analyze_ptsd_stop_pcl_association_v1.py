#!/usr/bin/env python3
"""Frozen gate-licensed between-person PTSD-STOP PCL association."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gated_clinical_correlation import benjamini_hochberg, load_gate

FEATURE_ORDER = (
    "f0_mean_hz", "f0_robust_range_hz", "energy_db",
    "duration_log_syllable_residual",
)
EXPECTED_PAIRED_PARTICIPANTS = 213
OUTCOME = "pcl_mean_days_0_60"
BOOTSTRAP_REPLICATES = 1000
SEED = 13


def bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    rng = np.random.Generator(np.random.PCG64(SEED))
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(x), size=len(x))
        rho = spearmanr(x[index], y[index]).statistic
        if math.isfinite(float(rho)):
            values.append(float(rho))
    if len(values) < 950:
        return math.nan, math.nan, len(values)
    lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
    return float(lower), float(upper), len(values)


def run(acoustics_path: Path, labels_path: Path, gate_path: Path, output: Path) -> dict:
    passing = load_gate(gate_path, "PTSD-STOP")
    if not passing:
        result = {
            "protocol": "ptsd_stop_between_person_pcl_v1",
            "status": "not_run_no_gate_passing_features",
            "labels_loaded": False,
            "features": [],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    acoustics = pd.read_csv(acoustics_path, dtype={"participant_id": str})
    labels = pd.read_csv(labels_path, dtype={"participant_id": str})
    if acoustics["participant_id"].duplicated().any() or labels["participant_id"].duplicated().any():
        raise ValueError("participant rows must be unique")
    merged = acoustics.merge(labels[["participant_id", OUTCOME]], on="participant_id", how="inner", validate="one_to_one")
    if len(merged) != EXPECTED_PAIRED_PARTICIPANTS:
        raise ValueError("between-person paired participant count mismatch")
    records = []
    for feature in FEATURE_ORDER:
        if feature not in passing:
            continue
        subset = merged[[feature, OUTCOME]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) != EXPECTED_PAIRED_PARTICIPANTS:
            raise ValueError(f"missing paired values for {feature}")
        x = subset[feature].to_numpy(float)
        y = subset[OUTCOME].to_numpy(float)
        test = spearmanr(x, y)
        lower, upper, successes = bootstrap_ci(x, y)
        records.append({
            "feature": feature,
            "n": len(subset),
            "spearman_rho": float(test.statistic),
            "p_value": float(test.pvalue),
            "bootstrap_ci_lower": lower,
            "bootstrap_ci_upper": upper,
            "bootstrap_successes": successes,
            "acoustic_reliability": passing[feature],
        })
    for record, q_value in zip(records, benjamini_hochberg([row["p_value"] for row in records])):
        record["bh_q_value"] = q_value
    result = {
        "protocol": "ptsd_stop_between_person_pcl_v1",
        "corpus": "PTSD-STOP",
        "status": "complete",
        "labels_loaded": True,
        "outcome": OUTCOME,
        "estimand": "between_person_spearman_participant_median_location_vs_official_days_0_60_pcl",
        "feature_order": list(FEATURE_ORDER),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "disattenuation_reported": False,
        "features": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustics", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.acoustics, args.labels, args.gate, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
