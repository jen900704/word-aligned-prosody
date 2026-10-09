#!/usr/bin/env python3
"""Run the frozen, same-corpus gate-licensed clinical correlation analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURE_ORDER = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "duration_log_syllable_residual",
)
PHQ8_RELIABILITY = 0.870
BOOTSTRAP_REPLICATES = 1000
SEED = 13


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.tolist()


def load_gate(path: Path, corpus: str) -> dict[str, float]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("corpus") != corpus:
        raise ValueError("gate corpus does not match requested corpus")
    records = gate.get("features")
    if not isinstance(records, list):
        raise ValueError("gate features list is absent")
    seen = set()
    passing = {}
    for record in records:
        feature = record.get("feature")
        if feature not in FEATURE_ORDER or feature in seen:
            raise ValueError("unknown or duplicate gate feature")
        seen.add(feature)
        status = record.get("status")
        if status not in {"pass", "fail", "technical_review_no_decision"}:
            raise ValueError("malformed gate status")
        if status == "pass":
            reliability = float(record.get("corrected_reliability"))
            lower = float(record.get("ci_lower"))
            robustness_delta = float(record.get("max_robustness_delta", 0.0))
            if reliability < 0.60 or lower < 0.40 or robustness_delta > 0.15:
                raise ValueError("pass status conflicts with frozen aggregate thresholds")
            passing[feature] = reliability
    if seen != set(FEATURE_ORDER):
        raise ValueError("gate does not contain the exact four-feature family")
    return passing


def _bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    rng = np.random.Generator(np.random.PCG64(SEED))
    values = []
    n = len(x)
    for _ in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, n, size=n)
        rho = spearmanr(x[index], y[index]).statistic
        if math.isfinite(float(rho)):
            values.append(float(rho))
    if len(values) < 950:
        return math.nan, math.nan, len(values)
    lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
    return float(lower), float(upper), len(values)


def run_analysis(
    acoustics_path: Path,
    labels_path: Path,
    gate_path: Path,
    output_path: Path,
    corpus: str,
    id_column: str,
    outcome_column: str,
    outcome_reliability: float,
) -> dict:
    # The gate is deliberately read before the label file. With no passing
    # feature the function returns without touching labels_path.
    passing = load_gate(gate_path, corpus)
    if not passing:
        result = {
            "protocol": "clinical_regime_amendment_v2_plus_v3_1_addendum",
            "corpus": corpus,
            "status": "not_run_no_gate_passing_features",
            "labels_loaded": False,
            "features": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    acoustics = pd.read_csv(acoustics_path)
    labels = pd.read_csv(labels_path)
    if id_column not in acoustics or id_column not in labels or outcome_column not in labels:
        raise ValueError("required participant/outcome column is absent")
    if acoustics[id_column].duplicated().any() or labels[id_column].duplicated().any():
        raise ValueError("participant rows must be unique before joining")
    merged = acoustics.merge(
        labels[[id_column, outcome_column]], on=id_column, how="inner", validate="one_to_one"
    )
    records = []
    for feature in FEATURE_ORDER:
        if feature not in passing:
            continue
        if feature not in merged:
            raise ValueError(f"gate-passing feature absent from acoustic summaries: {feature}")
        subset = merged[[feature, outcome_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < 30:
            raise ValueError(f"fewer than 30 paired participants for {feature}")
        x = subset[feature].to_numpy(dtype=float)
        y = subset[outcome_column].to_numpy(dtype=float)
        test = spearmanr(x, y)
        rho = float(test.statistic)
        p_value = float(test.pvalue)
        lower, upper, successes = _bootstrap_ci(x, y)
        attenuation_denominator = math.sqrt(passing[feature] * outcome_reliability)
        records.append(
            {
                "feature": feature,
                "n": len(subset),
                "spearman_rho": rho,
                "p_value": p_value,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "bootstrap_successes": successes,
                "acoustic_reliability": passing[feature],
                "outcome_reliability": outcome_reliability,
                "disattenuated_rho_unclipped": rho / attenuation_denominator,
            }
        )
    adjusted = benjamini_hochberg([record["p_value"] for record in records])
    for record, q_value in zip(records, adjusted):
        record["bh_q_value"] = q_value
    result = {
        "protocol": "clinical_regime_amendment_v2_plus_v3_1_addendum",
        "corpus": corpus,
        "status": "complete",
        "labels_loaded": True,
        "id_column": id_column,
        "outcome_column": outcome_column,
        "feature_order": list(FEATURE_ORDER),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "features": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustics", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--id-column", default="participant_id")
    parser.add_argument("--outcome-column", required=True)
    parser.add_argument("--outcome-reliability", type=float, default=PHQ8_RELIABILITY)
    args = parser.parse_args()
    run_analysis(
        args.acoustics,
        args.labels,
        args.gate,
        args.output,
        args.corpus,
        args.id_column,
        args.outcome_column,
        args.outcome_reliability,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

