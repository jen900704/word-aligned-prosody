#!/usr/bin/env python3
"""Amendment J4 and J5: attenuation ceiling, minimum detectable effect, and a
family-wise permutation null for the clinical associations.

Refits nothing and replaces nothing.  The frozen correlations and the frozen
Benjamini-Hochberg result remain primary; these are bounds reported beside them.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_ORDER = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "duration_log_syllable_residual",
)
PERMUTATIONS = 10000
SEED = 13
POWER_Z = 0.8416212335729143   # one-sided z for 80% power
ALPHA_Z = 1.959963984540054    # two-sided z for alpha = 0.05


def mde(n: int) -> float:
    if n <= 3:
        raise ValueError("n must exceed three for a Fisher-z minimum detectable effect")
    return float(np.tanh((ALPHA_Z + POWER_Z) / math.sqrt(n - 3)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correlations", required=True, type=Path)
    parser.add_argument("--summaries", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frozen = json.loads(args.correlations.read_text(encoding="utf-8"))
    if not frozen.get("labels_loaded"):
        raise ValueError("frozen correlation file is not an unblinded outcome analysis")
    by_feature = {record["feature"]: record for record in frozen["features"]}
    if tuple(frozen["feature_order"]) != FEATURE_ORDER:
        raise ValueError("frozen feature order mismatch")

    summaries = pd.read_csv(args.summaries)
    labels = pd.read_csv(args.labels)
    outcome_column = frozen["outcome_column"]
    id_column = frozen["id_column"]
    merged = summaries.merge(labels, on=id_column, how="inner")
    merged = merged.dropna(subset=[outcome_column, *FEATURE_ORDER])
    n = len(merged)
    frozen_n = {record["n"] for record in frozen["features"]}
    if frozen_n != {n}:
        raise ValueError(f"participant count {n} does not match the frozen n {frozen_n}")

    observed_rho = {}
    ranked_features = {}
    for feature in FEATURE_ORDER:
        rho = float(stats.spearmanr(merged[feature], merged[outcome_column]).statistic)
        if not math.isclose(rho, by_feature[feature]["spearman_rho"], abs_tol=1e-9):
            raise ValueError(f"recomputed rho does not reproduce the frozen value for {feature}")
        observed_rho[feature] = rho
        ranked_features[feature] = stats.rankdata(merged[feature].to_numpy(float))
    ranked_outcome = stats.rankdata(merged[outcome_column].to_numpy(float))

    design = np.column_stack([ranked_features[f] for f in FEATURE_ORDER])
    design = (design - design.mean(axis=0)) / design.std(axis=0)
    centred_outcome = (ranked_outcome - ranked_outcome.mean()) / ranked_outcome.std()
    observed_max = float(np.max(np.abs(design.T @ centred_outcome / n)))

    generator = np.random.Generator(np.random.PCG64(SEED))
    null_max = np.empty(PERMUTATIONS, dtype=float)
    for index in range(PERMUTATIONS):
        permuted = generator.permutation(centred_outcome)
        null_max[index] = np.max(np.abs(design.T @ permuted / n))
    permutation_p = float((1.0 + np.sum(null_max >= observed_max - 1e-12)) / (PERMUTATIONS + 1.0))

    records = []
    for feature in FEATURE_ORDER:
        record = by_feature[feature]
        r_xx = float(record["acoustic_reliability"])
        r_yy = float(record["outcome_reliability"])
        ceiling = math.sqrt(max(r_xx, 0.0) * max(r_yy, 0.0))
        records.append(
            {
                "feature": feature,
                "n": int(record["n"]),
                "observed_spearman_rho": float(record["spearman_rho"]),
                "acoustic_reliability": r_xx,
                "outcome_reliability_cited": r_yy,
                "attenuation_ceiling": ceiling,
                "disattenuated_rho": float(record["disattenuated_rho_unclipped"]),
                "share_of_ceiling_attained": abs(float(record["spearman_rho"])) / ceiling,
                "frozen_bh_q_value": float(record["bh_q_value"]),
            }
        )

    payload = {
        "analysis": "clinical_bounds_J4_J5_v1",
        "amendment": "MEASUREMENT_MODEL_amendment_J_20260904.md",
        "corpus": frozen.get("corpus"),
        "outcome_column": outcome_column,
        "models_refit": 0,
        "frozen_correlations_unchanged": True,
        "attenuation_ceiling": {
            "definition": "sqrt(r_xx * r_yy), the largest correlation the two "
            "instruments could show if the underlying constructs were identical",
            "records": records,
            "largest_share_of_ceiling_attained": max(
                r["share_of_ceiling_attained"] for r in records
            ),
        },
        "minimum_detectable_effect": {
            "definition": "tanh((z_0.975 + z_0.80) / sqrt(n - 3)) at 80% power, alpha 0.05 two-sided",
            "n": n,
            "value": mde(n),
        },
        "family_permutation_null": {
            "statistic": "max |Spearman rho| over the four frozen features",
            "replicates": PERMUTATIONS,
            "seed": SEED,
            "generator": "PCG64",
            "observed": observed_max,
            "null_median": float(np.median(null_max)),
            "null_p95": float(np.percentile(null_max, 95)),
            "p_value": permutation_p,
            "note": "reported beside the frozen Benjamini-Hochberg result, which stays primary",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"n": n, "mde": mde(n), "permutation_p": permutation_p}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
