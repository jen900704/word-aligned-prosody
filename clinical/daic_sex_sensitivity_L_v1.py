#!/usr/bin/env python3
"""Amendment L: sex-adjusted sensitivity for DAIC-WOZ.

Reuses the frozen preprocessing by importing the frozen reliability script.
Refits nothing frozen and replaces no frozen estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daic_aggregate_reliability_blind_v1 as frozen

CORRELATION_NAME = {
    "f0_mean_hz": "f0_mean_hz",
    "f0_robust_range_hz": "f0_robust_range_hz",
    "energy_db": "energy_db",
    "syllable_adjusted_log_duration": "duration_log_syllable_residual",
}


def residualize_on_sex(values: np.ndarray, sex: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(sex)), sex.astype(float)])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def load_sex(split_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(split_dir.glob("*split_Depression_AVEC2017.csv")):
        frame = pd.read_csv(path)
        frame.columns = [c.strip() for c in frame.columns]
        id_col = next(c for c in frame.columns if c.lower() == "participant_id")
        frames.append(frame[[id_col, "Gender"]].rename(columns={id_col: "participant_id"}))
    sex = pd.concat(frames, ignore_index=True)
    sex["participant_id"] = sex["participant_id"].astype(str)
    sex = sex.drop_duplicates(subset="participant_id")
    if sex["Gender"].isna().any():
        raise ValueError("a corpus-supplied sex label is missing")
    return sex


def half_summaries(frame: pd.DataFrame, feature: str, assignment: pd.Series) -> pd.DataFrame:
    work = frozen.feature_frame(frame, feature, assignment)
    residuals = frozen.residualize_by_half(work, feature)
    counts = residuals.groupby(["participant_id", "half"], sort=True).size()
    summary = residuals.groupby(["participant_id", "half"], sort=True)["residual"].median()
    pivot = summary.unstack()
    keep = counts.unstack()
    eligible = keep.index[
        (keep.get(0, 0) >= frozen.MIN_TOKENS_PER_HALF) & (keep.get(1, 0) >= frozen.MIN_TOKENS_PER_HALF)
    ]
    pivot = pivot.loc[pivot.index.isin(eligible)].dropna()
    return pivot.rename(columns={0: "half_0", 1: "half_1"}).reset_index()


def corrected(x: np.ndarray, y: np.ndarray) -> float:
    raw = float(np.corrcoef(x, y)[0, 1])
    return frozen.spearman_brown(raw)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    n = len(pvalues)
    q = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, pvalues[i] * n / (rank + 1))
        q[i] = running
    return [float(v) for v in q]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--frozen-reliability", required=True, type=Path)
    parser.add_argument("--frozen-correlations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sex = load_sex(args.split_dir)
    labels = pd.read_csv(args.labels)
    labels["participant_id"] = labels["participant_id"].astype(str)
    frozen_rel = {r["feature"]: r for r in json.loads(args.frozen_reliability.read_text())["features"]}
    frozen_cor = {r["feature"]: r for r in json.loads(args.frozen_correlations.read_text())["features"]}

    frame = frozen.load_tokens(args.token_dir)
    assignment = frozen.chronological_assignment(frame)

    joined_ids = set(sex["participant_id"]) & set(labels["participant_id"])
    support = labels.merge(sex, on="participant_id")
    sex_phq = stats.spearmanr(support["Gender"], support["PHQ8_Score"])
    records = []
    raw_p, res_p = [], []

    for feature in frozen.FEATURES:
        pairs = half_summaries(frame, feature, assignment)
        merged = pairs.merge(sex, on="participant_id").merge(labels, on="participant_id")
        g = merged["Gender"].to_numpy(float)
        h0 = merged["half_0"].to_numpy(float)
        h1 = merged["half_1"].to_numpy(float)
        full = (h0 + h1) / 2.0
        phq = merged["PHQ8_Score"].to_numpy(float)

        rel_raw = corrected(h0, h1)
        rel_res = corrected(residualize_on_sex(h0, g), residualize_on_sex(h1, g))
        strata = {}
        for level in sorted(set(g.astype(int))):
            mask = g.astype(int) == level
            strata[str(level)] = {
                "participants": int(mask.sum()),
                "corrected_reliability": corrected(h0[mask], h1[mask]) if mask.sum() >= 20 else None,
            }

        rho_raw = float(stats.spearmanr(full, phq).statistic)
        p_raw = float(stats.spearmanr(full, phq).pvalue)
        rr = stats.spearmanr(residualize_on_sex(full, g), residualize_on_sex(phq, g))
        raw_p.append(p_raw)
        res_p.append(float(rr.pvalue))
        records.append({
            "feature": feature,
            "correlation_name": CORRELATION_NAME[feature],
            "paired_participants": int(len(merged)),
            "frozen_corrected_reliability": frozen_rel[feature]["corrected_reliability"],
            "recomputed_corrected_reliability": rel_raw,
            "sex_residualized_reliability": rel_res,
            "within_sex_reliability": strata,
            "frozen_spearman_rho": frozen_cor[CORRELATION_NAME[feature]]["spearman_rho"],
            "recomputed_spearman_rho": rho_raw,
            "sex_residualized_spearman_rho": float(rr.statistic),
        })

    q_res = benjamini_hochberg(res_p)
    for record, q in zip(records, q_res):
        record["sex_residualized_bh_q"] = q

    payload = {
        "analysis": "daic_sex_sensitivity_L_v1",
        "amendment": "DAIC_sex_sensitivity_amendment_L_20260905.md",
        "corpus": "DAIC-WOZ",
        "sex_label_source": "corpus-supplied Gender in the AVEC 2017 split files",
        "models_refit": 0,
        "frozen_estimates_unchanged": True,
        "support": {
            "participants_with_sex_and_phq8": int(len(support)),
            "sex_composition": {str(k): int(v) for k, v in support["Gender"].value_counts().items()},
            "sex_vs_phq8_spearman": float(sex_phq.statistic),
            "sex_vs_phq8_p": float(sex_phq.pvalue),
        },
        "features": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"features": len(records), "sex_vs_phq8_rho": float(sex_phq.statistic)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
