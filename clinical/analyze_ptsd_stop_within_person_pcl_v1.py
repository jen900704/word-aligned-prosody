#!/usr/bin/env python3
"""Gate-licensed earliest-to-latest PTSD-STOP acoustic/PCL change analysis."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gated_clinical_correlation import benjamini_hochberg, load_gate
from ptsd_stop_aggregate_reliability_blind_v1 import (
    FEATURES as AGGREGATE_FEATURES,
    apply_identity_authority,
    load_tokens,
)
from prepare_ptsd_stop_gate_licensed_inputs_v1 import FEATURE_MAP

FEATURE_ORDER = tuple(FEATURE_MAP[name] for name in AGGREGATE_FEATURES)
MIN_SESSION_TOKENS = 50
MIN_REPEATED_PARTICIPANTS = 50
BOOTSTRAP_REPLICATES = 1000
SEED = 13


def structural_sessions(frame: pd.DataFrame, identity_path: Path) -> tuple[pd.DataFrame, dict]:
    identity = pd.read_csv(identity_path, dtype=str)
    required = {
        "recording_id", "authoritative_participant_id",
        "authoritative_user_day_id", "authoritative_date",
        "temporal_match_status",
    }
    if not required.issubset(identity.columns) or identity["recording_id"].duplicated().any():
        raise ValueError("malformed temporal identity authority")
    temporal = identity[list(required)].copy()
    temporal = temporal.loc[temporal["temporal_match_status"].eq("one_to_one")]
    temporal["authoritative_date"] = pd.to_datetime(
        temporal["authoritative_date"], errors="coerce"
    )
    temporal = temporal.dropna(subset=["authoritative_date", "authoritative_user_day_id"])
    cell_counts = temporal.groupby(
        ["authoritative_participant_id", "authoritative_date"], sort=True
    )["authoritative_user_day_id"].nunique()
    valid_cells = cell_counts.loc[cell_counts.eq(1)].reset_index()[
        ["authoritative_participant_id", "authoritative_date"]
    ]
    temporal = temporal.merge(
        valid_cells,
        on=["authoritative_participant_id", "authoritative_date"],
        how="inner",
        validate="many_to_one",
    )
    merged = frame.merge(
        temporal[[
            "recording_id", "authoritative_participant_id",
            "authoritative_user_day_id", "authoritative_date",
        ]],
        on="recording_id", how="inner", validate="many_to_one",
    )
    if not merged["participant_id"].eq(merged["authoritative_participant_id"]).all():
        raise ValueError("temporal and RQ2 participant authority conflict")
    flow = {
        "token_rows_before_temporal_filter": len(frame),
        "token_rows_after_temporal_filter": len(merged),
        "temporally_unique_recordings": temporal["recording_id"].nunique(),
        "unambiguous_user_date_cells": len(valid_cells),
        "users_with_two_or_more_unambiguous_dates": int(
            valid_cells.groupby("authoritative_participant_id")["authoritative_date"]
            .nunique().ge(2).sum()
        ),
    }
    return merged, flow


def session_summary(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    keys = ["participant_id", "authoritative_user_day_id", "authoritative_date"]
    work = frame[keys + ["relative_word_position"]].copy()
    if feature == "syllable_adjusted_log_duration":
        work["y"] = pd.to_numeric(frame["log_word_duration"], errors="coerce")
        work["syllables"] = pd.to_numeric(frame["authoritative_syllable_count"], errors="coerce")
        valid = np.isfinite(work["y"]) & np.isfinite(work["syllables"]) & (work["syllables"] > 0)
        work = work.loc[valid].copy()
        design = np.column_stack([
            np.ones(len(work)), work["relative_word_position"].to_numpy(float),
            work["syllables"].to_numpy(float),
        ])
    else:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
        work = work.loc[np.isfinite(work["y"])].copy()
        design = np.column_stack([
            np.ones(len(work)), work["relative_word_position"].to_numpy(float),
        ])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"rank-deficient session nuisance model for {feature}")
    beta, *_ = np.linalg.lstsq(design, work["y"].to_numpy(float), rcond=None)
    work["residual"] = work["y"].to_numpy(float) - design @ beta
    out = work.groupby(keys, sort=True)["residual"].agg(["median", "size"]).reset_index()
    out = out.loc[out["size"] >= MIN_SESSION_TOKENS].copy()
    return out.rename(columns={"median": "acoustic"})


def load_day_pcl(mysql: str) -> pd.DataFrame:
    query = (
        "SELECT user_id,user_day_id,PCL FROM outcomes_v8_2_stratified "
        "WHERE is_outcome_present=1 AND is_prospective_time=0 AND PCL IS NOT NULL"
    )
    completed = subprocess.run(
        [mysql, "--batch", "--skip-column-names", "ptsd_stop", "-e", query],
        check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    labels = pd.DataFrame(rows, columns=["user_id", "authoritative_user_day_id", "PCL_SCORE"])
    labels["participant_id"] = "P" + labels["user_id"].str.replace(r"[^0-9]", "", regex=True)
    labels["PCL_SCORE"] = pd.to_numeric(labels["PCL_SCORE"], errors="raise")
    if labels["authoritative_user_day_id"].duplicated().any() or not np.isfinite(labels["PCL_SCORE"]).all():
        raise ValueError("daily PCL authority is not one row per user-day with finite score")
    return labels[["participant_id", "authoritative_user_day_id", "PCL_SCORE"]]


def outcome_longitudinal_support(labels: pd.DataFrame) -> dict:
    """Summarize whether the sanctioned PCL field actually varies within users."""
    per_user = labels.groupby("participant_id", sort=True)["PCL_SCORE"].nunique()
    return {
        "users": int(len(per_user)),
        "users_with_two_or_more_scores": int(per_user.ge(2).sum()),
        "minimum_unique_scores_per_user": int(per_user.min()),
        "maximum_unique_scores_per_user": int(per_user.max()),
    }


def changes(sessions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    joined = sessions.merge(
        labels, on=["participant_id", "authoritative_user_day_id"],
        how="inner", validate="one_to_one",
    ).sort_values(["participant_id", "authoritative_date"], kind="mergesort")
    counts = joined.groupby("participant_id")["authoritative_date"].nunique()
    eligible = set(counts.loc[counts.ge(2)].index)
    joined = joined.loc[joined["participant_id"].isin(eligible)]
    first = joined.groupby("participant_id", sort=True).first()
    last = joined.groupby("participant_id", sort=True).last()
    out = pd.DataFrame({
        "participant_id": first.index,
        "delta_acoustic": last["acoustic"] - first["acoustic"],
        "delta_pcl": last["PCL_SCORE"] - first["PCL_SCORE"],
        "followup_days": (last["authoritative_date"] - first["authoritative_date"]).dt.days,
    }).reset_index(drop=True)
    return out


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
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return float(low), float(high), len(values)


def run(gate_path: Path, token_dir: Path, identity_path: Path, output: Path, mysql: str) -> dict:
    passing = load_gate(gate_path, "PTSD-STOP")
    if not passing:
        result = {"protocol": "ptsd_stop_within_person_pcl_v1", "status": "not_run_no_gate_passing_features", "labels_loaded": False, "features": []}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    frame = apply_identity_authority(load_tokens(token_dir), identity_path)
    temporal, structural_flow = structural_sessions(frame, identity_path)
    summaries = {}
    structural_eligible = {}
    for raw_feature in AGGREGATE_FEATURES:
        feature = FEATURE_MAP[raw_feature]
        if feature in passing:
            summary = session_summary(temporal, raw_feature)
            n_repeated = int(summary.groupby("participant_id")["authoritative_date"].nunique().ge(2).sum())
            summaries[feature] = summary
            structural_eligible[feature] = n_repeated
    if not any(value >= MIN_REPEATED_PARTICIPANTS for value in structural_eligible.values()):
        result = {"protocol": "ptsd_stop_within_person_pcl_v1", "status": "structurally_not_estimable", "labels_loaded": False, "structural_flow": structural_flow, "feature_repeated_support": structural_eligible, "features": []}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    labels = load_day_pcl(mysql)
    outcome_support = outcome_longitudinal_support(labels)
    if outcome_support["users_with_two_or_more_scores"] < MIN_REPEATED_PARTICIPANTS:
        result = {
            "protocol": "ptsd_stop_within_person_pcl_v1",
            "status": "outcome_not_longitudinal_no_within_person_variation",
            "labels_loaded": True,
            "outcome_source": "outcomes_v8_2_stratified.PCL[is_outcome_present=1,is_prospective_time=0]",
            "estimand": "spearman_earliest_to_latest_delta_acoustic_vs_delta_pcl",
            "structural_flow": structural_flow,
            "feature_repeated_support_before_pcl": structural_eligible,
            "outcome_longitudinal_support": outcome_support,
            "features": [],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    records = []
    for feature in FEATURE_ORDER:
        if feature not in passing or structural_eligible.get(feature, 0) < MIN_REPEATED_PARTICIPANTS:
            continue
        change = changes(summaries[feature], labels)
        if len(change) < MIN_REPEATED_PARTICIPANTS:
            records.append({"feature": feature, "status": "structurally_not_estimable_after_pcl_join", "n": len(change)})
            continue
        x = change["delta_acoustic"].to_numpy(float)
        y = change["delta_pcl"].to_numpy(float)
        if np.std(x) <= 0 or np.std(y) <= 0:
            records.append({
                "feature": feature,
                "status": "not_estimable_constant_change",
                "n": len(change),
            })
            continue
        test = spearmanr(x, y)
        low, high, successes = bootstrap_ci(x, y)
        if (
            not math.isfinite(float(test.statistic))
            or not math.isfinite(float(test.pvalue))
            or successes < 950
        ):
            records.append({
                "feature": feature,
                "status": "technical_review_no_decision",
                "n": len(change),
                "bootstrap_successes": successes,
            })
            continue
        records.append({
            "feature": feature, "status": "complete", "n": len(change),
            "spearman_rho": float(test.statistic), "p_value": float(test.pvalue),
            "bootstrap_ci_lower": low, "bootstrap_ci_upper": high,
            "bootstrap_successes": successes,
            "followup_days_median": float(change["followup_days"].median()),
            "followup_days_min": int(change["followup_days"].min()),
            "followup_days_max": int(change["followup_days"].max()),
        })
    complete = [row for row in records if row["status"] == "complete"]
    for row, q in zip(complete, benjamini_hochberg([row["p_value"] for row in complete])):
        row["bh_q_value"] = q
    result = {
        "protocol": "ptsd_stop_within_person_pcl_v1",
        "status": "complete" if complete else "structurally_not_estimable_after_pcl_join",
        "labels_loaded": True,
        "outcome_source": "outcomes_v8_2_stratified.PCL[is_outcome_present=1,is_prospective_time=0]",
        "estimand": "spearman_earliest_to_latest_delta_acoustic_vs_delta_pcl",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": SEED,
        "structural_flow": structural_flow,
        "feature_repeated_support_before_pcl": structural_eligible,
        "outcome_longitudinal_support": outcome_support,
        "features": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--identity-authority", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mysql", default="mysql")
    args = parser.parse_args()
    run(args.gate, args.token_dir, args.identity_authority, args.output, args.mysql)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
