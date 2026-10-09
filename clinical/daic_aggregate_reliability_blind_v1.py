#!/usr/bin/env python3
"""Outcome-blind DAIC RQ2 aggregate reliability under the frozen amendment.

Scientific estimates are written only to a PRIVATE JSON. A separate technical
JSON contains completion/support status but no reliability values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "syllable_adjusted_log_duration",
)
SOURCE_COLUMNS = (
    "participant_id",
    "utterance_index",
    "word_index",
    "normalized_word",
    "alignment_status",
    "log_word_duration",
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "authoritative_syllable_count",
)
MIN_TOKENS_PER_HALF = 50
MIN_PAIRED_PARTICIPANTS = 50
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 13
BOOTSTRAP_MIN_SUCCESS = 950
SALTED_SPLITS = 25
GATE_RELIABILITY = 0.60
GATE_CI_LOWER = 0.40
GATE_MAX_DELTA = 0.15
EXPECTED_CLOSURE_STATUS = "TECHNICAL_EXTRACTION_COMPLETE"
EXPECTED_PARTICIPANTS = 189
FORBIDDEN = ("phq", "pcl", "label", "outcome", "diagnos", "demograph")


def reject(path: Path) -> None:
    if any(part in str(path).lower() for part in FORBIDDEN):
        raise ValueError(f"forbidden clinical path: {path}")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], proportion: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.quantile(finite, proportion)) if len(finite) else None


def spearman_brown(r_value: float) -> float:
    r_value = float(r_value)
    denominator = 1.0 + r_value
    if not np.isfinite(r_value) or abs(denominator) < 1e-12:
        raise ValueError("invalid half-correlation for Spearman-Brown correction")
    return 2.0 * r_value / denominator


def pearson_pair(frame: pd.DataFrame) -> tuple[float, float]:
    if len(frame) < MIN_PAIRED_PARTICIPANTS:
        raise ValueError("paired participant support below 50")
    left = frame["half_0"].to_numpy(float)
    right = frame["half_1"].to_numpy(float)
    if np.std(left) <= 0 or np.std(right) <= 0:
        raise ValueError("zero variance in participant summaries")
    raw = float(np.corrcoef(left, right)[0, 1])
    return raw, spearman_brown(raw)


def verify_closure(path: Path) -> dict:
    reject(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "status": EXPECTED_CLOSURE_STATUS,
        "participants": EXPECTED_PARTICIPANTS,
        "labels_loaded": False,
        "scientific_results_inspected": False,
        "reliability_analysis_performed": False,
        "outcome_analysis_performed": False,
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise ValueError(f"extraction closure mismatch for {key}")
    return record


def load_tokens(token_dir: Path) -> pd.DataFrame:
    reject(token_dir)
    paths = sorted(token_dir.glob("*.csv"))
    if len(paths) != EXPECTED_PARTICIPANTS:
        raise ValueError(f"expected 189 token files, found {len(paths)}")
    frames = []
    for path in paths:
        reject(path)
        frames.append(pd.read_csv(path, usecols=SOURCE_COLUMNS, low_memory=False))
    frame = pd.concat(frames, ignore_index=True)
    frame["participant_id"] = frame["participant_id"].astype(str)
    if frame["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise ValueError("token files do not contain 189 unique participants")
    frame = frame.loc[frame["alignment_status"].eq("aligned")].copy()
    frame["utterance_index"] = pd.to_numeric(frame["utterance_index"], errors="raise").astype(int)
    frame["word_index"] = pd.to_numeric(frame["word_index"], errors="raise").astype(int)
    maximum = frame.groupby(["participant_id", "utterance_index"], sort=False)["word_index"].transform("max")
    frame["relative_word_position"] = np.where(maximum > 0, frame["word_index"] / maximum, 0.0)
    return frame


def chronological_assignment(frame: pd.DataFrame) -> pd.Series:
    return frame["utterance_index"].astype(int) % 2


def salted_assignment(frame: pd.DataFrame, salt_index: int) -> pd.Series:
    salt = f"daic-balanced-split-{salt_index:02d}"
    unique = frame[["participant_id", "utterance_index"]].drop_duplicates()
    mapping: dict[tuple[str, int], int] = {}
    for participant, group in unique.groupby("participant_id", sort=True):
        ordered = sorted(
            (int(value) for value in group["utterance_index"]),
            key=lambda value: hashlib.sha256(
                f"{salt}|{participant}|{value}".encode("utf-8")
            ).digest(),
        )
        for rank, utterance in enumerate(ordered):
            mapping[(str(participant), utterance)] = rank % 2
    return pd.Series(
        [mapping[(participant, utterance)] for participant, utterance in zip(frame["participant_id"], frame["utterance_index"])],
        index=frame.index,
        dtype=int,
    )


def feature_frame(frame: pd.DataFrame, feature: str, assignment: pd.Series) -> pd.DataFrame:
    if feature not in FEATURES:
        raise ValueError(f"unexpected feature: {feature}")
    work = frame[["participant_id", "utterance_index", "relative_word_position"]].copy()
    work["half"] = assignment.loc[work.index].astype(int)
    if feature == "syllable_adjusted_log_duration":
        work["y"] = pd.to_numeric(frame["log_word_duration"], errors="coerce")
        work["syllables"] = pd.to_numeric(frame["authoritative_syllable_count"], errors="coerce")
        work = work.loc[np.isfinite(work["y"]) & np.isfinite(work["syllables"]) & (work["syllables"] > 0)].copy()
    else:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
        work = work.loc[np.isfinite(work["y"])].copy()
    if not set(work["half"].unique()).issubset({0, 1}):
        raise ValueError("split assignment is not binary")
    return work


def residualize_by_half(work: pd.DataFrame, feature: str) -> pd.DataFrame:
    pieces = []
    for half in (0, 1):
        part = work.loc[work["half"].eq(half)].copy()
        columns = [np.ones(len(part)), part["relative_word_position"].to_numpy(float)]
        if feature == "syllable_adjusted_log_duration":
            columns.append(part["syllables"].to_numpy(float))
        design = np.column_stack(columns)
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("nuisance residualization is rank deficient")
        beta, *_ = np.linalg.lstsq(design, part["y"].to_numpy(float), rcond=None)
        part["residual"] = part["y"].to_numpy(float) - design @ beta
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def participant_pairs(residuals: pd.DataFrame, volume_mode: str = "pooled_token") -> pd.DataFrame:
    token_counts = (
        residuals.groupby(["participant_id", "half"], sort=True)
        .size()
        .rename("token_count")
        .reset_index()
    )
    if volume_mode == "pooled_token":
        summary = (
            residuals.groupby(["participant_id", "half"], sort=True)["residual"]
            .median()
            .rename("median")
            .reset_index()
        )
    elif volume_mode == "utterance_medians":
        utterance = residuals.groupby(["participant_id", "half", "utterance_index"], sort=True)["residual"].median().reset_index()
        summary = (
            utterance.groupby(["participant_id", "half"], sort=True)["residual"]
            .median()
            .rename("median")
            .reset_index()
        )
    else:
        raise ValueError("unknown volume mode")
    summary = summary.merge(token_counts, on=["participant_id", "half"], validate="one_to_one")
    summary = summary.loc[summary["token_count"] >= MIN_TOKENS_PER_HALF]
    pivot = summary.pivot(index="participant_id", columns="half", values="median")
    if 0 not in pivot.columns or 1 not in pivot.columns:
        raise ValueError("one split half is absent")
    pivot = pivot[[0, 1]].dropna().rename(columns={0: "half_0", 1: "half_1"})
    return pivot.reset_index()


def stability(frame: pd.DataFrame, feature: str, assignment: pd.Series, volume_mode: str = "pooled_token") -> dict:
    work = feature_frame(frame, feature, assignment)
    residuals = residualize_by_half(work, feature)
    pairs = participant_pairs(residuals, volume_mode)
    raw, corrected = pearson_pair(pairs)
    return {"raw_half_correlation": raw, "corrected_reliability": corrected, "paired_participants": len(pairs), "pairs": pairs}


def participant_bootstrap(pairs: pd.DataFrame) -> dict:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    values: list[float] = []
    failures: list[dict] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(pairs), size=len(pairs))
        sampled = pairs.iloc[indices]
        try:
            _, corrected = pearson_pair(sampled)
            if not np.isfinite(corrected):
                raise ValueError("nonfinite corrected reliability")
            values.append(corrected)
        except Exception as error:
            failures.append({"replicate": replicate, "error_type": type(error).__name__})
    return {
        "type": "participant_cluster_nonparametric",
        "rng": "numpy.random.PCG64",
        "seed": BOOTSTRAP_SEED,
        "requested_replicates": BOOTSTRAP_REPLICATES,
        "successful_replicates": len(values),
        "failed_replicates": len(failures),
        "failed_details": failures,
        "ci_95": [percentile(values, 0.025), percentile(values, 0.975)],
        "technical_review_required": len(values) < BOOTSTRAP_MIN_SUCCESS,
    }


def leave_one_participant_out(pairs: pd.DataFrame, primary: float) -> dict:
    records = []
    for participant in pairs["participant_id"]:
        subset = pairs.loc[~pairs["participant_id"].eq(participant)]
        try:
            _, corrected = pearson_pair(subset)
            records.append({"deleted_participant": participant, "status": "complete", "corrected_reliability": corrected, "absolute_delta": abs(corrected - primary)})
        except Exception as error:
            records.append({"deleted_participant": participant, "status": "failed", "error_type": type(error).__name__})
    complete = [record["absolute_delta"] for record in records if record["status"] == "complete"]
    return {"records": records, "completed": len(complete), "failed": len(records) - len(complete), "max_absolute_delta": max(complete) if complete else None}


def analyze_feature(frame: pd.DataFrame, feature: str) -> tuple[dict, dict]:
    primary = stability(frame, feature, chronological_assignment(frame))
    bootstrap = participant_bootstrap(primary["pairs"])
    salted = []
    for split in range(SALTED_SPLITS):
        try:
            result = stability(frame, feature, salted_assignment(frame, split))
            salted.append({"split": split, "status": "complete", "paired_participants": result["paired_participants"], "corrected_reliability": result["corrected_reliability"], "absolute_delta": abs(result["corrected_reliability"] - primary["corrected_reliability"])})
        except Exception as error:
            salted.append({"split": split, "status": "failed", "error_type": type(error).__name__})
    volume = stability(frame, feature, chronological_assignment(frame), "utterance_medians")
    volume_delta = abs(volume["corrected_reliability"] - primary["corrected_reliability"])
    influence = leave_one_participant_out(primary["pairs"], primary["corrected_reliability"])
    salted_deltas = [record["absolute_delta"] for record in salted if record["status"] == "complete"]
    robustness_values = salted_deltas + [volume_delta]
    if influence["max_absolute_delta"] is not None:
        robustness_values.append(float(influence["max_absolute_delta"]))
    max_delta = max(robustness_values) if robustness_values else None
    lower = bootstrap["ci_95"][0]
    technical_failures = []
    if bootstrap["technical_review_required"]:
        technical_failures.append("bootstrap_success_below_950")
    if len(salted_deltas) != SALTED_SPLITS:
        technical_failures.append("salted_split_incomplete")
    if influence["failed"]:
        technical_failures.append("participant_influence_incomplete")
    if lower is None or max_delta is None:
        technical_failures.append("required_gate_quantity_absent")
    if technical_failures:
        gate_status = "technical_review_no_decision"
    else:
        passed = primary["corrected_reliability"] >= GATE_RELIABILITY and float(lower) >= GATE_CI_LOWER and float(max_delta) <= GATE_MAX_DELTA
        gate_status = "pass" if passed else "fail"
    private = {
        "feature": feature,
        "primary_split": "chronological_odd_even_utterance_index",
        "nuisance_model": "feature ~ 1 + relative_word_position; duration additionally adjusts for authoritative syllable count; fit separately by half",
        "primary": {key: value for key, value in primary.items() if key != "pairs"},
        "bootstrap": bootstrap,
        "salted_balanced_splits": salted,
        "participant_influence": influence,
        "volume_weighting": {"mode": "median_of_utterance_medians", "paired_participants": volume["paired_participants"], "corrected_reliability": volume["corrected_reliability"], "absolute_delta": volume_delta},
        "max_robustness_delta": max_delta,
        "gate": {"status": gate_status, "corrected_reliability": primary["corrected_reliability"], "ci_95": bootstrap["ci_95"], "max_robustness_delta": max_delta, "thresholds": {"corrected_reliability": GATE_RELIABILITY, "ci_lower": GATE_CI_LOWER, "max_robustness_delta": GATE_MAX_DELTA}, "technical_failures": technical_failures},
    }
    technical = {
        "feature": feature,
        "status": "complete" if not technical_failures else "technical_review",
        "paired_participants": primary["paired_participants"],
        "bootstrap_successful_replicates": bootstrap["successful_replicates"],
        "bootstrap_failed_replicates": bootstrap["failed_replicates"],
        "salted_splits_completed": len(salted_deltas),
        "participant_deletions_completed": influence["completed"],
        "participant_deletions_failed": influence["failed"],
        "scientific_estimates_exposed": False,
    }
    return private, technical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--extraction-closure", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--technical-output", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.token_dir, args.extraction_closure, args.private_output, args.technical_output):
        reject(path)
    closure = verify_closure(args.extraction_closure)
    frame = load_tokens(args.token_dir)
    private_features = []
    technical_features = []
    for feature in FEATURES:
        private, technical = analyze_feature(frame, feature)
        private_features.append(private)
        technical_features.append(technical)
        print(json.dumps({"feature": feature, "status": technical["status"], "scientific_estimates_exposed": False}), flush=True)
    private_report = {
        "analysis": "daic_aggregate_reliability_rq2",
        "protocol": "clinical_regime_amendment_v2_plus_v3_1_addendum",
        "input_closure_sha256": file_sha(args.extraction_closure),
        "labels_loaded": False,
        "outcome_analysis_performed": False,
        "features": private_features,
    }
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(json.dumps(private_report, indent=2), encoding="utf-8")
    technical_report = {
        "analysis": "daic_aggregate_reliability_rq2_blind_technical",
        "input_closure_sha256": file_sha(args.extraction_closure),
        "private_output_sha256": file_sha(args.private_output),
        "labels_loaded": False,
        "scientific_estimates_inspected": False,
        "features": technical_features,
        "all_features_complete": all(record["status"] == "complete" for record in technical_features),
    }
    args.technical_output.parent.mkdir(parents=True, exist_ok=True)
    args.technical_output.write_text(json.dumps(technical_report, indent=2), encoding="utf-8")
    print(json.dumps({"all_features_complete": technical_report["all_features_complete"], "labels_loaded": False, "scientific_estimates_exposed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
