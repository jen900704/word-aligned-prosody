#!/usr/bin/env python3
"""Outcome-blind PTSD-STOP recording-set stability under the frozen amendment.

Scientific estimates are written only to a PRIVATE JSON.  Standard output and
the technical JSON contain support and completion information only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .daic_aggregate_reliability_blind_v1 import (
        BOOTSTRAP_MIN_SUCCESS,
        BOOTSTRAP_REPLICATES,
        FEATURES,
        GATE_CI_LOWER,
        GATE_MAX_DELTA,
        GATE_RELIABILITY,
        MIN_PAIRED_PARTICIPANTS,
        MIN_TOKENS_PER_HALF,
        SALTED_SPLITS,
        feature_frame,
        file_sha,
        leave_one_participant_out,
        participant_bootstrap,
        pearson_pair,
        residualize_by_half,
    )
except ImportError:  # direct script execution
    from daic_aggregate_reliability_blind_v1 import (
        BOOTSTRAP_MIN_SUCCESS,
        BOOTSTRAP_REPLICATES,
        FEATURES,
        GATE_CI_LOWER,
        GATE_MAX_DELTA,
        GATE_RELIABILITY,
        MIN_PAIRED_PARTICIPANTS,
        MIN_TOKENS_PER_HALF,
        SALTED_SPLITS,
        feature_frame,
        file_sha,
        leave_one_participant_out,
        participant_bootstrap,
        pearson_pair,
        residualize_by_half,
    )

EXPECTED_RECORDINGS = 13_600
EXPECTED_PARTICIPANTS = 225
EXPECTED_ANALYSIS_PARTICIPANTS = 219
IDENTITY_AUTHORITY_SHA256 = "465e17c90586f922ba6a9ea878d303bedf5e66b0b5e2d4a8f25d230bd2404002"
EXPECTED_CLOSURE_STATUS = "TECHNICAL_EXTRACTION_COMPLETE"
PROTOCOL = "clinical_regime_amendment_v2_plus_v3_1_plus_ptsd_stop_ingestion_v1_3"
PRIMARY_SPLIT_SALT = "ptsd-stop-primary-recording-split-v1"
SOURCE_COLUMNS = (
    "participant_id",
    "recording_id",
    "role_assignment_tie",
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
FORBIDDEN = ("phq", "pcl", "label", "outcome", "diagnos", "demograph", "crosswalk")
IDENTITY_COLUMNS = (
    "recording_id",
    "source_participant_id",
    "authoritative_participant_id",
    "identity_match_status",
)


def reject(path: Path) -> None:
    if any(term in str(path).lower() for term in FORBIDDEN):
        raise ValueError(f"forbidden clinical path: {path}")


def verify_closure(path: Path) -> dict:
    reject(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "status": EXPECTED_CLOSURE_STATUS,
        "protocol": PROTOCOL,
        "recordings": EXPECTED_RECORDINGS,
        "participants": EXPECTED_PARTICIPANTS,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_results_inspected": False,
        "reliability_analysis_performed": False,
        "outcome_analysis_performed": False,
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise ValueError(f"extraction closure mismatch for {key}")
    return record


def load_token_paths(
    paths: list[Path], expected_recordings: int, expected_participants: int
) -> pd.DataFrame:
    if len(paths) != expected_recordings:
        raise ValueError(
            f"expected {expected_recordings} token files, found {len(paths)}"
        )
    frames = []
    empty_recording_ids = []
    for path in paths:
        # Recording basenames end in opaque random identifiers that can contain
        # clinical-looking letter sequences by chance. The token directory is
        # the allowlist boundary; recording identity is validated below.
        reject(path.parent)
        frame = pd.read_csv(path, usecols=SOURCE_COLUMNS, low_memory=False)
        if frame.empty:
            empty_recording_ids.append(path.stem)
            continue
        if frame["recording_id"].astype(str).nunique() != 1:
            raise ValueError(f"token file does not contain one recording: {path}")
        if str(frame["recording_id"].iloc[0]) != path.stem:
            raise ValueError(f"recording ID does not match token filename: {path}")
        frames.append(frame)
    if not frames:
        raise ValueError("all token files are empty")
    frame = pd.concat(frames, ignore_index=True)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["recording_id"] = frame["recording_id"].astype(str)
    if frame["participant_id"].nunique() != expected_participants:
        raise ValueError(
            f"token files do not contain {expected_participants} canonical participants"
        )
    observed_recordings = set(frame["recording_id"].astype(str).unique())
    all_recordings = observed_recordings | set(empty_recording_ids)
    if len(all_recordings) != expected_recordings:
        raise ValueError("nonempty and empty token files do not resolve uniquely")
    frame = frame.loc[frame["alignment_status"].eq("aligned")].copy()
    frame["utterance_index"] = pd.to_numeric(frame["utterance_index"], errors="raise").astype(int)
    frame["word_index"] = pd.to_numeric(frame["word_index"], errors="raise").astype(int)
    maximum = frame.groupby(
        ["recording_id", "utterance_index"], sort=False
    )["word_index"].transform("max")
    frame["relative_word_position"] = np.where(
        maximum > 0, frame["word_index"] / maximum, 0.0
    )
    tie_text = frame["role_assignment_tie"].astype(str).str.lower()
    if not set(tie_text.unique()).issubset({"true", "false"}):
        raise ValueError("role_assignment_tie is not Boolean")
    frame["role_assignment_tie"] = tie_text.eq("true")
    frame.attrs["token_files"] = expected_recordings
    frame.attrs["empty_token_files"] = len(empty_recording_ids)
    return frame


def load_tokens(token_dir: Path) -> pd.DataFrame:
    reject(token_dir)
    paths = sorted(token_dir.glob("*.csv"))
    return load_token_paths(paths, EXPECTED_RECORDINGS, EXPECTED_PARTICIPANTS)


def apply_identity_authority(
    frame: pd.DataFrame,
    path: Path,
    expected_sha256: str = IDENTITY_AUTHORITY_SHA256,
    expected_recordings: int = EXPECTED_RECORDINGS,
    expected_participants: int = EXPECTED_ANALYSIS_PARTICIPANTS,
) -> pd.DataFrame:
    reject(path)
    if file_sha(path) != expected_sha256:
        raise ValueError("identity authority checksum mismatch")
    identity = pd.read_csv(path, dtype=str, usecols=IDENTITY_COLUMNS)
    if len(identity) != expected_recordings:
        raise ValueError("identity authority recording cardinality mismatch")
    if identity["recording_id"].duplicated().any():
        raise ValueError("identity authority recording IDs are not unique")
    if not identity["identity_match_status"].eq("one_authoritative_user").all():
        raise ValueError("identity authority contains unresolved users")
    if identity["authoritative_participant_id"].nunique() != expected_participants:
        raise ValueError("identity authority participant cardinality mismatch")
    attrs = dict(frame.attrs)
    merged = frame.merge(
        identity,
        on="recording_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("token recording is absent from identity authority")
    if not merged["participant_id"].eq(merged["source_participant_id"]).all():
        raise ValueError("token source participant conflicts with identity authority")
    merged["participant_id"] = merged["authoritative_participant_id"]
    merged = merged.drop(columns=[
        "source_participant_id",
        "authoritative_participant_id",
        "identity_match_status",
        "_merge",
    ])
    if merged["participant_id"].nunique() != expected_participants:
        raise ValueError("nonempty tokens do not contain all authority participants")
    merged.attrs.update(attrs)
    merged.attrs["source_participants"] = EXPECTED_PARTICIPANTS
    merged.attrs["analysis_participants"] = expected_participants
    merged.attrs["identity_authority_sha256"] = expected_sha256
    return merged


def recording_assignment(frame: pd.DataFrame, salt: str) -> pd.Series:
    unique = frame[["participant_id", "recording_id"]].drop_duplicates()
    mapping: dict[tuple[str, str], int] = {}
    for participant, group in unique.groupby("participant_id", sort=True):
        recordings = sorted(
            (str(value) for value in group["recording_id"]),
            key=lambda value: hashlib.sha256(
                f"{salt}|{participant}|{value}".encode("utf-8")
            ).digest(),
        )
        for rank, recording in enumerate(recordings):
            mapping[(str(participant), recording)] = rank % 2
    return pd.Series(
        [
            mapping[(participant, recording)]
            for participant, recording in zip(
                frame["participant_id"], frame["recording_id"]
            )
        ],
        index=frame.index,
        dtype=int,
    )


def participant_pairs(
    residuals: pd.DataFrame, volume_mode: str = "pooled_token"
) -> pd.DataFrame:
    token_counts = (
        residuals.groupby(["participant_id", "half"], sort=True)
        .size()
        .rename("token_count")
        .reset_index()
    )
    recording_counts = (
        residuals.groupby(["participant_id", "half"], sort=True)["recording_id"]
        .nunique()
        .rename("recording_count")
        .reset_index()
    )
    if volume_mode == "pooled_token":
        summary = (
            residuals.groupby(["participant_id", "half"], sort=True)["residual"]
            .median()
            .rename("median")
            .reset_index()
        )
    elif volume_mode == "recording_medians":
        recording = (
            residuals.groupby(
                ["participant_id", "half", "recording_id"], sort=True
            )["residual"]
            .median()
            .reset_index()
        )
        summary = (
            recording.groupby(["participant_id", "half"], sort=True)["residual"]
            .median()
            .rename("median")
            .reset_index()
        )
    else:
        raise ValueError("unknown volume mode")
    summary = summary.merge(
        token_counts, on=["participant_id", "half"], validate="one_to_one"
    ).merge(
        recording_counts, on=["participant_id", "half"], validate="one_to_one"
    )
    summary = summary.loc[summary["token_count"] >= MIN_TOKENS_PER_HALF]
    pivot = summary.pivot(index="participant_id", columns="half", values="median")
    if 0 not in pivot.columns or 1 not in pivot.columns:
        raise ValueError("one recording-set half is absent")
    pivot = pivot[[0, 1]].dropna().rename(columns={0: "half_0", 1: "half_1"})
    return pivot.reset_index()


def stability(
    frame: pd.DataFrame, feature: str, assignment: pd.Series,
    volume_mode: str = "pooled_token",
) -> dict:
    work = feature_frame(frame, feature, assignment)
    work["recording_id"] = frame.loc[work.index, "recording_id"].astype(str)
    residuals = residualize_by_half(work, feature)
    pairs = participant_pairs(residuals, volume_mode)
    raw, corrected = pearson_pair(pairs)
    return {
        "raw_half_correlation": raw,
        "corrected_reliability": corrected,
        "paired_participants": len(pairs),
        "pairs": pairs,
    }


def analyze_feature(frame: pd.DataFrame, feature: str) -> tuple[dict, dict]:
    primary_assignment = recording_assignment(frame, PRIMARY_SPLIT_SALT)
    primary = stability(frame, feature, primary_assignment)
    bootstrap = participant_bootstrap(primary["pairs"])

    salted = []
    for split in range(SALTED_SPLITS):
        try:
            result = stability(
                frame,
                feature,
                recording_assignment(frame, f"ptsd-stop-balanced-recording-{split:02d}"),
            )
            salted.append({
                "split": split,
                "status": "complete",
                "paired_participants": result["paired_participants"],
                "corrected_reliability": result["corrected_reliability"],
                "absolute_delta": abs(
                    result["corrected_reliability"]
                    - primary["corrected_reliability"]
                ),
            })
        except Exception as error:
            salted.append({
                "split": split,
                "status": "failed",
                "error_type": type(error).__name__,
            })

    volume = stability(frame, feature, primary_assignment, "recording_medians")
    volume_delta = abs(
        volume["corrected_reliability"] - primary["corrected_reliability"]
    )
    influence = leave_one_participant_out(
        primary["pairs"], primary["corrected_reliability"]
    )

    non_tie = frame.loc[~frame["role_assignment_tie"]].copy()
    tie_sensitivity = stability(
        non_tie,
        feature,
        recording_assignment(non_tie, PRIMARY_SPLIT_SALT),
    )
    tie_delta = abs(
        tie_sensitivity["corrected_reliability"]
        - primary["corrected_reliability"]
    )

    salted_deltas = [
        record["absolute_delta"]
        for record in salted
        if record["status"] == "complete"
    ]
    robustness_values = salted_deltas + [volume_delta, tie_delta]
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
        passed = (
            primary["corrected_reliability"] >= GATE_RELIABILITY
            and float(lower) >= GATE_CI_LOWER
            and float(max_delta) <= GATE_MAX_DELTA
        )
        gate_status = "pass" if passed else "fail"

    private = {
        "feature": feature,
        "primary_split": "deterministic_balanced_whole_recording_halves",
        "primary_split_salt": PRIMARY_SPLIT_SALT,
        "nuisance_model": (
            "feature ~ 1 + relative_word_position; duration additionally adjusts "
            "for authoritative syllable count; fit separately by half"
        ),
        "primary": {key: value for key, value in primary.items() if key != "pairs"},
        "bootstrap": bootstrap,
        "salted_balanced_recording_splits": salted,
        "participant_influence": influence,
        "volume_weighting": {
            "mode": "median_of_recording_medians",
            "paired_participants": volume["paired_participants"],
            "corrected_reliability": volume["corrected_reliability"],
            "absolute_delta": volume_delta,
        },
        "tie_exclusion_sensitivity": {
            "mode": "exclude_operational_role_tie_recordings",
            "paired_participants": tie_sensitivity["paired_participants"],
            "corrected_reliability": tie_sensitivity["corrected_reliability"],
            "absolute_delta": tie_delta,
        },
        "max_robustness_delta": max_delta,
        "gate": {
            "status": gate_status,
            "corrected_reliability": primary["corrected_reliability"],
            "ci_95": bootstrap["ci_95"],
            "max_robustness_delta": max_delta,
            "thresholds": {
                "corrected_reliability": GATE_RELIABILITY,
                "ci_lower": GATE_CI_LOWER,
                "max_robustness_delta": GATE_MAX_DELTA,
            },
            "technical_failures": technical_failures,
        },
    }
    technical = {
        "feature": feature,
        "status": "complete" if not technical_failures else "technical_review",
        "paired_participants": primary["paired_participants"],
        "non_tie_paired_participants": tie_sensitivity["paired_participants"],
        "bootstrap_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_successful_replicates": bootstrap["successful_replicates"],
        "bootstrap_failed_replicates": bootstrap["failed_replicates"],
        "salted_splits_completed": len(salted_deltas),
        "participant_deletions_completed": influence["completed"],
        "participant_deletions_failed": influence["failed"],
        "tie_exclusion_completed": True,
        "scientific_estimates_exposed": False,
    }
    return private, technical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--extraction-closure", required=True, type=Path)
    parser.add_argument("--identity-authority", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--technical-output", required=True, type=Path)
    args = parser.parse_args()
    for path in (
        args.token_dir,
        args.extraction_closure,
        args.identity_authority,
        args.private_output,
        args.technical_output,
    ):
        reject(path)
    verify_closure(args.extraction_closure)
    frame = apply_identity_authority(
        load_tokens(args.token_dir), args.identity_authority
    )
    private_features = []
    technical_features = []
    for feature in FEATURES:
        private, technical = analyze_feature(frame, feature)
        private_features.append(private)
        technical_features.append(technical)
        print(json.dumps({
            "feature": feature,
            "status": technical["status"],
            "scientific_estimates_exposed": False,
        }), flush=True)
    private_report = {
        "analysis": "ptsd_stop_aggregate_recording_set_reliability_rq2",
        "protocol": PROTOCOL,
        "input_closure_sha256": file_sha(args.extraction_closure),
        "identity_authority_sha256": file_sha(args.identity_authority),
        "labels_loaded": False,
        "outcomes_loaded": False,
        "outcome_analysis_performed": False,
        "corpus_support": {
            "token_files": int(frame.attrs.get("token_files", EXPECTED_RECORDINGS)),
            "empty_token_files": int(frame.attrs.get("empty_token_files", 0)),
            "source_participants": int(frame.attrs["source_participants"]),
            "analysis_participants": int(frame.attrs["analysis_participants"]),
        },
        "features": private_features,
    }
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(
        json.dumps(private_report, indent=2), encoding="utf-8"
    )
    technical_report = {
        "analysis": "ptsd_stop_aggregate_recording_set_reliability_rq2_blind_technical",
        "input_closure_sha256": file_sha(args.extraction_closure),
        "identity_authority_sha256": file_sha(args.identity_authority),
        "private_output_sha256": file_sha(args.private_output),
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_estimates_inspected": False,
        "corpus_support": {
            "token_files": int(frame.attrs.get("token_files", EXPECTED_RECORDINGS)),
            "empty_token_files": int(frame.attrs.get("empty_token_files", 0)),
            "source_participants": int(frame.attrs["source_participants"]),
            "analysis_participants": int(frame.attrs["analysis_participants"]),
        },
        "features": technical_features,
        "all_features_complete": all(
            record["status"] == "complete" for record in technical_features
        ),
    }
    args.technical_output.parent.mkdir(parents=True, exist_ok=True)
    args.technical_output.write_text(
        json.dumps(technical_report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "all_features_complete": technical_report["all_features_complete"],
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_estimates_exposed": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
