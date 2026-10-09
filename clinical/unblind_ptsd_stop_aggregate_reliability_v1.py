#!/usr/bin/env python3
"""Verify blind PTSD-STOP RQ2 completion and expose only the frozen gate report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_FEATURES = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "syllable_adjusted_log_duration",
)
GATE_RELIABILITY = 0.60
GATE_CI_LOWER = 0.40
GATE_MAX_DELTA = 0.15
IDENTITY_AUTHORITY_SHA256 = "465e17c90586f922ba6a9ea878d303bedf5e66b0b5e2d4a8f25d230bd2404002"
EXPECTED_SOURCE_PARTICIPANTS = 225
EXPECTED_ANALYSIS_PARTICIPANTS = 219
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


def finite(*values: float) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("nonfinite required aggregate reliability quantity")


def validate_and_expose(private_path: Path, technical_path: Path) -> dict:
    reject(private_path)
    reject(technical_path)
    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    if technical.get("analysis") != (
        "ptsd_stop_aggregate_recording_set_reliability_rq2_blind_technical"
    ):
        raise ValueError("unexpected technical analysis identifier")
    exact_false = (
        "labels_loaded",
        "outcomes_loaded",
        "scientific_estimates_inspected",
    )
    if any(technical.get(key) is not False for key in exact_false):
        raise ValueError("technical report does not preserve blinding")
    if technical.get("all_features_complete") is not True:
        raise ValueError("blind aggregate reliability is not technically complete")
    technical_support = technical.get("corpus_support", {})
    if technical.get("identity_authority_sha256") != IDENTITY_AUTHORITY_SHA256:
        raise ValueError("technical identity authority mismatch")
    if technical_support.get("source_participants") != EXPECTED_SOURCE_PARTICIPANTS:
        raise ValueError("technical source participant count mismatch")
    if technical_support.get("analysis_participants") != EXPECTED_ANALYSIS_PARTICIPANTS:
        raise ValueError("technical analysis participant count mismatch")
    technical_features = technical.get("features", [])
    if tuple(record.get("feature") for record in technical_features) != EXPECTED_FEATURES:
        raise ValueError("technical feature order mismatch")
    if any(record.get("status") != "complete" for record in technical_features):
        raise ValueError("technical feature is not complete")
    if technical.get("private_output_sha256") != file_sha(private_path):
        raise ValueError("PRIVATE output hash mismatch")

    private = json.loads(private_path.read_text(encoding="utf-8"))
    if private.get("analysis") != "ptsd_stop_aggregate_recording_set_reliability_rq2":
        raise ValueError("unexpected PRIVATE analysis identifier")
    if private.get("labels_loaded") is not False:
        raise ValueError("PRIVATE report loaded labels")
    if private.get("outcomes_loaded") is not False:
        raise ValueError("PRIVATE report loaded outcomes")
    if private.get("outcome_analysis_performed") is not False:
        raise ValueError("PRIVATE report performed outcome analysis")
    private_support = private.get("corpus_support", {})
    if private.get("identity_authority_sha256") != IDENTITY_AUTHORITY_SHA256:
        raise ValueError("PRIVATE identity authority mismatch")
    if private_support != technical_support:
        raise ValueError("PRIVATE and technical corpus support mismatch")
    features = private.get("features", [])
    if tuple(record.get("feature") for record in features) != EXPECTED_FEATURES:
        raise ValueError("PRIVATE feature order mismatch")

    exposed = []
    for record in features:
        primary = record["primary"]
        bootstrap = record["bootstrap"]
        gate = record["gate"]
        corrected = float(gate["corrected_reliability"])
        interval = [float(value) for value in gate["ci_95"]]
        max_delta = float(gate["max_robustness_delta"])
        finite(corrected, *interval, max_delta)
        expected_pass = (
            corrected >= GATE_RELIABILITY
            and interval[0] >= GATE_CI_LOWER
            and max_delta <= GATE_MAX_DELTA
        )
        expected_status = "pass" if expected_pass else "fail"
        if gate.get("status") != expected_status or gate.get("technical_failures"):
            raise ValueError("stored gate status conflicts with frozen thresholds")

        volume = record["volume_weighting"]
        tie = record["tie_exclusion_sensitivity"]
        influence = record["participant_influence"]
        salted = record["salted_balanced_recording_splits"]
        if len(salted) != 25 or any(item.get("status") != "complete" for item in salted):
            raise ValueError("salted split sensitivity is incomplete")
        salted_values = [float(item["corrected_reliability"]) for item in salted]
        finite(
            primary["raw_half_correlation"],
            volume["corrected_reliability"],
            volume["absolute_delta"],
            tie["corrected_reliability"],
            tie["absolute_delta"],
            influence["max_absolute_delta"],
            *salted_values,
        )
        exposed.append({
            "feature": record["feature"],
            "paired_participants": int(primary["paired_participants"]),
            "raw_half_correlation": float(primary["raw_half_correlation"]),
            "corrected_reliability": corrected,
            "ci_95": interval,
            "bootstrap_successful_replicates": int(bootstrap["successful_replicates"]),
            "bootstrap_failed_replicates": int(bootstrap["failed_replicates"]),
            "salted_split_corrected_range": [min(salted_values), max(salted_values)],
            "leave_one_participant_out_max_delta": float(influence["max_absolute_delta"]),
            "recording_median_corrected_reliability": float(volume["corrected_reliability"]),
            "recording_median_absolute_delta": float(volume["absolute_delta"]),
            "non_tie_paired_participants": int(tie["paired_participants"]),
            "tie_excluded_corrected_reliability": float(tie["corrected_reliability"]),
            "tie_exclusion_absolute_delta": float(tie["absolute_delta"]),
            "max_robustness_delta": max_delta,
            "status": expected_status,
        })

    licensed = [record["feature"] for record in exposed if record["status"] == "pass"]
    return {
        "analysis": "ptsd_stop_aggregate_reliability_rq2_unblinded",
        "split_unit": "deterministic balanced whole-recording halves within participant",
        "interpretation": (
            "recording-set aggregate stability; not participant-by-word repeatability "
            "and not longitudinal test-retest"
        ),
        "private_output_sha256": file_sha(private_path),
        "technical_output_sha256": file_sha(technical_path),
        "identity_authority_sha256": IDENTITY_AUTHORITY_SHA256,
        "corpus_support": private_support,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "outcome_analysis_performed": False,
        "gate_thresholds": {
            "corrected_reliability": GATE_RELIABILITY,
            "ci_lower": GATE_CI_LOWER,
            "max_robustness_delta": GATE_MAX_DELTA,
        },
        "features": exposed,
        "gate_licensed_features": licensed,
        "pcl_access_licensed": bool(licensed),
        "public_reporting_status": "pending_team_governance_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-input", required=True, type=Path)
    parser.add_argument("--technical-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reject(args.output)
    report = validate_and_expose(args.private_input, args.technical_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "UNBLINDED",
        "feature_statuses": {
            record["feature"]: record["status"] for record in report["features"]
        },
        "pcl_access_licensed": report["pcl_access_licensed"],
        "outcomes_loaded": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
