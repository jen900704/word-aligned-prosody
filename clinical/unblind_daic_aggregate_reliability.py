#!/usr/bin/env python3
"""Verify blind DAIC RQ2 completion, then expose the locked reliability report."""
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


def validate_and_expose(private_path: Path, technical_path: Path) -> dict:
    reject(private_path)
    reject(technical_path)
    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    if technical.get("analysis") != "daic_aggregate_reliability_rq2_blind_technical":
        raise ValueError("unexpected technical analysis identifier")
    if technical.get("labels_loaded") is not False:
        raise ValueError("technical report does not preserve outcome blinding")
    if technical.get("scientific_estimates_inspected") is not False:
        raise ValueError("technical report says estimates were inspected")
    if technical.get("all_features_complete") is not True:
        raise ValueError("blind aggregate reliability is not technically complete")
    technical_features = technical.get("features", [])
    if tuple(record.get("feature") for record in technical_features) != EXPECTED_FEATURES:
        raise ValueError("technical feature order mismatch")
    if any(record.get("status") != "complete" for record in technical_features):
        raise ValueError("technical feature is not complete")
    if technical.get("private_output_sha256") != file_sha(private_path):
        raise ValueError("PRIVATE output hash mismatch")
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if private.get("analysis") != "daic_aggregate_reliability_rq2":
        raise ValueError("unexpected PRIVATE analysis identifier")
    if private.get("labels_loaded") is not False or private.get("outcome_analysis_performed") is not False:
        raise ValueError("PRIVATE report does not preserve outcome blinding")
    features = private.get("features", [])
    if tuple(record.get("feature") for record in features) != EXPECTED_FEATURES:
        raise ValueError("PRIVATE feature order mismatch")
    exposed = []
    for record in features:
        gate = record.get("gate", {})
        corrected = float(gate["corrected_reliability"])
        interval = [float(value) for value in gate["ci_95"]]
        max_delta = float(gate["max_robustness_delta"])
        if not all(math.isfinite(value) for value in (corrected, *interval, max_delta)):
            raise ValueError("nonfinite required gate quantity")
        expected_pass = corrected >= GATE_RELIABILITY and interval[0] >= GATE_CI_LOWER and max_delta <= GATE_MAX_DELTA
        expected_status = "pass" if expected_pass else "fail"
        if gate.get("status") != expected_status or gate.get("technical_failures"):
            raise ValueError("stored gate status conflicts with frozen thresholds")
        primary = record["primary"]
        bootstrap = record["bootstrap"]
        exposed.append(
            {
                "feature": record["feature"],
                "paired_participants": int(primary["paired_participants"]),
                "raw_half_correlation": float(primary["raw_half_correlation"]),
                "corrected_reliability": corrected,
                "ci_95": interval,
                "bootstrap_successful_replicates": int(bootstrap["successful_replicates"]),
                "bootstrap_failed_replicates": int(bootstrap["failed_replicates"]),
                "max_robustness_delta": max_delta,
                "status": expected_status,
            }
        )
    return {
        "analysis": "daic_aggregate_reliability_rq2_unblinded",
        "split_unit": "chronologically ordered odd/even participant utterances",
        "interpretation": "within-interview aggregate consistency; not temporal test-retest and not lexical repeatability",
        "private_output_sha256": file_sha(private_path),
        "technical_output_sha256": file_sha(technical_path),
        "labels_loaded": False,
        "outcome_analysis_performed": False,
        "gate_thresholds": {
            "corrected_reliability": GATE_RELIABILITY,
            "ci_lower": GATE_CI_LOWER,
            "max_robustness_delta": GATE_MAX_DELTA,
        },
        "features": exposed,
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
    print(json.dumps({"status": "UNBLINDED", "feature_statuses": {record["feature"]: record["status"] for record in report["features"]}, "labels_loaded": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
