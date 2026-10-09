#!/usr/bin/env python3
"""Merge four outcome-blind PTSD-STOP extraction reports into a SAFE closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PROTOCOL = "clinical_regime_amendment_v2_plus_v3_1_plus_ptsd_stop_ingestion_v1_3"
EXPECTED_EXTRACTOR = "ptsd_stop_frozen_esd_contract_v1_3"
EXPECTED_RECORDINGS = 13_600
EXPECTED_PARTICIPANTS = 225
EXPECTED_SHARDS = 4
EXPECTED_PER_SHARD = EXPECTED_RECORDINGS // EXPECTED_SHARDS
EXPECTED_CMU = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
EXPECTED_ALIGNER = "488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d"
EXPECTED_MANIFEST = "ed6d568659946452c288e0fec0beb3f6b76ec0ed802f5671976e199a9553633d"
FROZEN_GIT_COMMIT = "4c5b1df68864c26bbe090c22d6436e9ddff83ec2"
FROZEN_GIT_TAG = "ptsd-stop-extraction-schema-v1.1"
FORBIDDEN = ("pcl", "label", "outcome", "diagnos", "demograph", "crosswalk")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject(path: Path) -> None:
    if any(term in str(path).lower() for term in FORBIDDEN):
        raise ValueError(f"forbidden clinical path: {path}")


def _verify_report(report: dict, source: Path) -> None:
    exact = {
        "protocol": EXPECTED_PROTOCOL,
        "extractor_version": EXPECTED_EXTRACTOR,
        "n": EXPECTED_PER_SHARD,
        "expected_full_n": EXPECTED_RECORDINGS,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_results_inspected": False,
        "source_manifest_sha256": EXPECTED_MANIFEST,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(f"{source}: technical report mismatch for {key}")
    hashes = report.get("resource_hashes", {})
    if hashes.get("cmudict_sha256") != EXPECTED_CMU:
        raise ValueError(f"{source}: CMUdict hash mismatch")
    if hashes.get("alignment_checkpoint_sha256") != EXPECTED_ALIGNER:
        raise ValueError(f"{source}: alignment checkpoint hash mismatch")
    if len(report.get("recordings", [])) != EXPECTED_PER_SHARD:
        raise ValueError(f"{source}: recording report count mismatch")


def build_closure(report_paths: list[Path]) -> dict:
    if len(report_paths) != EXPECTED_SHARDS:
        raise ValueError("exactly four shard reports are required")
    reports = []
    source_hashes = {}
    for path in report_paths:
        reject(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        _verify_report(report, path)
        reports.append(report)
        source_hashes[path.name] = file_sha(path)

    shard_indices = [int(report["shard"]["index"]) for report in reports]
    shard_counts = [int(report["shard"]["count"]) for report in reports]
    if set(shard_indices) != set(range(EXPECTED_SHARDS)):
        raise ValueError("shard indices are incomplete or duplicated")
    if set(shard_counts) != {EXPECTED_SHARDS}:
        raise ValueError("shard count mismatch")

    recordings = [record for report in reports for record in report["recordings"]]
    recording_ids = [str(record["recording_id"]) for record in recordings]
    if len(recordings) != EXPECTED_RECORDINGS:
        raise ValueError("full recording count mismatch")
    if len(set(recording_ids)) != EXPECTED_RECORDINGS:
        raise ValueError("recording IDs are not unique")
    participants = {str(record["participant_id"]) for record in recordings}
    if len(participants) != EXPECTED_PARTICIPANTS:
        raise ValueError("canonical participant count mismatch")

    normalized = sum(int(record["normalized_tokens"]) for record in recordings)
    aligned = sum(int(record["aligned_tokens"]) for record in recordings)
    unaligned = sum(int(record["unaligned_tokens"]) for record in recordings)
    failures = sum(int(record["utterance_alignment_failures"]) for record in recordings)
    if aligned + unaligned != normalized:
        raise ValueError("token accounting mismatch")
    if aligned <= 0:
        raise ValueError("no aligned tokens")
    if failures:
        raise ValueError("utterance alignment failure present")
    if any(not record.get("private_output_sha256") for record in recordings):
        raise ValueError("private output hash missing")

    return {
        "stage": "PTSD-STOP full outcome-blind acoustic extraction",
        "status": "TECHNICAL_EXTRACTION_COMPLETE",
        "protocol": EXPECTED_PROTOCOL,
        "extractor_version": EXPECTED_EXTRACTOR,
        "frozen_git_commit": FROZEN_GIT_COMMIT,
        "frozen_git_tag": FROZEN_GIT_TAG,
        "source_manifest_sha256": EXPECTED_MANIFEST,
        "source_report_sha256": source_hashes,
        "recordings": EXPECTED_RECORDINGS,
        "participants": EXPECTED_PARTICIPANTS,
        "role_assignment_tie_recordings": sum(bool(record["role_assignment_tie"]) for record in recordings),
        "valid_participant_intervals": sum(int(record["valid_participant_intervals"]) for record in recordings),
        "invalid_participant_rows": sum(int(record["invalid_participant_rows"]) for record in recordings),
        "empty_participant_rows": sum(int(record["empty_participant_rows"]) for record in recordings),
        "intervals_beyond_audio": sum(int(record["intervals_beyond_audio"]) for record in recordings),
        "normalized_tokens": normalized,
        "aligned_tokens": aligned,
        "unaligned_tokens": unaligned,
        "alignment_rate": aligned / normalized,
        "recordings_with_any_unaligned": sum(int(record["unaligned_tokens"]) > 0 for record in recordings),
        "utterance_alignment_failures": failures,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_results_inspected": False,
        "resource_hashes": reports[0]["resource_hashes"],
        "runtime": {
            "sum_worker_seconds": sum(float(report["total_s"]) for report in reports),
            "shard_total_seconds": {
                str(report["shard"]["index"]): float(report["total_s"])
                for report in reports
            },
        },
        "reliability_analysis_performed": False,
        "outcome_analysis_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs=4, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reject(args.output)
    closure = build_closure(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(closure, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": closure["status"],
        "recordings": closure["recordings"],
        "participants": closure["participants"],
        "labels_loaded": False,
        "outcomes_loaded": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
