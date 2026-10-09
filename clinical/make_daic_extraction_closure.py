#!/usr/bin/env python3
"""Create an outcome-blind SAFE closure from a DAIC technical report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PROTOCOL = "clinical_regime_amendment_v2_plus_v3_1_addendum"
EXPECTED_EXTRACTOR = "daic_frozen_esd_contract_v3"
EXPECTED_N = 189
EXPECTED_CMU = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
EXPECTED_ALIGNER = "488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d"
FROZEN_GIT_COMMIT = "07537eca85806b516b895154ad20b706a7341ae8"
FROZEN_GIT_TAG = "clinical-regime-amendment-v2"
FORBIDDEN = ("phq", "pcl", "label", "outcome", "diagnos", "demograph")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject(path: Path) -> None:
    if any(part in str(path).lower() for part in FORBIDDEN):
        raise ValueError(f"forbidden clinical path: {path}")


def build_closure(report_path: Path) -> dict:
    reject(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exact = {
        "protocol": EXPECTED_PROTOCOL,
        "extractor_version": EXPECTED_EXTRACTOR,
        "n": EXPECTED_N,
        "labels_loaded": False,
        "scientific_results_inspected": False,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(f"technical report mismatch for {key}")
    hashes = report.get("resource_hashes", {})
    if hashes.get("cmudict_sha256") != EXPECTED_CMU:
        raise ValueError("CMUdict hash mismatch")
    if hashes.get("alignment_checkpoint_sha256") != EXPECTED_ALIGNER:
        raise ValueError("alignment checkpoint hash mismatch")
    participants = report.get("participants", [])
    if len(participants) != EXPECTED_N:
        raise ValueError("participant report count mismatch")
    ids = [str(record.get("participant_id")) for record in participants]
    if len(set(ids)) != EXPECTED_N:
        raise ValueError("participant IDs are not unique")
    normalized = sum(int(record["normalized_tokens"]) for record in participants)
    aligned = sum(int(record["aligned_tokens"]) for record in participants)
    unaligned = sum(int(record["unaligned_tokens"]) for record in participants)
    utterances = sum(int(record["participant_utterances"]) for record in participants)
    failures = sum(int(record["utterance_alignment_failures"]) for record in participants)
    if aligned + unaligned != normalized:
        raise ValueError("token accounting mismatch")
    if failures:
        raise ValueError("utterance alignment failure present")
    if aligned <= 0:
        raise ValueError("no aligned tokens")
    return {
        "stage": "DAIC full acoustic extraction",
        "status": "TECHNICAL_EXTRACTION_COMPLETE",
        "protocol": EXPECTED_PROTOCOL,
        "extractor_version": EXPECTED_EXTRACTOR,
        "frozen_git_commit": FROZEN_GIT_COMMIT,
        "frozen_git_tag": FROZEN_GIT_TAG,
        "source_report_sha256": file_sha(report_path),
        "participants": EXPECTED_N,
        "participant_utterances": utterances,
        "normalized_tokens": normalized,
        "aligned_tokens": aligned,
        "unaligned_tokens": unaligned,
        "alignment_rate": aligned / normalized,
        "participants_with_any_unaligned": sum(
            int(record["unaligned_tokens"]) > 0 for record in participants
        ),
        "utterance_alignment_failures": failures,
        "labels_loaded": False,
        "scientific_results_inspected": False,
        "resource_hashes": hashes,
        "runtime": report.get("runtime"),
        "total_s": float(report["total_s"]),
        "median_participant_s": float(report["median_participant_s"]),
        "reliability_analysis_performed": False,
        "outcome_analysis_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reject(args.output)
    closure = build_closure(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(closure, indent=2), encoding="utf-8")
    print(json.dumps({"status": closure["status"], "participants": closure["participants"], "labels_loaded": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
