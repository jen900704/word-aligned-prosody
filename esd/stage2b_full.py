#!/usr/bin/env python3
"""Deterministic, resumable Stage-2B extraction shards and aggregation only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import esd_stage2_adapter as adapter
from esd_contract import (APPROVED_STAGE1_MANIFEST_SHA256, ContractError,
                          ELIGIBLE_ESD_SPEAKERS, sha256, write_csv, write_json)
from frozen_word_aligner import ALIGN_CHECKPOINT_SHA256, FrozenWordAligner

SHARD_COUNT = 32
EXPECTED_UTTERANCES = 15750
PLAN_FIELDS = ("manifest_row_index", "shard_id", "sample_id")
PLAN_NAME = "stage2b_shard_manifest_PRIVATE.csv"
PLAN_REPORT_NAME = "stage2b_shard_plan_SAFE.json"
COMPLETION_NAME = "shard_completion_SAFE.json"
AGGREGATE_NAME = "esd_word_features_PRIVATE.csv"
QA_NAME = "stage2b_aggregate_QA_SAFE.json"


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contract_record():
    return {
        "adapter_version": adapter.ADAPTER_VERSION,
        "schema": list(adapter.FIELDS),
        "adapter_provenance": adapter.PROVENANCE,
        "approved_manifest_sha256": APPROVED_STAGE1_MANIFEST_SHA256,
        "cmudict_sha256": adapter.CMUDICT_SHA256,
        "alignment_checkpoint_sha256": ALIGN_CHECKPOINT_SHA256,
        "shard_count": SHARD_COUNT,
        "assignment_rule": "zero_based_manifest_row_index_modulo_32",
    }


def create_plan(manifest: Path, output_root: Path):
    rows = adapter.read_manifest(manifest)
    if len(rows) != EXPECTED_UTTERANCES:
        raise ContractError("Stage-2B requires exactly 15750 manifest rows")
    if tuple(sorted({row["speaker_id"] for row in rows})) != ELIGIBLE_ESD_SPEAKERS or any(row["speaker_id"] == "0014" for row in rows):
        raise ContractError("Stage-2B eligible-speaker scope mismatch")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != EXPECTED_UTTERANCES:
        raise ContractError("Stage-2B manifest sample IDs are not unique")
    assignments = [{"manifest_row_index": index, "shard_id": index % SHARD_COUNT, "sample_id": sample_id}
                   for index, sample_id in enumerate(sample_ids)]
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / PLAN_NAME
    write_csv(plan_path, assignments, PLAN_FIELDS)
    counts = Counter(row["shard_id"] for row in assignments)
    report = {
        "plan_version": "esd_stage2b_shard_plan_v1",
        "approved_manifest_sha256": APPROVED_STAGE1_MANIFEST_SHA256,
        "assignment_rule": "zero_based_manifest_row_index_modulo_32",
        "shard_count": SHARD_COUNT,
        "expected_utterances": EXPECTED_UTTERANCES,
        "total_rows": len(assignments),
        "rows_per_shard": {str(i): counts[i] for i in range(SHARD_COUNT)},
        "duplicate_assignments": len(assignments) - len(set(sample_ids)),
        "missing_assignments": EXPECTED_UTTERANCES - len(set(sample_ids)),
        "shard_manifest_sha256": sha256(plan_path),
        "eligible_speakers": list(ELIGIBLE_ESD_SPEAKERS),
        "speaker_0014_excluded": True,
    }
    write_json(output_root / PLAN_REPORT_NAME, report)
    return report


def read_plan(output_root: Path, manifest_rows):
    plan_path = output_root / PLAN_NAME
    report_path = output_root / PLAN_REPORT_NAME
    if not plan_path.is_file() or not report_path.is_file():
        raise ContractError("frozen Stage-2B shard plan absent")
    report = json.loads(report_path.read_text())
    if report.get("shard_manifest_sha256") != sha256(plan_path):
        raise ContractError("Stage-2B shard-plan checksum mismatch")
    with plan_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PLAN_FIELDS:
            raise ContractError("Stage-2B shard-plan schema mismatch")
        assignments = list(reader)
    if len(assignments) != EXPECTED_UTTERANCES or len(manifest_rows) != EXPECTED_UTTERANCES:
        raise ContractError("Stage-2B shard-plan row count mismatch")
    seen = Counter()
    for index, (assignment, manifest_row) in enumerate(zip(assignments, manifest_rows)):
        if int(assignment["manifest_row_index"]) != index or int(assignment["shard_id"]) != index % SHARD_COUNT or assignment["sample_id"] != manifest_row["sample_id"]:
            raise ContractError("Stage-2B shard assignment differs from frozen manifest order")
        seen[assignment["sample_id"]] += 1
    if len(seen) != EXPECTED_UTTERANCES or any(count != 1 for count in seen.values()):
        raise ContractError("Stage-2B shard plan does not assign every sample exactly once")
    return assignments, report


def _unit_path(shard_root: Path, sample_id: str):
    return shard_root / "extraction_PRIVATE" / "by_utterance" / f"{sample_id.replace(':', '__')}.csv"


def valid_completed_unit(path: Path, sample_id: str):
    if not path.is_file(): return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(adapter.FIELDS): return False
            rows = list(reader)
    except (OSError, csv.Error): return False
    identities = [(row.get("sample_id"), row.get("word_index")) for row in rows]
    return bool(rows) and all(row.get("sample_id") == sample_id for row in rows) and len(identities) == len(set(identities))


def run_shard(manifest: Path, output_root: Path, cmudict: Path, model_dir: Path, shard_id: int):
    if not 0 <= shard_id < SHARD_COUNT:
        raise ContractError("shard ID must be in 0..31")
    runtime_path = output_root / "runtime_verification_SAFE.json"
    if not runtime_path.is_file() or json.loads(runtime_path.read_text()).get("all_checks_passed") is not True:
        raise ContractError("successful Stage-2 runtime verification absent")
    manifest_rows = adapter.read_manifest(manifest)
    assignments, plan_report = read_plan(output_root, manifest_rows)
    assigned = [manifest_rows[i] for i, assignment in enumerate(assignments) if int(assignment["shard_id"]) == shard_id]
    shard_root = output_root / "shards" / f"shard_{shard_id:02d}"
    statuses = []
    pending = [row for row in assigned if not valid_completed_unit(_unit_path(shard_root, row["sample_id"]), row["sample_id"])]
    if pending:
        completion_path = shard_root / COMPLETION_NAME
        if completion_path.is_file(): completion_path.unlink()
        cmu = adapter.verify_cmudict(cmudict)
        adapter.verify_alignment_checkpoint(model_dir)
        aligner = FrozenWordAligner(model_dir, "cuda")
        for item in assigned:
            target = _unit_path(shard_root, item["sample_id"])
            if valid_completed_unit(target, item["sample_id"]):
                statuses.append({"sample_id": item["sample_id"], "status": "skipped_valid"})
                continue
            try:
                adapter.atomic_csv(target, adapter.process_utterance(item, aligner, cmu))
                status = "complete" if valid_completed_unit(target, item["sample_id"]) else "invalid_output"
                statuses.append({"sample_id": item["sample_id"], "status": status})
            except Exception as exc:
                statuses.append({"sample_id": item["sample_id"], "status": "failed", "error_code": type(exc).__name__})
    else:
        statuses = [{"sample_id": item["sample_id"], "status": "skipped_valid"} for item in assigned]
    write_json(shard_root / "adapter_run_SAFE.json", {
        "mode": "full_shard", "shard_id": shard_id, "assigned_utterances": len(assigned),
        "statuses": statuses, "gate_e_evaluated": False, "reliability_evaluated": False,
    })
    valid = sum(valid_completed_unit(_unit_path(shard_root, item["sample_id"]), item["sample_id"]) for item in assigned)
    if valid != len(assigned):
        raise ContractError(f"shard {shard_id} incomplete: {valid}/{len(assigned)} valid utterances")
    contract = contract_record()
    marker = {
        "completion_version": "esd_stage2b_shard_completion_v1", "shard_id": shard_id,
        "assigned_utterances": len(assigned), "valid_utterances": valid, "shard_complete": True,
        "shard_manifest_sha256": plan_report["shard_manifest_sha256"],
        "contract": contract, "contract_sha256": _json_sha(contract),
        "gate_e_evaluated": False, "reliability_evaluated": False,
    }
    write_json(shard_root / COMPLETION_NAME, marker)
    return marker


def aggregate(manifest: Path, output_root: Path):
    manifest_rows = adapter.read_manifest(manifest)
    assignments, plan_report = read_plan(output_root, manifest_rows)
    contract = contract_record(); contract_sha = _json_sha(contract)
    all_rows = []
    sample_counts = Counter(); token_counts = Counter(); status_counts = Counter()
    for shard_id in range(SHARD_COUNT):
        shard_root = output_root / "shards" / f"shard_{shard_id:02d}"
        marker_path = shard_root / COMPLETION_NAME
        if not marker_path.is_file():
            raise ContractError(f"completion marker absent for shard {shard_id}")
        marker = json.loads(marker_path.read_text())
        expected_count = sum(int(row["shard_id"]) == shard_id for row in assignments)
        if marker.get("shard_complete") is not True or marker.get("shard_id") != shard_id or marker.get("valid_utterances") != expected_count:
            raise ContractError(f"invalid completion marker for shard {shard_id}")
        if marker.get("shard_manifest_sha256") != plan_report["shard_manifest_sha256"] or marker.get("contract_sha256") != contract_sha or marker.get("contract") != contract:
            raise ContractError(f"schema/provenance/resource contract mismatch for shard {shard_id}")
        for assignment in (row for row in assignments if int(row["shard_id"]) == shard_id):
            path = _unit_path(shard_root, assignment["sample_id"])
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != tuple(adapter.FIELDS):
                    raise ContractError("per-utterance schema mismatch during aggregation")
                rows = list(reader)
            if not rows or any(row["sample_id"] != assignment["sample_id"] for row in rows):
                raise ContractError("per-utterance sample identity mismatch during aggregation")
            sample_counts[assignment["sample_id"]] += 1
            for row in rows:
                identity = (row["sample_id"], row["word_index"])
                token_counts[identity] += 1
                status_counts[f"alignment_status:{row.get('alignment_status','')}"] += 1
                status_counts[f"feature_status:{row.get('feature_status','')}"] += 1
            all_rows.extend(rows)
    expected_ids = {row["sample_id"] for row in manifest_rows}
    represented = set(sample_counts)
    duplicate_samples = sum(count - 1 for count in sample_counts.values() if count > 1)
    duplicate_tokens = sum(count - 1 for count in token_counts.values() if count > 1)
    missing = len(expected_ids - represented)
    unexpected = len(represented - expected_ids)
    if len(represented) != EXPECTED_UTTERANCES or duplicate_samples or duplicate_tokens or missing or unexpected:
        raise ContractError("aggregate sample/token identity validation failed")
    aggregate_path = output_root / "acoustics_raw" / AGGREGATE_NAME
    adapter.atomic_csv(aggregate_path, all_rows)
    qa = {
        "qa_version": "esd_stage2b_aggregate_qa_v1", "qa_pass": True,
        "all_32_shards_complete": True, "expected_utterances": EXPECTED_UTTERANCES,
        "represented_utterances": len(represented), "word_tokens": len(all_rows),
        "missing_sample_assignments": missing, "unexpected_sample_assignments": unexpected,
        "duplicate_sample_ids": duplicate_samples, "duplicate_token_identities": duplicate_tokens,
        "identical_schema_provenance_resource_hashes": True,
        "contract_sha256": contract_sha,
        "resource_hashes": {"cmudict_sha256": adapter.CMUDICT_SHA256, "alignment_checkpoint_sha256": ALIGN_CHECKPOINT_SHA256},
        "shard_manifest_sha256": plan_report["shard_manifest_sha256"],
        "aggregate_private_csv_sha256": sha256(aggregate_path),
        "status_counts": dict(sorted(status_counts.items())),
        "gate_e_evaluated": False, "reliability_evaluated": False,
    }
    write_json(output_root / QA_NAME, qa)
    return qa


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "shard", "aggregate"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--output-root", required=True, type=Path)
        if name == "shard":
            command.add_argument("--cmudict", required=True, type=Path)
            command.add_argument("--model-dir", required=True, type=Path)
            command.add_argument("--shard-id", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "plan": create_plan(args.manifest, args.output_root)
    elif args.command == "shard": run_shard(args.manifest, args.output_root, args.cmudict, args.model_dir, args.shard_id)
    else: aggregate(args.manifest, args.output_root)
    return 0


if __name__ == "__main__": raise SystemExit(main())
