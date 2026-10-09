#!/usr/bin/env python3
"""Read-only technical review of existing Stage-2A smoke CSV outputs.

This module does not import or invoke alignment, acoustics, reliability, Gate E,
or audio libraries. Its SAFE outputs contain only schemas and aggregate counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from esd_contract import ContractError, write_json
from esd_stage2_adapter import FIELDS

REVIEW_VERSION = "stage2a_smoke_technical_review_v1"
SOURCE_JOB_ID = 1752180
NUMERIC_FIELDS = (
    "word_index", "lexical_occurrence_index", "alignment_confidence",
    "word_start_sec", "word_end_sec", "word_duration_sec", "log_word_duration",
    "f0_mean_hz", "f0_p10_hz", "f0_p90_hz", "f0_robust_range_hz",
    "energy_db", "authoritative_syllable_count",
)
FLOAT_FIELDS = set(NUMERIC_FIELDS) - {"word_index", "lexical_occurrence_index", "authoritative_syllable_count"}
F0_FIELDS = ("f0_mean_hz", "f0_p10_hz", "f0_p90_hz", "f0_robust_range_hz")
PRIVATE_VALUE_FIELDS = {"source_word", "normalized_word"}
# esd_stage2_adapter.valid_unit freezes one acoustic event per sample_id +
# word_index. Python's float CSV serialization uses a shortest round-trip
# representation, so this tolerance covers binary subtraction roundoff only.
DUPLICATE_KEY_COLUMNS = ("sample_id", "word_index")
DURATION_IDENTITY_ABS_TOLERANCE_SEC = 1e-12


def _load_json(path: Path):
    if not path.is_file():
        raise ContractError(f"required input absent: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid required JSON: {path.name}") from exc


def _number(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return "invalid"


def _feature_summary(rows, field, positive=False):
    values = [_number(row.get(field)) for row in rows]
    finite = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    missing = sum(v is None for v in values)
    nonfinite = sum(v == "invalid" or isinstance(v, float) and not math.isfinite(v) for v in values)
    result = {"available_count": len(finite), "missing_count": missing, "nonfinite_count": nonfinite}
    if positive:
        result["nonpositive_count"] = sum(v <= 0 for v in finite)
    if finite:
        ordered = sorted(finite)
        result["finite_min"] = ordered[0]
        result["finite_median"] = ordered[len(ordered) // 2]
        result["finite_max"] = ordered[-1]
    return result


def review(smoke_root: Path, output: Path | None = None, human_template: Path | None = None):
    smoke_root = smoke_root.resolve()
    runtime = _load_json(smoke_root / "runtime_verification_SAFE.json")
    adapter = _load_json(smoke_root / "extraction_PRIVATE" / "adapter_run_SAFE.json")
    qa = _load_json(smoke_root / "stage2_smoke_QA_SAFE.json")
    files = sorted((smoke_root / "extraction_PRIVATE" / "by_utterance").glob("*.csv"))
    checks = {
        "runtime_all_checks_passed": runtime.get("all_checks_passed") is True,
        "adapter_mode_smoke": adapter.get("mode") == "smoke",
        "gate_e_not_evaluated": adapter.get("gate_e_evaluated") is False,
        "adapter_utterance_count_exactly_10": adapter.get("utterance_count") == 10,
        "per_utterance_csv_count_exactly_10": len(files) == 10,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ContractError("Stage-2A review input contract failed: " + ", ".join(failed))

    rows = []
    schemas = []
    row_counts = []
    per_file_rows = []
    nonmonotonic = 0
    for index, path in enumerate(files, 1):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                schema = tuple(reader.fieldnames or ())
                current = list(reader)
        except (OSError, csv.Error) as exc:
            raise ContractError(f"could not read PRIVATE CSV number {index}") from exc
        schemas.append(schema)
        row_counts.append(len(current))
        per_file_rows.append({"file_ordinal": index, "row_count": len(current)})
        previous = None
        for row in current:
            start = _number(row.get("word_start_sec"))
            if isinstance(start, float) and math.isfinite(start):
                if previous is not None and start < previous:
                    nonmonotonic += 1
                previous = start
        rows.extend(current)

    expected_schema = tuple(FIELDS)
    distinct_schemas = sorted({schema for schema in schemas})
    schema_mismatches = sum(schema != expected_schema for schema in schemas)
    missingness = {}
    for field in expected_schema:
        count = sum(str(row.get(field, "")).strip() == "" for row in rows)
        missingness[field] = {"missing_count": count, "missing_percent": (100.0 * count / len(rows)) if rows else 0.0}

    dtype_summary = {}
    nonfinite_by_column = {}
    for field in expected_schema:
        if field not in NUMERIC_FIELDS:
            dtype_summary[field] = {"contract_type": "string", "nonmissing_count": len(rows) - missingness[field]["missing_count"]}
            continue
        values = [_number(row.get(field)) for row in rows]
        invalid = sum(v == "invalid" for v in values)
        nonfinite = sum(isinstance(v, float) and not math.isfinite(v) for v in values)
        finite = sum(isinstance(v, float) and math.isfinite(v) for v in values)
        dtype_summary[field] = {"contract_type": "float" if field in FLOAT_FIELDS else "integer", "finite_numeric_count": finite, "invalid_numeric_count": invalid, "missing_count": sum(v is None for v in values)}
        nonfinite_by_column[field] = nonfinite + invalid

    timing = Counter({name: 0 for name in (
        "missing_word_start", "missing_word_end", "negative_word_start",
        "end_not_greater_than_start", "nonpositive_word_duration",
        "nonfinite_word_start", "nonfinite_word_end", "nonfinite_word_duration",
        "duration_identity_mismatch_count", "non_monotonic_starts",
    )})
    timing["non_monotonic_starts"] = nonmonotonic
    duplicate_key_frequencies = Counter()
    for row in rows:
        start, end, duration = (_number(row.get(name)) for name in ("word_start_sec", "word_end_sec", "word_duration_sec"))
        if start is None: timing["missing_word_start"] += 1
        if end is None: timing["missing_word_end"] += 1
        if start == "invalid" or isinstance(start, float) and not math.isfinite(start): timing["nonfinite_word_start"] += 1
        if end == "invalid" or isinstance(end, float) and not math.isfinite(end): timing["nonfinite_word_end"] += 1
        if duration == "invalid" or isinstance(duration, float) and not math.isfinite(duration): timing["nonfinite_word_duration"] += 1
        if isinstance(start, float) and math.isfinite(start) and start < 0: timing["negative_word_start"] += 1
        if isinstance(start, float) and math.isfinite(start) and isinstance(end, float) and math.isfinite(end):
            if end <= start: timing["end_not_greater_than_start"] += 1
            if isinstance(duration, float) and math.isfinite(duration) and abs((end-start)-duration) > DURATION_IDENTITY_ABS_TOLERANCE_SEC:
                timing["duration_identity_mismatch_count"] += 1
        if isinstance(duration, float) and math.isfinite(duration) and duration <= 0:
            timing["nonpositive_word_duration"] += 1
        primary = tuple(row.get(column, "") for column in DUPLICATE_KEY_COLUMNS)
        duplicate_key_frequencies[primary] += 1

    duplicate_key_count = sum(count > 1 for count in duplicate_key_frequencies.values())
    duplicate_row_count = sum(count - 1 for count in duplicate_key_frequencies.values() if count > 1)
    duplicate_checks = {
        "duplicate_key_columns": list(DUPLICATE_KEY_COLUMNS),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_key_count": duplicate_key_count,
        "duplicate_rows_present": duplicate_row_count > 0,
    }

    f0 = {field: _feature_summary(rows, field, positive=True) for field in F0_FIELDS}
    robust = {
        "zero_range_count": sum(v == 0 for v in (_number(r.get("f0_robust_range_hz")) for r in rows) if isinstance(v, float) and math.isfinite(v)),
        "negative_range_count": sum(v < 0 for v in (_number(r.get("f0_robust_range_hz")) for r in rows) if isinstance(v, float) and math.isfinite(v)),
        "p10_greater_than_p90_count": sum(1 for r in rows if isinstance((lo := _number(r.get("f0_p10_hz"))), float) and math.isfinite(lo) and isinstance((hi := _number(r.get("f0_p90_hz"))), float) and math.isfinite(hi) and lo > hi),
        "summary": _feature_summary(rows, "f0_robust_range_hz", positive=False),
    }
    syllables = _feature_summary(rows, "authoritative_syllable_count", positive=True)
    extraction = {
        "feature_status_counts": dict(sorted(Counter(r.get("feature_status", "") for r in rows).items())),
        "alignment_status_counts": dict(sorted(Counter(r.get("alignment_status", "") for r in rows).items())),
        "feature_error_present_count": sum(bool(r.get("feature_error_code", "").strip()) for r in rows),
        "alignment_error_present_count": sum(bool(r.get("alignment_error_code", "").strip()) for r in rows),
    }
    adapter_statuses = adapter.get("statuses", [])
    qa_metrics = qa.get("metrics", qa)
    consistency = {
        "adapter_status_record_count": len(adapter_statuses) if isinstance(adapter_statuses, list) else None,
        "adapter_complete_or_skipped_valid_count": sum(s.get("status") in {"complete", "skipped_valid"} for s in adapter_statuses) if isinstance(adapter_statuses, list) else None,
        "csv_file_count": len(files), "csv_word_token_count": len(rows),
        "qa_utterance_outputs": qa_metrics.get("utterance_outputs"), "qa_word_tokens": qa_metrics.get("word_tokens"),
        "adapter_vs_csv_file_count_match": isinstance(adapter_statuses, list) and len(adapter_statuses) == len(files),
        "qa_vs_csv_utterance_count_match": qa_metrics.get("utterance_outputs") == len(files),
        "qa_vs_csv_word_token_count_match": qa_metrics.get("word_tokens") == len(rows),
    }
    hard = []
    hard_counts = {
        "schema_mismatch_files": schema_mismatches,
        "invalid_start_end": sum(timing[name] for name in (
            "missing_word_start", "missing_word_end", "negative_word_start",
            "end_not_greater_than_start", "nonpositive_word_duration",
            "nonfinite_word_start", "nonfinite_word_end", "nonfinite_word_duration",
            "duration_identity_mismatch_count",
        )),
        "non_monotonic_starts": timing["non_monotonic_starts"],
        "duplicate_token_evidence": duplicate_row_count,
        "nonfinite_or_invalid_numeric_values": sum(nonfinite_by_column.values()),
        "invalid_f0_range_order": robust["negative_range_count"] + robust["p10_greater_than_p90_count"],
        "nonpositive_syllable_counts": syllables["nonpositive_count"],
    }
    hard = [{"classification": "HARD_STRUCTURAL_ERROR", "check": key, "count": value} for key, value in hard_counts.items() if value]
    warnings = []
    if f0["f0_mean_hz"]["missing_count"]:
        warnings.append({"classification": "EXPECTED_MISSINGNESS", "check": "f0_mean_missing", "count": f0["f0_mean_hz"]["missing_count"], "automatic_defect": False})
    if syllables["missing_count"]:
        warnings.append({"classification": "DESCRIPTIVE_WARNING", "check": "syllable_count_missing", "count": syllables["missing_count"], "automatic_defect": False})
    if not warnings:
        warnings.append({"classification": "NO_ISSUE", "check": "no_descriptive_warning_observed", "count": 0})

    artifact = {
        "review_version": REVIEW_VERSION, "source_smoke_root": str(smoke_root), "source_job_id": SOURCE_JOB_ID,
        "audio_reprocessed": False, "alignment_reprocessed": False, "acoustics_reprocessed": False,
        "gate_e_evaluated": False, "reliability_evaluated": False,
        "input_contract_checks": checks,
        "schema_checks": {"expected_columns": list(expected_schema), "file_count": len(files), "per_file_row_counts": per_file_rows, "exact_schema_identity": schema_mismatches == 0, "schema_mismatch_file_count": schema_mismatches, "distinct_schema_count": len(distinct_schemas), "distinct_schemas": [list(x) for x in distinct_schemas], "dtype_summary": dtype_summary},
        "timing_checks": {**dict(sorted(timing.items())), "duration_identity_absolute_tolerance_sec": DURATION_IDENTITY_ABS_TOLERANCE_SEC, "duration_identity_tolerance_provenance": "IEEE-754 subtraction after adapter shortest-round-trip float CSV serialization"},
        "duplicate_checks": duplicate_checks,
        "feature_missingness": {"per_column": missingness, "nonfinite_numeric_by_column": nonfinite_by_column, "f0": f0, "f0_robust_range": robust, "energy": _feature_summary(rows, "energy_db"), "raw_duration": _feature_summary(rows, "word_duration_sec", positive=True), "log_duration": _feature_summary(rows, "log_word_duration")},
        "syllable_checks": syllables, "extraction_failure_evidence": extraction,
        "cross_file_consistency": consistency, "hard_structural_errors": hard, "descriptive_warnings": warnings,
        "machine_review_complete": True, "human_review_status": "pending", "confirmed_technical_defect": None,
        "technical_defect_threshold_prespecified": False, "stage2b_authorized": False,
    }
    output = output or smoke_root / "stage2_smoke_technical_review_SAFE.json"
    write_json(output, artifact)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    template = {
        "reviewer": "", "review_timestamp": "", "source_machine_review_sha256": digest,
        "reviewed_smoke_root": str(smoke_root), "confirmed_technical_defect": None,
        "decision_notes": "", "stage2b_authorized": False,
    }
    human_template = human_template or smoke_root / "stage2_smoke_human_review_SAFE.json"
    write_json(human_template, template)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--human-template", type=Path)
    args = parser.parse_args(argv)
    review(args.smoke_root, args.output, args.human_template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
