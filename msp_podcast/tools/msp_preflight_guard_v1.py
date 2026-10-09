#!/usr/bin/env python3
"""Fail-closed technical preflight for MSP acoustic processing (schema-chain/implementation v3).

This guard makes no scientific choices. It verifies the actual frozen SAFE
artifact schemas and their cross-artifact SHA chain, plus an explicitly
human-approved frozen model lock.
"""
from __future__ import annotations
import argparse, json, re

HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_KEY_PARTS = ("emotion", "emoval", "emoact", "emodom", "emoclass")
SAFE_AUDIT_KEYS = frozenset({"emotion_fields_used", "emotion_values_accessed"})

class PreflightError(ValueError):
    pass

def _sha_ok(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None

def _no_forbidden_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl not in SAFE_AUDIT_KEYS and any(x in kl for x in FORBIDDEN_KEY_PARTS):
                return False
            if not _no_forbidden_keys(v):
                return False
    elif isinstance(obj, list):
        return all(_no_forbidden_keys(x) for x in obj)
    return True

def validate_model_lock(lock):
    required = {
        "status", "human_approved", "frozen_before_msp_acoustic_extraction",
        "frozen_before_msp_reliability_results", "primary_summary",
        "nuisance_model", "primary_reliability_scalar", "bootstrap",
        "volume_domination_rule", "transcript_alignment_content_qa_rule",
        "git_commit", "git_tag",
    }
    if not isinstance(lock, dict) or not required <= set(lock):
        raise PreflightError("model lock missing required fields")
    if lock["status"] != "frozen" or lock["human_approved"] is not True:
        raise PreflightError("model lock is not human-approved frozen")
    if lock["frozen_before_msp_acoustic_extraction"] is not True or lock["frozen_before_msp_reliability_results"] is not True:
        raise PreflightError("model lock timing declaration missing")
    for key in ("primary_summary", "primary_reliability_scalar", "volume_domination_rule", "transcript_alignment_content_qa_rule", "git_commit", "git_tag"):
        if not isinstance(lock[key], str) or not lock[key].strip():
            raise PreflightError(f"model lock field {key} is empty")
    if not isinstance(lock["nuisance_model"], dict) or not lock["nuisance_model"]:
        raise PreflightError("nuisance model must be explicit")
    boot = lock["bootstrap"]
    if not isinstance(boot, dict) or not {"replicates", "seed", "failure_rule", "ci_rule"} <= set(boot):
        raise PreflightError("bootstrap lock incomplete")
    if not isinstance(boot["replicates"], int) or boot["replicates"] <= 0 or not isinstance(boot["seed"], int):
        raise PreflightError("bootstrap count/seed invalid")
    if not str(boot["failure_rule"]).strip() or not str(boot["ci_rule"]).strip():
        raise PreflightError("bootstrap rules empty")
    if lock.get("implementation_status") != "frozen":
        raise PreflightError("model implementation is not frozen")
    impl = lock.get("implementation")
    required_impl = {"files_sha256", "synthetic_test_total", "synthetic_test_passed",
                     "real_msp_acoustic_values_accessed", "real_msp_reliability_values_accessed",
                     "implementation_commit", "implementation_tag"}
    if not isinstance(impl, dict) or not required_impl <= set(impl):
        raise PreflightError("implementation lock incomplete")
    required_files = {
        "msp_primary_residualization_v2.py", "msp_profile_reml_v1.py",
        "msp_lexical_secondary_v1.py", "msp_reliability_v2.py",
        "msp_acoustic_adapter_v1.py",
        "test_msp_primary_residualization_v2.py", "test_msp_profile_reml_v1.py",
        "test_msp_lexical_secondary_v1.py", "test_msp_reliability_v2.py",
        "test_msp_acoustic_adapter_v1.py",
        "esd_frozen_acoustic_adapter_sha256", "seamless_frozen_pipeline_sha256",
    }
    hashes = impl.get("files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != required_files or not all(_sha_ok(v) for v in hashes.values()):
        raise PreflightError("implementation hash set mismatch")
    if impl.get("synthetic_test_total") != 43 or impl.get("synthetic_test_passed") != 43:
        raise PreflightError("implementation synthetic tests incomplete")
    if impl.get("real_msp_acoustic_values_accessed") is not False or impl.get("real_msp_reliability_values_accessed") is not False:
        raise PreflightError("implementation was not frozen prospectively")
    if not str(impl.get("implementation_commit", "")).strip() or not str(impl.get("implementation_tag", "")).strip():
        raise PreflightError("implementation provenance missing")
    if not _no_forbidden_keys(lock):
        raise PreflightError("forbidden emotion key in model lock")
    return True

def evaluate_preflight(structural, alignment, final_scope, support_report, model_lock):
    checks = {}
    checks["structural_projection"] = bool(
        isinstance(structural, dict)
        and structural.get("audit") == "msp_structural_projection_v1"
        and structural.get("projected_rows", 0) > 0
        and structural.get("emotion_fields_used") is False
        and structural.get("acoustic_values_accessed") is False
        and structural.get("reliability_values_accessed") is False
        and _sha_ok(structural.get("output_sha256"))
    )
    checks["transcript_alignment_qa"] = bool(
        isinstance(alignment, dict)
        and alignment.get("audit") == "msp_textgrid_content_qa_v1"
        and alignment.get("filename_coverage_exact") is True
        and alignment.get("files_passing_content_qa", 0) > 0
        and _sha_ok(alignment.get("eligible_filename_list_sha256"))
        and _sha_ok(alignment.get("private_token_manifest_sha256"))
        and alignment.get("transcript_text_accessed") is False
        and alignment.get("phone_marks_parsed_or_used") is False
        and alignment.get("word_marks_emitted") is False
        and alignment.get("emotion_fields_used") is False
        and alignment.get("audio_accessed") is False
        and alignment.get("acoustic_values_accessed") is False
        and alignment.get("reliability_values_accessed") is False
        and alignment.get("acoustic_processing_authorized") is False
    )
    checks["final_scope"] = bool(
        isinstance(final_scope, dict)
        and final_scope.get("audit") == "msp_final_scope_and_split_manifest_v1"
        and final_scope.get("final_scope_selected") is True
        and 1 <= final_scope.get("selected_speakers", 0) <= 100
        and final_scope.get("speaker_cap") == 100
        and final_scope.get("private_manifest_rows", 0) > 0
        and final_scope.get("primary_split_assigned") is True
        and final_scope.get("alternative_splits_assigned") == 25
        and final_scope.get("emotion_fields_used") is False
        and final_scope.get("audio_accessed") is False
        and final_scope.get("acoustic_values_accessed") is False
        and final_scope.get("reliability_values_accessed") is False
        and _sha_ok(final_scope.get("private_manifest_sha256"))
        and _sha_ok(final_scope.get("structural_sha256"))
        and _sha_ok(final_scope.get("eligible_list_sha256"))
    )
    checks["structural_hash_chain"] = bool(
        checks["structural_projection"] and checks["final_scope"]
        and final_scope.get("structural_sha256") == structural.get("output_sha256")
    )
    checks["eligible_list_hash_chain"] = bool(
        checks["transcript_alignment_qa"] and checks["final_scope"]
        and final_scope.get("eligible_list_sha256") == alignment.get("eligible_filename_list_sha256")
    )
    support_keys = ("qualified_eligible_utterances", "selected_eligible_utterances", "qualified_recording_units", "selected_recording_units", "selected_distinct_unordered_alt_partitions")
    checks["structural_support_report"] = bool(
        isinstance(support_report, dict)
        and support_report.get("audit") == "msp_structural_support_report_v1"
        and support_report.get("status") == "PASS"
        and support_report.get("speaker_cap") == 100
        and support_report.get("selected_speakers") == final_scope.get("selected_speakers")
        and support_report.get("qualified_speakers", 0) >= support_report.get("selected_speakers", 0) >= 1
        and support_report.get("reselection_performed") is False
        and support_report.get("selection_change_authorized") is False
        and support_report.get("significance_tests_run") is False
        and support_report.get("speaker_ids_emitted") is False
        and support_report.get("emotion_fields_used") is False
        and support_report.get("audio_accessed") is False
        and support_report.get("acoustic_values_accessed") is False
        and support_report.get("reliability_values_accessed") is False
        and support_report.get("clinical_outcomes_accessed") is False
        and all(isinstance(support_report.get(k), dict) and support_report[k].get("n",0)>0 for k in support_keys)
        and _sha_ok(support_report.get("structural_sha256"))
        and _sha_ok(support_report.get("eligible_list_sha256"))
        and _sha_ok(support_report.get("final_private_manifest_sha256"))
    )
    checks["support_hash_chain"] = bool(
        checks["structural_support_report"] and checks["final_scope"]
        and support_report.get("structural_sha256") == final_scope.get("structural_sha256")
        and support_report.get("eligible_list_sha256") == final_scope.get("eligible_list_sha256")
        and support_report.get("final_private_manifest_sha256") == final_scope.get("private_manifest_sha256")
    )
    try:
        checks["model_lock"] = validate_model_lock(model_lock)
    except PreflightError:
        checks["model_lock"] = False
    checks["forbidden_keys_absent"] = all(_no_forbidden_keys(x) for x in (structural, alignment, final_scope, support_report, model_lock))
    authorized = all(checks.values())
    return {
        "audit": "msp_acoustic_preflight_guard_v4",
        "checks": checks,
        "msp_audio_processing_authorized": authorized,
        "scientific_choices_made_by_guard": False,
        "clinical_outcomes_accessed": False,
        "reliability_values_accessed": False,
    }

def main():
    ap = argparse.ArgumentParser()
    for name in ("structural", "alignment", "final_scope", "support_report", "model_lock"):
        ap.add_argument(f"--{name.replace('_','-')}", required=True)
    args = ap.parse_args()
    vals = {}
    for name in ("structural", "alignment", "final_scope", "support_report", "model_lock"):
        with open(getattr(args, name), encoding="utf-8") as f:
            vals[name] = json.load(f)
    out = evaluate_preflight(vals["structural"], vals["alignment"], vals["final_scope"], vals["support_report"], vals["model_lock"])
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["msp_audio_processing_authorized"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
