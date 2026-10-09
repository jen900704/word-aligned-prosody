#!/usr/bin/env python3
"""Computational-only parallel orchestration for the frozen Stage-2C estimator.

This module owns no scientific constants or fitting logic.  It partitions the
frozen bootstrap plan, records each attempt, and deterministically reconstructs
the objects produced by :mod:`esd_reml_reliability`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import esd_reml_reliability as frozen
from esd_contract import COMPARABLE_FEATURES, ContractError, reject_forbidden_path, write_json

ORCHESTRATION_LABEL = "computational_parallelization_only"
ARTIFACT_VERSION = 1


def _require_feature(feature):
    if feature not in COMPARABLE_FEATURES:
        raise ContractError(f"feature is not frozen-comparable: {feature!r}")


def _plan():
    plan = frozen.make_bootstrap_plan()
    digest = frozen.validate_bootstrap_plan(plan)
    if digest != frozen.BOOTSTRAP_PLAN_SHA256:
        raise ContractError("frozen bootstrap plan identity mismatch")
    return plan, digest


def bootstrap_chunk(rows, feature, start, stop, fit_fn=frozen.fit_condition_reml):
    """Run the frozen bootstrap attempts in the half-open interval [start, stop)."""
    _require_feature(feature)
    if isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int):
        raise ContractError("bootstrap chunk bounds must be integers")
    if not 0 <= start < stop <= frozen.BOOTSTRAP_REPLICATES:
        raise ContractError("bootstrap chunk must be within [0, 1000) and nonempty")
    plan, digest = _plan()
    attempts = []
    for replicate in range(start, stop):
        try:
            sampled, _ = frozen.bootstrap_sample(rows, plan[replicate])
            value = float(frozen.five_condition_summary(sampled, feature, fit_fn)["equal_condition_mean"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ContractError("invalid bootstrap estimate")
            attempts.append({"replicate": replicate, "status": "success", "equal_condition_mean": value})
        except Exception as exc:
            attempts.append({"replicate": replicate, "status": "failed", **frozen.failure_record(exc)})
    return {
        "artifact": "stage2c_bootstrap_chunk", "artifact_version": ARTIFACT_VERSION,
        "orchestration": ORCHESTRATION_LABEL, "feature": feature,
        "plan_sha256": digest, "start": start, "stop": stop, "attempts": attempts,
    }


def core_feature(rows, feature, fit_fn=frozen.fit_condition_reml):
    """Compute the non-bootstrap portion of frozen ``analyze_feature``."""
    _require_feature(feature)
    try:
        primary = frozen.five_condition_summary(rows, feature, fit_fn)
    except Exception as exc:
        result = {"feature": feature, "status": "technical_review_no_decision",
                  **frozen.failure_record(exc), "primary": None}
        return {"artifact": "stage2c_feature_core", "artifact_version": ARTIFACT_VERSION,
                "orchestration": ORCHESTRATION_LABEL, "feature": feature, "result": result}
    result = {
        "feature": feature, "support": frozen.feature_support(rows, feature), "primary": primary,
        "leave_one_speaker_out": frozen.leave_one_speaker_out(rows, feature, fit_fn),
        "lexical_composition_sensitivity": frozen.lexical_composition_sensitivity(
            rows, feature, fit_fn, primary["equal_condition_mean"]),
        "cross_condition": {"status": "omitted_secondary", "part_of_gate_e": False},
    }
    return {"artifact": "stage2c_feature_core", "artifact_version": ARTIFACT_VERSION,
            "orchestration": ORCHESTRATION_LABEL, "feature": feature, "result": result}


def _validate_artifact(record, kind):
    if not isinstance(record, dict) or record.get("artifact") != kind:
        raise ContractError(f"expected {kind} artifact")
    if record.get("artifact_version") != ARTIFACT_VERSION or record.get("orchestration") != ORCHESTRATION_LABEL:
        raise ContractError("orchestration artifact identity mismatch")


def merge_bootstrap_chunks(feature, chunks):
    """Validate exact coverage and recreate frozen ``speaker_cluster_bootstrap`` output."""
    import numpy as np

    _require_feature(feature)
    _, digest = _plan()
    by_replicate = {}
    for chunk in chunks:
        _validate_artifact(chunk, "stage2c_bootstrap_chunk")
        if chunk.get("feature") != feature:
            raise ContractError("bootstrap chunk feature mismatch")
        if chunk.get("plan_sha256") != digest:
            raise ContractError("bootstrap chunk plan SHA mismatch")
        start, stop, attempts = chunk.get("start"), chunk.get("stop"), chunk.get("attempts")
        if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int)
                or not isinstance(stop, int) or not 0 <= start < stop <= frozen.BOOTSTRAP_REPLICATES):
            raise ContractError("invalid bootstrap chunk bounds")
        if not isinstance(attempts, list) or len(attempts) != stop - start:
            raise ContractError("bootstrap chunk attempt count mismatch")
        if [x.get("replicate") if isinstance(x, dict) else None for x in attempts] != list(range(start, stop)):
            raise ContractError("bootstrap chunk indices do not match its declared interval")
        for attempt in attempts:
            replicate = attempt["replicate"]
            if replicate in by_replicate:
                raise ContractError(f"duplicate bootstrap replicate: {replicate}")
            status = attempt.get("status")
            if status == "success":
                if set(attempt) != {"replicate", "status", "equal_condition_mean"}:
                    raise ContractError("malformed successful bootstrap attempt")
                value = attempt["equal_condition_mean"]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                    raise ContractError("invalid successful bootstrap estimate")
            elif status == "failed":
                allowed = {"replicate", "status", "error_type", "fit_diagnostic"}
                if set(attempt) - allowed or not isinstance(attempt.get("error_type"), str):
                    raise ContractError("malformed failed bootstrap attempt")
            else:
                raise ContractError("bootstrap attempt has invalid status")
            by_replicate[replicate] = attempt
    expected = set(range(frozen.BOOTSTRAP_REPLICATES))
    if set(by_replicate) != expected:
        missing = sorted(expected - set(by_replicate))
        extra = sorted(set(by_replicate) - expected)
        raise ContractError(f"bootstrap coverage mismatch: missing={missing[:10]!r}, extra={extra[:10]!r}")
    ordered = [by_replicate[i] for i in range(frozen.BOOTSTRAP_REPLICATES)]
    estimates = [float(x["equal_condition_mean"]) for x in ordered if x["status"] == "success"]
    failures = [{k: v for k, v in x.items() if k != "status"} for x in ordered if x["status"] == "failed"]
    success = len(estimates)
    ci = [float(x) for x in np.quantile(estimates, [.025, .975], method="linear")] if success >= frozen.BOOTSTRAP_REVIEW_MIN_SUCCESS else None
    return {
        "type": "speaker_cluster_nonparametric", "rng": "numpy.random.Generator(numpy.random.PCG64(13))",
        "seed": frozen.BOOTSTRAP_SEED, "plan_sha256": digest,
        "requested_replicates": frozen.BOOTSTRAP_REPLICATES,
        "successful_replicates": success, "failed_replicates": len(failures),
        "failed_details": failures, "ci_95": ci,
        "technical_review_required": success < frozen.BOOTSTRAP_REVIEW_MIN_SUCCESS,
        "minimum_success_for_no_review": frozen.BOOTSTRAP_REVIEW_MIN_SUCCESS,
        "replacement_draws_generated": False,
    }


def merge_feature(feature, core, chunks):
    """Merge one feature into the semantic structure of frozen ``analyze_feature``."""
    _require_feature(feature)
    _validate_artifact(core, "stage2c_feature_core")
    if core.get("feature") != feature or not isinstance(core.get("result"), dict):
        raise ContractError("core feature mismatch or malformed result")
    result = core["result"]
    if result.get("feature") != feature:
        raise ContractError("core result feature mismatch")
    # Coverage and provenance are mandatory even when the frozen primary is
    # non-estimable.  The reconstructed bootstrap is then deliberately omitted
    # to preserve frozen analyze_feature's early-return semantic shape.
    bootstrap = merge_bootstrap_chunks(feature, chunks)
    if result.get("primary") is None:
        return dict(result)
    required = {"support", "primary", "leave_one_speaker_out", "lexical_composition_sensitivity", "cross_condition"}
    if not required <= set(result):
        raise ContractError("incomplete core result")
    gate = frozen.evaluate_feature_gate(feature, result["support"], result["primary"], bootstrap,
                                        result["leave_one_speaker_out"])
    return {"feature": feature, "status": gate["status"], "support": result["support"],
            "primary": result["primary"], "bootstrap": bootstrap,
            "leave_one_speaker_out": result["leave_one_speaker_out"],
            "lexical_composition_sensitivity": result["lexical_composition_sensitivity"],
            "cross_condition": result["cross_condition"], "gate_e": gate}


def merge_all(feature_reports, duration_model):
    """Assemble the frozen main-output semantics plus labeled orchestration metadata."""
    reports = list(feature_reports)
    names = [x.get("feature") if isinstance(x, dict) else None for x in reports]
    if len(reports) != len(COMPARABLE_FEATURES) or set(names) != set(COMPARABLE_FEATURES) or len(names) != len(set(names)):
        raise ContractError("final merge requires each frozen comparable feature exactly once")
    by = {x["feature"]: x for x in reports}
    ordered = [by[x] for x in COMPARABLE_FEATURES]
    gates = [r.get("gate_e", {"feature": r["feature"], "gate_e_evaluable": False}) for r in ordered]
    return {"analysis": "frozen_stage2c_esd_wordlocal_reml", "duration_model": duration_model,
            "features": ordered, "gate_e": frozen.overall_gate_e(gates),
            "scientific_claims_authorized": False,
            "orchestration_metadata": {"purpose": ORCHESTRATION_LABEL,
                "bootstrap_partitioning": "deterministic_half_open_replicate_ranges",
                "bootstrap_plan_sha256": frozen.BOOTSTRAP_PLAN_SHA256,
                "scientific_semantics_changed": False}}


def _read_json(path):
    reject_forbidden_path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read orchestration artifact {path}: {type(exc).__name__}") from exc


def _real_rows(args):
    return frozen.load_real_input(args.stage1_manifest, args.stage2b_aggregate,
                                  args.stage2b_closure, args.expected_closure_sha256)


def _output(args):
    root = args.rescue_output_root.resolve()
    output = args.output.resolve()
    reject_forbidden_path(root); reject_forbidden_path(output)
    if root.name == "stage2c_real_v1" or "stage2c_real_v1" in output.parts:
        raise ContractError("parallel rescue must never target stage2c_real_v1")
    if output == root or root not in output.parents:
        raise ContractError("output must be a file below --rescue-output-root")
    return output


def _add_real(parser):
    parser.add_argument("--stage1-manifest", required=True, type=Path)
    parser.add_argument("--stage2b-aggregate", required=True, type=Path)
    parser.add_argument("--stage2b-closure", required=True, type=Path)
    parser.add_argument("--expected-closure-sha256", required=True)


def _add_output(parser):
    parser.add_argument("--rescue-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    chunk = sub.add_parser("bootstrap-chunk"); _add_real(chunk); _add_output(chunk)
    chunk.add_argument("--feature", required=True, choices=COMPARABLE_FEATURES, type=str)
    chunk.add_argument("--start", required=True, type=int); chunk.add_argument("--stop", required=True, type=int)
    core = sub.add_parser("core"); _add_real(core); _add_output(core)
    core.add_argument("--feature", required=True, choices=COMPARABLE_FEATURES, type=str)
    one = sub.add_parser("merge-feature"); _add_real(one); _add_output(one)
    one.add_argument("--feature", required=True, choices=COMPARABLE_FEATURES, type=str)
    one.add_argument("--core", required=True, type=Path); one.add_argument("--chunks", required=True, nargs="+", type=Path)
    all_ = sub.add_parser("merge-all"); _add_real(all_); _add_output(all_)
    all_.add_argument("--features", required=True, nargs="+", type=Path)
    args = parser.parse_args(argv)
    output = _output(args)
    if args.command == "bootstrap-chunk":
        rows, _ = _real_rows(args); value = bootstrap_chunk(rows, args.feature, args.start, args.stop)
    elif args.command == "core":
        rows, _ = _real_rows(args); value = core_feature(rows, args.feature)
    elif args.command == "merge-feature":
        _real_rows(args)  # validate frozen real-input provenance before artifact merge
        value = merge_feature(args.feature, _read_json(args.core), [_read_json(x) for x in args.chunks])
    else:
        _, duration_model = _real_rows(args)
        value = merge_all([_read_json(x) for x in args.features], duration_model)
    write_json(output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
