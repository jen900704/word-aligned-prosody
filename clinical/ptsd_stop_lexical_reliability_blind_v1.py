#!/usr/bin/env python3
"""Blind PTSD-STOP participant-by-word crossed-REML repeatability.

Run one frozen feature per process. Scientific estimates and checkpoints remain
PRIVATE; stdout and technical JSON expose support, timing, and completion only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# The frozen launcher runs at most one serial refit queue per feature.  Keeping
# numerical kernels single-threaded makes the four-process runtime projection
# and peak-memory calculation interpretable.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

try:
    from .daic_aggregate_reliability_blind_v1 import (
        BOOTSTRAP_MIN_SUCCESS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        FEATURES,
        file_sha,
        percentile,
    )
    from .daic_lexical_reliability_blind_v1 import (
        MIN_CELL_TOKENS,
        PROGRESS_INTERVAL,
        VC_FORMULA,
        bootstrap_draws,
        bootstrap_frame,
        duration_fixed_effect_residual,
        fit_crossed_reml,
    )
    from .ptsd_stop_aggregate_reliability_blind_v1 import (
        EXPECTED_ANALYSIS_PARTICIPANTS,
        apply_identity_authority,
        load_tokens,
        reject,
        verify_closure,
    )
except ImportError:  # direct script execution
    from daic_aggregate_reliability_blind_v1 import (
        BOOTSTRAP_MIN_SUCCESS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        FEATURES,
        file_sha,
        percentile,
    )
    from daic_lexical_reliability_blind_v1 import (
        MIN_CELL_TOKENS,
        PROGRESS_INTERVAL,
        VC_FORMULA,
        bootstrap_draws,
        bootstrap_frame,
        duration_fixed_effect_residual,
        fit_crossed_reml,
    )
    from ptsd_stop_aggregate_reliability_blind_v1 import (
        EXPECTED_ANALYSIS_PARTICIPANTS,
        apply_identity_authority,
        load_tokens,
        reject,
        verify_closure,
    )

PROTOCOL = "clinical_regime_amendment_v2_plus_v3_1_plus_ptsd_stop_ingestion_v1_3"
PRECISION_PROTOCOL = "ptsd_stop_precision_within_person_amendment_v1"
PRECISION_MODES = {
    "full_precision_B1000": (1000, 950),
    "reduced_precision_B200": (200, 190),
    "point_only_compute_limited": (0, 0),
}
MIN_LEXICAL_PARTICIPANTS = 50


def peak_rss_mib() -> float:
    """Return process peak RSS without exposing any scientific value."""
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak / 1024.0 if os.name != "darwin" else peak / (1024.0 ** 2)
    except (ImportError, OSError, ValueError):
        return float("nan")


def load_compute_decision(path: Path) -> tuple[str, int, int, str]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("protocol") != PRECISION_PROTOCOL:
        raise ValueError("compute decision protocol mismatch")
    mode = str(decision.get("mode"))
    if mode not in PRECISION_MODES:
        raise ValueError("unknown compute precision mode")
    target, minimum = PRECISION_MODES[mode]
    if decision.get("bootstrap_target_per_feature") != target:
        raise ValueError("compute decision bootstrap target mismatch")
    if decision.get("bootstrap_minimum_success") != minimum:
        raise ValueError("compute decision bootstrap minimum mismatch")
    return mode, target, minimum, file_sha(path)


def prepare_feature(frame: pd.DataFrame, feature: str) -> tuple[pd.DataFrame, dict]:
    if feature not in FEATURES:
        raise ValueError(f"unknown feature: {feature}")
    work = frame[["participant_id", "normalized_word"]].copy()
    if feature == "syllable_adjusted_log_duration":
        work["y"] = duration_fixed_effect_residual(frame)
    else:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
    work = work.loc[np.isfinite(work["y"])].copy()
    work["participant_id"] = work["participant_id"].astype(str)
    work["normalized_word"] = work["normalized_word"].astype(str)
    counts = work.groupby(
        ["participant_id", "normalized_word"], sort=False
    ).size()
    eligible = set(counts.loc[counts >= MIN_CELL_TOKENS].index)
    mask = [
        pair in eligible
        for pair in zip(work["participant_id"], work["normalized_word"])
    ]
    work = work.loc[mask].reset_index(drop=True)
    support = {
        "participants": int(work["participant_id"].nunique()),
        "authoritative_participants": EXPECTED_ANALYSIS_PARTICIPANTS,
        "words": int(work["normalized_word"].nunique()),
        "matched_participant_word_cells": int(len(eligible)),
        "tokens": int(len(work)),
        "minimum_tokens_per_cell": MIN_CELL_TOKENS,
    }
    if support["participants"] < MIN_LEXICAL_PARTICIPANTS:
        raise ValueError(
            "fewer than 50 authoritative participants retain repeated lexical support"
        )
    if not len(work):
        raise ValueError("no repeated participant-word cells")
    return work, support


def write_technical(
    path: Path,
    feature: str,
    support: dict,
    point_status: str,
    completed: int,
    failed: int,
    elapsed: float,
    final: bool,
    precision_mode: str,
    bootstrap_target: int,
    bootstrap_minimum: int,
    compute_decision_sha256: str,
) -> None:
    record = {
        "analysis": "ptsd_stop_lexical_reliability_rq1_blind_technical",
        "feature": feature,
        "support": support,
        "point_fit_status": point_status,
        "precision_mode": precision_mode,
        "bootstrap_requested": bootstrap_target,
        "bootstrap_completed": completed,
        "bootstrap_failed": failed,
        "bootstrap_minimum_success": bootstrap_minimum,
        "compute_decision_sha256": compute_decision_sha256,
        "elapsed_s": elapsed,
        "final": final,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_estimates_inspected": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", required=True, choices=FEATURES)
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--extraction-closure", required=True, type=Path)
    parser.add_argument("--identity-authority", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--technical-output", required=True, type=Path)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--compute-decision", type=Path)
    args = parser.parse_args()
    for path in (
        args.token_dir,
        args.extraction_closure,
        args.identity_authority,
        args.private_output,
        args.technical_output,
    ):
        reject(path)
    if args.benchmark_only:
        if args.feature != FEATURES[0]:
            raise ValueError("benchmark-only is fixed to the first feature")
        if args.compute_decision is not None:
            raise ValueError("benchmark-only cannot consume a compute decision")
        precision_mode = "benchmark_one_point_plus_replicate_0"
        bootstrap_target = 1
        bootstrap_minimum = 0
        decision_sha = "not_yet_selected"
    else:
        if args.compute_decision is None:
            raise ValueError("--compute-decision is required outside benchmark mode")
        reject(args.compute_decision)
        precision_mode, bootstrap_target, bootstrap_minimum, decision_sha = (
            load_compute_decision(args.compute_decision)
        )
    verify_closure(args.extraction_closure)
    frame = apply_identity_authority(
        load_tokens(args.token_dir), args.identity_authority
    )
    work, support = prepare_feature(frame, args.feature)
    participants = tuple(sorted(work["participant_id"].unique()))
    draws = bootstrap_draws(participants)
    checkpoint = args.private_output.with_suffix(
        args.private_output.suffix + ".checkpoint"
    )
    started = time.time()

    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        exact = {
            "feature": args.feature,
            "seed": BOOTSTRAP_SEED,
            "protocol": PROTOCOL,
            "input_closure_sha256": file_sha(args.extraction_closure),
            "identity_authority_sha256": file_sha(args.identity_authority),
        }
        for key, expected in exact.items():
            if state.get(key) != expected:
                raise ValueError(f"PRIVATE checkpoint mismatch for {key}")
        point = state["point"]
        records = state.get("bootstrap_records", [])
    else:
        point = fit_crossed_reml(work)
        records = []
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({
            "feature": args.feature,
            "seed": BOOTSTRAP_SEED,
            "protocol": PROTOCOL,
            "input_closure_sha256": file_sha(args.extraction_closure),
            "identity_authority_sha256": file_sha(args.identity_authority),
            "point": point,
            "bootstrap_records": records,
        }, indent=2), encoding="utf-8")
        print(json.dumps({
            "feature": args.feature,
            "point_fit": "complete",
            "fit_seconds": point["fit_seconds"],
            "scientific_estimate_exposed": False,
        }), flush=True)

    completed_replicates = {int(record["replicate"]) for record in records}
    loop_target = 1 if args.benchmark_only else bootstrap_target
    for replicate in range(loop_target):
        if replicate in completed_replicates:
            continue
        try:
            sampled = bootstrap_frame(work, participants, draws[replicate])
            fit = fit_crossed_reml(sampled)
            records.append({
                "replicate": replicate,
                "status": "complete",
                "repeatability": fit["repeatability"],
                "fit_seconds": fit["fit_seconds"],
            })
        except Exception as error:
            records.append({
                "replicate": replicate,
                "status": "failed",
                "error_type": type(error).__name__,
            })
        records.sort(key=lambda record: int(record["replicate"]))
        if (
            (replicate + 1) % PROGRESS_INTERVAL == 0
            or replicate == loop_target - 1
        ):
            state = {
                "feature": args.feature,
                "seed": BOOTSTRAP_SEED,
                "protocol": PROTOCOL,
                "input_closure_sha256": file_sha(args.extraction_closure),
            "identity_authority_sha256": file_sha(args.identity_authority),
                "point": point,
                "bootstrap_records": records,
            }
            checkpoint.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
            successes = sum(record["status"] == "complete" for record in records)
            failures = len(records) - successes
            write_technical(
                args.technical_output,
                args.feature,
                support,
                "complete",
                successes,
                failures,
                time.time() - started,
                False,
                precision_mode,
                bootstrap_target,
                bootstrap_minimum,
                decision_sha,
            )
            print(json.dumps({
                "feature": args.feature,
                "bootstrap_attempted": len(records),
                "bootstrap_complete": successes,
                "bootstrap_failed": failures,
                "scientific_estimate_exposed": False,
            }), flush=True)

    if args.benchmark_only:
        record0 = next(
            record for record in records if int(record["replicate"]) == 0
        )
        benchmark = {
            "analysis": "ptsd_stop_lexical_rq1_outcome_blind_benchmark",
            "protocol": PRECISION_PROTOCOL,
            "feature": args.feature,
            "support": support,
            "point_fit_status": "complete",
            "point_fit_seconds": float(point["fit_seconds"]),
            "bootstrap_replicate": 0,
            "bootstrap_status": record0["status"],
            "bootstrap_fit_seconds": record0.get("fit_seconds"),
            "elapsed_s": time.time() - started,
            "peak_rss_mib": peak_rss_mib(),
            "cpu_count": os.cpu_count(),
            "checkpoint_sha256": file_sha(checkpoint),
            "identity_authority_sha256": file_sha(args.identity_authority),
            "labels_loaded": False,
            "outcomes_loaded": False,
            "scientific_estimates_inspected": False,
            "scientific_estimates_exposed": False,
        }
        args.technical_output.parent.mkdir(parents=True, exist_ok=True)
        args.technical_output.write_text(
            json.dumps(benchmark, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "feature": args.feature,
            "benchmark_complete": record0["status"] == "complete",
            "scientific_estimate_exposed": False,
        }))
        return 0 if record0["status"] == "complete" else 2

    inferential_records = [
        record for record in records
        if int(record["replicate"]) < bootstrap_target
    ]
    values = [
        float(record["repeatability"])
        for record in inferential_records
        if record["status"] == "complete"
    ]
    failures = [
        record for record in inferential_records
        if record["status"] == "failed"
    ]
    private_report = {
        "analysis": "ptsd_stop_lexical_reliability_rq1",
        "protocol": PROTOCOL,
        "feature": args.feature,
        "estimator": (
            "crossed_REML_participant_word_interaction_over_"
            "interaction_plus_residual"
        ),
        "variance_component_formula": VC_FORMULA,
        "support": support,
        "point": point,
        "bootstrap": {
            "rng": "numpy.random.PCG64",
            "seed": BOOTSTRAP_SEED,
            "precision_mode": precision_mode,
            "requested_replicates": bootstrap_target,
            "successful_replicates": len(values),
            "failed_replicates": len(failures),
            "failed_details": failures,
            "ci_95": (
                [percentile(values, 0.025), percentile(values, 0.975)]
                if bootstrap_target > 0 and len(values) >= bootstrap_minimum
                else [None, None]
            ),
            "technical_review_required": (
                bootstrap_target == 0 or len(values) < bootstrap_minimum
            ),
        },
        "input_closure_sha256": file_sha(args.extraction_closure),
        "identity_authority_sha256": file_sha(args.identity_authority),
        "compute_decision_sha256": decision_sha,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "outcome_analysis_performed": False,
    }
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(
        json.dumps(private_report, indent=2), encoding="utf-8"
    )
    write_technical(
        args.technical_output,
        args.feature,
        support,
        "complete",
        len(values),
        len(failures),
        time.time() - started,
        True,
        precision_mode,
        bootstrap_target,
        bootstrap_minimum,
        decision_sha,
    )
    print(json.dumps({
        "feature": args.feature,
        "status": (
            "complete"
            if bootstrap_target > 0 and len(values) >= bootstrap_minimum
            else "technical_review"
        ),
        "bootstrap_complete": len(values),
        "bootstrap_failed": len(failures),
        "scientific_estimate_exposed": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
