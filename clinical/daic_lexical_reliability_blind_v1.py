#!/usr/bin/env python3
"""Blind DAIC RQ1 participant-by-word crossed-REML reliability.

Run one frozen feature per process. Scientific estimates and checkpoints remain
PRIVATE; stdout and the technical JSON expose only support, timing, and fit
completion status.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .daic_aggregate_reliability_blind_v1 import (
        EXPECTED_PARTICIPANTS,
        FEATURES,
        BOOTSTRAP_MIN_SUCCESS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        file_sha,
        load_tokens,
        percentile,
        verify_closure,
    )
except ImportError:  # direct script execution
    from daic_aggregate_reliability_blind_v1 import (
        EXPECTED_PARTICIPANTS,
        FEATURES,
        BOOTSTRAP_MIN_SUCCESS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
        file_sha,
        load_tokens,
        percentile,
        verify_closure,
    )

VC_FORMULA = {
    "S": "0 + C(participant)",
    "W": "0 + C(word)",
    "SW": "0 + C(participant_word)",
}
MIN_CELL_TOKENS = 2
PROGRESS_INTERVAL = 10
FORBIDDEN = ("phq", "pcl", "label", "outcome", "diagnos", "demograph")


def reject(path: Path) -> None:
    if any(part in str(path).lower() for part in FORBIDDEN):
        raise ValueError(f"forbidden clinical path: {path}")


def duration_fixed_effect_residual(frame: pd.DataFrame) -> pd.Series:
    """Residual for log(duration) ~ 1 + syllables + C(participant).

    The within-participant calculation is algebraically equivalent to the full
    dummy-coded OLS model and avoids a dense 278k by 189 design matrix.
    """
    y = pd.to_numeric(frame["log_word_duration"], errors="coerce")
    x = pd.to_numeric(frame["authoritative_syllable_count"], errors="coerce")
    valid = np.isfinite(y) & np.isfinite(x) & (x > 0)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    work = pd.DataFrame(
        {
            "participant": frame.loc[valid, "participant_id"].astype(str),
            "y": y.loc[valid].astype(float),
            "x": x.loc[valid].astype(float),
        },
        index=frame.index[valid],
    )
    y_centered = work["y"] - work.groupby("participant", sort=False)["y"].transform("mean")
    x_centered = work["x"] - work.groupby("participant", sort=False)["x"].transform("mean")
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("duration fixed-effect slope is not identifiable")
    beta = float(np.dot(x_centered, y_centered) / denominator)
    output.loc[work.index] = y_centered - beta * x_centered
    return output


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
    counts = work.groupby(["participant_id", "normalized_word"], sort=False).size()
    eligible = set(counts.loc[counts >= MIN_CELL_TOKENS].index)
    mask = [pair in eligible for pair in zip(work["participant_id"], work["normalized_word"])]
    work = work.loc[mask].reset_index(drop=True)
    support = {
        "participants": int(work["participant_id"].nunique()),
        "words": int(work["normalized_word"].nunique()),
        "matched_participant_word_cells": int(len(eligible)),
        "tokens": int(len(work)),
        "minimum_tokens_per_cell": MIN_CELL_TOKENS,
    }
    if support["participants"] != EXPECTED_PARTICIPANTS:
        raise ValueError("not all 189 participants retain repeated lexical support")
    if not len(work):
        raise ValueError("no repeated participant-word cells")
    return work, support


def fit_crossed_reml(work: pd.DataFrame) -> dict:
    from scipy import sparse
    from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec

    frame = work[["participant_id", "normalized_word", "y"]].copy()
    frame["participant"] = frame["participant_id"].astype(str)
    frame["word"] = frame["normalized_word"].astype(str)
    frame["participant_word"] = frame["participant"] + "\x1f" + frame["word"]
    matrices = []
    column_names = []
    for column in ("participant", "word", "participant_word"):
        codes, levels = pd.factorize(frame[column], sort=True)
        matrix = sparse.csr_array(
            (
                np.ones(len(frame), dtype=float),
                (np.arange(len(frame), dtype=int), codes.astype(int)),
            ),
            shape=(len(frame), len(levels)),
        )
        matrices.append([matrix])
        column_names.append([[str(level) for level in levels]])
    variance_components = VCSpec(
        names=["S", "W", "SW"],
        colnames=column_names,
        mats=matrices,
    )
    model = MixedLM(
        endog=frame["y"].to_numpy(float),
        exog=np.ones((len(frame), 1), dtype=float),
        groups=np.zeros(len(frame), dtype=int),
        exog_re=None,
        exog_vc=variance_components,
        use_sqrt=True,
    )
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True, method="lbfgs", disp=False)
    elapsed = time.time() - started
    if not bool(result.converged):
        raise ValueError("crossed REML did not converge")
    names = list(model.exog_vc.names)
    values = [float(value) for value in result.vcomp]
    if len(names) != len(values) or set(names) != {"S", "W", "SW"}:
        raise ValueError("unexpected variance-component mapping")
    components = dict(zip(names, values))
    residual = float(result.scale)
    if not all(np.isfinite(value) and value >= 0 for value in (*components.values(), residual)):
        raise ValueError("nonfinite or negative variance component")
    denominator = components["SW"] + residual
    if denominator <= 0:
        raise ValueError("zero participant-word plus residual denominator")
    return {
        "repeatability": components["SW"] / denominator,
        "variance_components": {**components, "e": residual},
        "converged": True,
        "fit_seconds": elapsed,
    }


def bootstrap_draws(participants: tuple[str, ...]) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return generator.integers(
        0,
        len(participants),
        size=(BOOTSTRAP_REPLICATES, len(participants)),
        endpoint=False,
    )


def bootstrap_frame(source: pd.DataFrame, participants: tuple[str, ...], draw: np.ndarray) -> pd.DataFrame:
    by_participant = {participant: group for participant, group in source.groupby("participant_id", sort=False)}
    pieces = []
    for cluster_index, source_index in enumerate(draw):
        source_participant = participants[int(source_index)]
        part = by_participant[source_participant].copy()
        part["participant_id"] = f"bootstrap_cluster_{cluster_index:03d}"
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def write_technical(path: Path, feature: str, support: dict, point_status: str, completed: int, failed: int, elapsed: float, final: bool) -> None:
    record = {
        "analysis": "daic_lexical_reliability_rq1_blind_technical",
        "feature": feature,
        "support": support,
        "point_fit_status": point_status,
        "bootstrap_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_completed": completed,
        "bootstrap_failed": failed,
        "bootstrap_minimum_success": BOOTSTRAP_MIN_SUCCESS,
        "elapsed_s": elapsed,
        "final": final,
        "labels_loaded": False,
        "scientific_estimates_inspected": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", required=True, choices=FEATURES)
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--extraction-closure", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--technical-output", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.token_dir, args.extraction_closure, args.private_output, args.technical_output):
        reject(path)
    verify_closure(args.extraction_closure)
    frame = load_tokens(args.token_dir)
    work, support = prepare_feature(frame, args.feature)
    participants = tuple(sorted(work["participant_id"].unique()))
    draws = bootstrap_draws(participants)
    checkpoint = args.private_output.with_suffix(args.private_output.suffix + ".checkpoint")
    started = time.time()

    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("feature") != args.feature or state.get("seed") != BOOTSTRAP_SEED:
            raise ValueError("PRIVATE checkpoint does not match frozen feature/seed")
        point = state["point"]
        records = state.get("bootstrap_records", [])
    else:
        point = fit_crossed_reml(work)
        records = []
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"feature": args.feature, "seed": BOOTSTRAP_SEED, "point": point, "bootstrap_records": records}, indent=2), encoding="utf-8")
        print(json.dumps({"feature": args.feature, "point_fit": "complete", "fit_seconds": point["fit_seconds"], "scientific_estimate_exposed": False}), flush=True)

    completed_replicates = {int(record["replicate"]) for record in records}
    for replicate in range(BOOTSTRAP_REPLICATES):
        if replicate in completed_replicates:
            continue
        try:
            sampled = bootstrap_frame(work, participants, draws[replicate])
            fit = fit_crossed_reml(sampled)
            records.append({"replicate": replicate, "status": "complete", "repeatability": fit["repeatability"], "fit_seconds": fit["fit_seconds"]})
        except Exception as error:
            records.append({"replicate": replicate, "status": "failed", "error_type": type(error).__name__})
        records.sort(key=lambda record: int(record["replicate"]))
        if (replicate + 1) % PROGRESS_INTERVAL == 0 or replicate == BOOTSTRAP_REPLICATES - 1:
            state = {"feature": args.feature, "seed": BOOTSTRAP_SEED, "point": point, "bootstrap_records": records}
            checkpoint.write_text(json.dumps(state, indent=2), encoding="utf-8")
            successes = sum(record["status"] == "complete" for record in records)
            failures = len(records) - successes
            write_technical(args.technical_output, args.feature, support, "complete", successes, failures, time.time() - started, False)
            print(json.dumps({"feature": args.feature, "bootstrap_attempted": len(records), "bootstrap_complete": successes, "bootstrap_failed": failures, "scientific_estimate_exposed": False}), flush=True)

    values = [float(record["repeatability"]) for record in records if record["status"] == "complete"]
    failures = [record for record in records if record["status"] == "failed"]
    private_report = {
        "analysis": "daic_lexical_reliability_rq1",
        "feature": args.feature,
        "estimator": "crossed_REML_participant_word_interaction_over_interaction_plus_residual",
        "variance_component_formula": VC_FORMULA,
        "support": support,
        "point": point,
        "bootstrap": {
            "rng": "numpy.random.PCG64",
            "seed": BOOTSTRAP_SEED,
            "requested_replicates": BOOTSTRAP_REPLICATES,
            "successful_replicates": len(values),
            "failed_replicates": len(failures),
            "failed_details": failures,
            "ci_95": [percentile(values, 0.025), percentile(values, 0.975)],
            "technical_review_required": len(values) < BOOTSTRAP_MIN_SUCCESS,
        },
        "input_closure_sha256": file_sha(args.extraction_closure),
        "labels_loaded": False,
        "outcome_analysis_performed": False,
    }
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(json.dumps(private_report, indent=2), encoding="utf-8")
    write_technical(args.technical_output, args.feature, support, "complete", len(values), len(failures), time.time() - started, True)
    print(json.dumps({"feature": args.feature, "status": "complete" if len(values) >= BOOTSTRAP_MIN_SUCCESS else "technical_review", "bootstrap_complete": len(values), "bootstrap_failed": len(failures), "scientific_estimate_exposed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
