#!/usr/bin/env python3
"""Prepare DAIC participant summaries and PHQ-8 labels after RQ2 gate pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .daic_aggregate_reliability_blind_v1 import (
        EXPECTED_PARTICIPANTS,
        FEATURES as AGGREGATE_FEATURES,
        load_tokens,
    )
except ImportError:  # direct script execution
    from daic_aggregate_reliability_blind_v1 import (
        EXPECTED_PARTICIPANTS,
        FEATURES as AGGREGATE_FEATURES,
        load_tokens,
    )

CORRELATION_FEATURES = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "duration_log_syllable_residual",
)
FEATURE_MAP = {
    "f0_mean_hz": "f0_mean_hz",
    "f0_robust_range_hz": "f0_robust_range_hz",
    "energy_db": "energy_db",
    "syllable_adjusted_log_duration": "duration_log_syllable_residual",
}
EXPECTED_TRAIN = 107
EXPECTED_DEV = 35
EXPECTED_TEST = 47
MIN_FULL_PARTICIPANT_TOKENS = 100


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("analysis") != "daic_aggregate_reliability_rq2_unblinded":
        raise ValueError("unexpected aggregate reliability report")
    if report.get("labels_loaded") is not False or report.get("outcome_analysis_performed") is not False:
        raise ValueError("aggregate report does not precede outcome access")
    features = report.get("features", [])
    if tuple(record.get("feature") for record in features) != AGGREGATE_FEATURES:
        raise ValueError("aggregate feature order mismatch")
    return report


def build_gate_adapter(report: dict) -> dict:
    records = []
    for record in report["features"]:
        interval = record.get("ci_95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("aggregate confidence interval absent")
        records.append(
            {
                "feature": FEATURE_MAP[record["feature"]],
                "status": record["status"],
                "corrected_reliability": float(record["corrected_reliability"]),
                "ci_lower": float(interval[0]),
                "ci_upper": float(interval[1]),
                "max_robustness_delta": float(record["max_robustness_delta"]),
            }
        )
    if tuple(record["feature"] for record in records) != CORRELATION_FEATURES:
        raise ValueError("correlation feature mapping mismatch")
    return {
        "corpus": "DAIC-WOZ",
        "source_analysis": report["analysis"],
        "features": records,
    }


def _residual_summary(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    work = frame[["participant_id", "relative_word_position"]].copy()
    columns = [np.ones(len(work)), work["relative_word_position"].to_numpy(float)]
    if feature == "syllable_adjusted_log_duration":
        work["y"] = pd.to_numeric(frame["log_word_duration"], errors="coerce")
        work["syllables"] = pd.to_numeric(frame["authoritative_syllable_count"], errors="coerce")
        valid = np.isfinite(work["y"]) & np.isfinite(work["syllables"]) & (work["syllables"] > 0)
        work = work.loc[valid].copy()
        columns = [
            np.ones(len(work)),
            work["relative_word_position"].to_numpy(float),
            work["syllables"].to_numpy(float),
        ]
    else:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
        work = work.loc[np.isfinite(work["y"])].copy()
        columns = [np.ones(len(work)), work["relative_word_position"].to_numpy(float)]
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"rank-deficient outcome-summary nuisance model for {feature}")
    beta, *_ = np.linalg.lstsq(design, work["y"].to_numpy(float), rcond=None)
    work["residual"] = work["y"].to_numpy(float) - design @ beta
    summary = work.groupby("participant_id", sort=True)["residual"].agg(["median", "size"]).reset_index()
    summary = summary.loc[summary["size"] >= MIN_FULL_PARTICIPANT_TOKENS, ["participant_id", "median"]]
    summary = summary.rename(columns={"median": FEATURE_MAP[feature]})
    return summary


def participant_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"participant_id": sorted(frame["participant_id"].unique())})
    for feature in AGGREGATE_FEATURES:
        output = output.merge(_residual_summary(frame, feature), on="participant_id", how="left", validate="one_to_one")
    if len(output) != EXPECTED_PARTICIPANTS:
        raise ValueError("participant acoustic summary row count mismatch")
    if output[list(CORRELATION_FEATURES)].isna().any().any():
        raise ValueError("a gate-family participant summary lacks frozen support")
    return output


def _label_piece(path: Path, score_column: str, expected_rows: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    id_candidates = [column for column in data.columns if column.lower() == "participant_id"]
    if len(id_candidates) != 1 or score_column not in data.columns:
        raise ValueError(f"unexpected PHQ schema: {path.name}")
    piece = data[[id_candidates[0], score_column]].rename(columns={id_candidates[0]: "participant_id", score_column: "PHQ8_Score"})
    piece["participant_id"] = piece["participant_id"].astype(int).astype(str)
    piece["PHQ8_Score"] = pd.to_numeric(piece["PHQ8_Score"], errors="raise")
    if len(piece) != expected_rows or piece["participant_id"].duplicated().any():
        raise ValueError(f"PHQ row-count or uniqueness mismatch: {path.name}")
    if not piece["PHQ8_Score"].between(0, 24).all():
        raise ValueError("PHQ-8 score outside 0..24")
    return piece


def combine_labels(train: Path, dev: Path, full_test: Path, public_test_roster: Path) -> pd.DataFrame:
    train_piece = _label_piece(train, "PHQ8_Score", EXPECTED_TRAIN)
    dev_piece = _label_piece(dev, "PHQ8_Score", EXPECTED_DEV)
    test_piece = _label_piece(full_test, "PHQ_Score", EXPECTED_TEST)
    roster = pd.read_csv(public_test_roster)
    id_candidates = [column for column in roster.columns if column.lower() == "participant_id"]
    if len(id_candidates) != 1 or len(roster) != EXPECTED_TEST:
        raise ValueError("public test roster schema or row count mismatch")
    roster_ids = set(roster[id_candidates[0]].astype(int).astype(str))
    if set(test_piece["participant_id"]) != roster_ids:
        raise ValueError("full-test PHQ IDs do not match the official test roster")
    labels = pd.concat([train_piece, dev_piece, test_piece], ignore_index=True)
    if len(labels) != EXPECTED_PARTICIPANTS or labels["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise ValueError("combined PHQ labels do not cover 189 unique participants")
    return labels.sort_values("participant_id", kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", required=True, type=Path)
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--dev-labels", required=True, type=Path)
    parser.add_argument("--full-test-labels", required=True, type=Path)
    parser.add_argument("--public-test-roster", required=True, type=Path)
    parser.add_argument("--gate-output", required=True, type=Path)
    parser.add_argument("--acoustic-output", required=True, type=Path)
    parser.add_argument("--label-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    # Read and validate the same-corpus gate before touching any label file.
    gate_report = load_gate_report(args.gate_report)
    gate_adapter = build_gate_adapter(gate_report)
    if not any(record["status"] == "pass" for record in gate_adapter["features"]):
        raise ValueError("no gate-passing DAIC feature; label access is forbidden")

    frame = load_tokens(args.token_dir)
    acoustics = participant_summaries(frame)
    labels = combine_labels(args.train_labels, args.dev_labels, args.full_test_labels, args.public_test_roster)
    if set(acoustics["participant_id"]) != set(labels["participant_id"]):
        raise ValueError("acoustic and PHQ participant coverage mismatch")

    for path in (args.gate_output, args.acoustic_output, args.label_output, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.write_text(json.dumps(gate_adapter, indent=2), encoding="utf-8")
    acoustics.to_csv(args.acoustic_output, index=False)
    labels.to_csv(args.label_output, index=False)
    manifest = {
        "status": "DAIC_GATE_LICENSED_INPUTS_COMPLETE",
        "corpus": "DAIC-WOZ",
        "participants": EXPECTED_PARTICIPANTS,
        "labels_loaded": True,
        "outcome_analysis_performed": False,
        "gate_report_sha256": file_sha(args.gate_report),
        "gate_adapter_sha256": file_sha(args.gate_output),
        "acoustic_summary_sha256": file_sha(args.acoustic_output),
        "phq8_labels_sha256": file_sha(args.label_output),
        "source_label_hashes": {
            "train": file_sha(args.train_labels),
            "dev": file_sha(args.dev_labels),
            "full_test": file_sha(args.full_test_labels),
            "public_test_roster": file_sha(args.public_test_roster),
        },
        "feature_order": list(CORRELATION_FEATURES),
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "participants": EXPECTED_PARTICIPANTS, "labels_loaded": True, "outcome_analysis_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
