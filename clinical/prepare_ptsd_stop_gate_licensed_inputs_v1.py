#!/usr/bin/env python3
"""Prepare PTSD-STOP participant summaries and PCL labels after RQ2 gate pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ptsd_stop_aggregate_reliability_blind_v1 import (
        EXPECTED_ANALYSIS_PARTICIPANTS,
        FEATURES as AGGREGATE_FEATURES,
        apply_identity_authority,
        load_tokens,
    )
except ImportError:
    from ptsd_stop_aggregate_reliability_blind_v1 import (
        EXPECTED_ANALYSIS_PARTICIPANTS,
        FEATURES as AGGREGATE_FEATURES,
        apply_identity_authority,
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
EXPECTED_LABEL_PARTICIPANTS = 213
MIN_FULL_PARTICIPANT_TOKENS = 100
PCL_COLUMN = "pcl_mean_days_0_60"
PCL_AUTHORITY_SHA256 = "870e511d783bfeaa06e6c1f684dac212ffec30de22cd4648089c421cf2bef992"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("analysis") != "ptsd_stop_aggregate_reliability_rq2_unblinded":
        raise ValueError("unexpected PTSD-STOP gate report")
    if report.get("outcomes_loaded") is not False:
        raise ValueError("gate report does not precede outcome access")
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
        records.append({
            "feature": FEATURE_MAP[record["feature"]],
            "status": record["status"],
            "corrected_reliability": float(record["corrected_reliability"]),
            "ci_lower": float(interval[0]),
            "ci_upper": float(interval[1]),
            "max_robustness_delta": float(record["max_robustness_delta"]),
        })
    if tuple(row["feature"] for row in records) != CORRELATION_FEATURES:
        raise ValueError("correlation feature mapping mismatch")
    return {"corpus": "PTSD-STOP", "source_analysis": report["analysis"], "features": records}


def _residual_summary(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    work = frame[["participant_id", "relative_word_position"]].copy()
    if feature == "syllable_adjusted_log_duration":
        work["y"] = pd.to_numeric(frame["log_word_duration"], errors="coerce")
        work["syllables"] = pd.to_numeric(frame["authoritative_syllable_count"], errors="coerce")
        valid = np.isfinite(work["y"]) & np.isfinite(work["syllables"]) & (work["syllables"] > 0)
        work = work.loc[valid].copy()
        columns = [np.ones(len(work)), work["relative_word_position"].to_numpy(float), work["syllables"].to_numpy(float)]
    else:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
        work = work.loc[np.isfinite(work["y"])].copy()
        columns = [np.ones(len(work)), work["relative_word_position"].to_numpy(float)]
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"rank-deficient summary nuisance model for {feature}")
    beta, *_ = np.linalg.lstsq(design, work["y"].to_numpy(float), rcond=None)
    work["residual"] = work["y"].to_numpy(float) - design @ beta
    summary = work.groupby("participant_id", sort=True)["residual"].agg(["median", "size"]).reset_index()
    summary = summary.loc[summary["size"] >= MIN_FULL_PARTICIPANT_TOKENS, ["participant_id", "median"]]
    return summary.rename(columns={"median": FEATURE_MAP[feature]})


def participant_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"participant_id": sorted(frame["participant_id"].unique())})
    for feature in AGGREGATE_FEATURES:
        output = output.merge(_residual_summary(frame, feature), on="participant_id", how="left", validate="one_to_one")
    if len(output) != EXPECTED_ANALYSIS_PARTICIPANTS:
        raise ValueError("authoritative participant summary row count mismatch")
    return output


def validate_labeled_acoustic_support(
    acoustics: pd.DataFrame, labels: pd.DataFrame
) -> None:
    """Require complete frozen features for the prespecified PCL cohort only."""
    label_ids = set(labels["participant_id"])
    if not label_ids.issubset(set(acoustics["participant_id"])):
        raise ValueError("official PCL users are not a subset of acoustic users")
    labeled = acoustics.loc[acoustics["participant_id"].isin(label_ids)]
    if len(labeled) != EXPECTED_LABEL_PARTICIPANTS:
        raise ValueError("labeled acoustic participant count mismatch")
    if labeled[list(CORRELATION_FEATURES)].isna().any().any():
        raise ValueError("a labeled participant lacks frozen acoustic support")


def load_labels(path: Path) -> pd.DataFrame:
    if file_sha(path) != PCL_AUTHORITY_SHA256:
        raise ValueError("official PCL authority checksum mismatch")
    labels = pd.read_csv(path, usecols=["person_id", PCL_COLUMN], dtype={"person_id": str})
    labels = labels.rename(columns={"person_id": "participant_id"})
    labels["participant_id"] = "P" + labels["participant_id"].str.replace(r"[^0-9]", "", regex=True)
    labels[PCL_COLUMN] = pd.to_numeric(labels[PCL_COLUMN], errors="raise")
    if len(labels) != EXPECTED_LABEL_PARTICIPANTS or labels["participant_id"].duplicated().any():
        raise ValueError("official PCL label cardinality mismatch")
    if not np.isfinite(labels[PCL_COLUMN]).all():
        raise ValueError("official PCL label is nonfinite")
    return labels.sort_values("participant_id", kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", required=True, type=Path)
    parser.add_argument("--token-dir", required=True, type=Path)
    parser.add_argument("--identity-authority", required=True, type=Path)
    parser.add_argument("--pcl-authority", required=True, type=Path)
    parser.add_argument("--gate-output", required=True, type=Path)
    parser.add_argument("--acoustic-output", required=True, type=Path)
    parser.add_argument("--label-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    gate_report = load_gate_report(args.gate_report)
    gate_adapter = build_gate_adapter(gate_report)
    if not any(row["status"] == "pass" for row in gate_adapter["features"]):
        raise ValueError("no gate-passing PTSD-STOP feature; PCL access forbidden")
    frame = apply_identity_authority(load_tokens(args.token_dir), args.identity_authority)
    acoustics = participant_summaries(frame)
    labels = load_labels(args.pcl_authority)
    validate_labeled_acoustic_support(acoustics, labels)
    for path in (args.gate_output, args.acoustic_output, args.label_output, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.write_text(json.dumps(gate_adapter, indent=2), encoding="utf-8")
    acoustics.to_csv(args.acoustic_output, index=False)
    labels.to_csv(args.label_output, index=False)
    manifest = {
        "status": "PTSD_STOP_GATE_LICENSED_INPUTS_COMPLETE",
        "analysis_participants": EXPECTED_ANALYSIS_PARTICIPANTS,
        "label_participants": EXPECTED_LABEL_PARTICIPANTS,
        "outcome": PCL_COLUMN,
        "gate_report_sha256": file_sha(args.gate_report),
        "identity_authority_sha256": file_sha(args.identity_authority),
        "pcl_authority_sha256": file_sha(args.pcl_authority),
        "acoustic_summary_sha256": file_sha(args.acoustic_output),
        "pcl_labels_sha256": file_sha(args.label_output),
        "feature_order": list(CORRELATION_FEATURES),
        "labels_loaded": True,
        "outcome_analysis_performed": False,
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "label_participants": EXPECTED_LABEL_PARTICIPANTS, "labels_loaded": True, "outcome_analysis_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
