#!/usr/bin/env python3
"""Prospectively specified MSP-style speaker-location analysis for ESD.

Scientific values are written only to a PRIVATE JSON.  The companion SAFE
receipt contains provenance and support counts but no correlations, intervals,
condition profiles, or cross-regime differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "syllable_adjusted_log_duration",
)
CONDITIONS = ("Angry", "Happy", "Neutral", "Sad", "Surprise")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 13
BOOTSTRAP_MIN_SUCCESS = 950
ALTERNATIVE_SPLITS = 25
MIN_TOKENS_PER_HALF = 50
DEFAULT_EXPECTED_SPEAKERS = 9
REQUIRED_COLUMNS = (
    "speaker_id",
    "prompt_id",
    "condition",
    "word_index",
    "normalized_word",
    "log_word_duration",
    "authoritative_syllable_count",
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_key(value: str) -> tuple[int, int | str, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def primary_prompt_map(prompt_ids: pd.Series) -> dict[str, int]:
    prompts = sorted({str(value) for value in prompt_ids}, key=numeric_key)
    if len(prompts) < 2:
        raise ValueError("fewer than two prompts")
    return {prompt: rank % 2 for rank, prompt in enumerate(prompts)}


def alternative_prompt_map(prompt_ids: pd.Series, split_index: int) -> dict[str, int]:
    if split_index < 0 or split_index >= ALTERNATIVE_SPLITS:
        raise ValueError("alternative split index outside frozen range")
    prompts = {str(value) for value in prompt_ids}
    ordered = sorted(
        prompts,
        key=lambda prompt: hashlib.sha256(
            f"ESD-T1-v2.1|{split_index}|{prompt}".encode("utf-8")
        ).digest(),
    )
    return {prompt: rank % 2 for rank, prompt in enumerate(ordered)}


def assign(frame: pd.DataFrame, mapping: dict[str, int]) -> pd.Series:
    result = frame["prompt_id"].astype(str).map(mapping)
    if result.isna().any() or not set(result.unique()).issubset({0, 1}):
        raise ValueError("invalid prompt-half mapping")
    return result.astype(int)


def spearman_brown(r_value: float) -> float:
    value = float(r_value)
    if not np.isfinite(value) or value <= -1:
        raise ValueError("invalid half-correlation for Spearman-Brown correction")
    return 2.0 * value / (1.0 + value)


def correlation_record(pairs: pd.DataFrame, expected_speakers: int) -> dict:
    if len(pairs) != expected_speakers:
        raise ValueError(f"expected {expected_speakers} paired speakers, found {len(pairs)}")
    left = pairs["half_0"].to_numpy(float)
    right = pairs["half_1"].to_numpy(float)
    if np.std(left) <= 0 or np.std(right) <= 0:
        raise ValueError("zero variance in speaker summaries")
    pearson = float(np.corrcoef(left, right)[0, 1])
    rank_left = pd.Series(left).rank(method="average").to_numpy(float)
    rank_right = pd.Series(right).rank(method="average").to_numpy(float)
    spearman = float(np.corrcoef(rank_left, rank_right)[0, 1])
    return {
        "raw_pearson": pearson,
        "spearman_brown": spearman_brown(pearson),
        "raw_spearman": spearman,
        "paired_speakers": len(pairs),
    }


def load_tokens(path: Path, expected_speakers: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing standardized ESD columns: {missing}")
    frame = frame.loc[frame["normalized_word"].fillna("").astype(str).str.len() > 0].copy()
    frame["speaker_id"] = frame["speaker_id"].astype(str)
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str).str.title()
    if set(frame["condition"].unique()) != set(CONDITIONS):
        raise ValueError("ESD condition set mismatch")
    if frame["speaker_id"].nunique() != expected_speakers:
        raise ValueError("ESD speaker count mismatch")
    frame["word_index"] = pd.to_numeric(frame["word_index"], errors="raise").astype(int)
    group = ["speaker_id", "condition", "prompt_id"]
    maximum = frame.groupby(group, sort=False)["word_index"].transform("max")
    minimum = frame.groupby(group, sort=False)["word_index"].transform("min")
    span = maximum - minimum
    frame["relative_word_position"] = np.where(
        span > 0,
        (frame["word_index"] - minimum) / span,
        0.0,
    )
    return frame


def residualize_once(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    columns = ["speaker_id", "prompt_id", "condition", "relative_word_position"]
    work = frame[columns].copy()
    if feature == "syllable_adjusted_log_duration":
        work["y"] = pd.to_numeric(frame["log_word_duration"], errors="coerce")
        work["syllables"] = pd.to_numeric(
            frame["authoritative_syllable_count"], errors="coerce"
        )
        valid = np.isfinite(work["y"]) & np.isfinite(work["syllables"]) & (work["syllables"] > 0)
    elif feature in FEATURES:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
        valid = np.isfinite(work["y"])
    else:
        raise ValueError(f"unexpected feature: {feature}")
    work = work.loc[valid].copy()
    design_parts = [np.ones(len(work)), work["relative_word_position"].to_numpy(float)]
    if feature == "syllable_adjusted_log_duration":
        design_parts.append(work["syllables"].to_numpy(float))
    design = np.column_stack(design_parts)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("global nuisance model is rank deficient")
    beta, *_ = np.linalg.lstsq(design, work["y"].to_numpy(float), rcond=None)
    work["residual"] = work["y"].to_numpy(float) - design @ beta
    return work


def speaker_pairs(
    residuals: pd.DataFrame,
    mapping: dict[str, int],
    condition: str | None = None,
    condition_balanced: bool = False,
) -> tuple[pd.DataFrame, dict]:
    work = residuals if condition is None else residuals.loc[residuals["condition"].eq(condition)]
    work = work.copy()
    work["half"] = assign(work, mapping)
    counts = work.groupby(["speaker_id", "half"], sort=True).size().unstack()
    valid_speakers = counts.index[(counts.get(0, 0) >= MIN_TOKENS_PER_HALF) & (counts.get(1, 0) >= MIN_TOKENS_PER_HALF)]
    work = work.loc[work["speaker_id"].isin(valid_speakers)]
    if condition_balanced:
        per_condition = work.groupby(
            ["speaker_id", "half", "condition"], sort=True
        )["residual"].median().reset_index()
        complete = per_condition.groupby(["speaker_id", "half"], sort=True)["condition"].nunique()
        complete_index = complete.loc[complete.eq(len(CONDITIONS))].index
        per_condition = per_condition.set_index(["speaker_id", "half"]).loc[complete_index].reset_index()
        summary = per_condition.groupby(["speaker_id", "half"], sort=True)["residual"].median()
    else:
        summary = work.groupby(["speaker_id", "half"], sort=True)["residual"].median()
    pivot = summary.unstack().dropna()
    if 0 not in pivot.columns or 1 not in pivot.columns:
        raise ValueError("one ESD prompt half is absent")
    pairs = pivot[[0, 1]].rename(columns={0: "half_0", 1: "half_1"}).reset_index()
    support = {
        "paired_speakers": len(pairs),
        "tokens_half_0": int(work["half"].eq(0).sum()),
        "tokens_half_1": int(work["half"].eq(1).sum()),
    }
    return pairs, support


def bootstrap(pairs: pd.DataFrame, expected_speakers: int) -> dict:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    values: list[float] = []
    failures: list[int] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = pairs.iloc[generator.integers(0, len(pairs), size=len(pairs))]
        try:
            values.append(correlation_record(sampled, expected_speakers)["spearman_brown"])
        except (ValueError, FloatingPointError):
            failures.append(replicate)
    interval = None
    if len(values) >= BOOTSTRAP_MIN_SUCCESS:
        interval = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "requested": BOOTSTRAP_REPLICATES,
        "successful": len(values),
        "failed": len(failures),
        "failed_indices": failures,
        "ci_95": interval,
        "status": "complete" if interval is not None else "technical_review_no_interval",
    }


def split_summary(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    return {
        "n_complete": len(data),
        "median": float(np.median(data)),
        "q1": float(np.quantile(data, 0.25)),
        "q3": float(np.quantile(data, 0.75)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def leave_one_out(pairs: pd.DataFrame) -> dict:
    primary = correlation_record(pairs, len(pairs))["spearman_brown"]
    records = []
    for speaker in pairs["speaker_id"]:
        subset = pairs.loc[~pairs["speaker_id"].eq(speaker)]
        try:
            value = correlation_record(subset, len(subset))["spearman_brown"]
            records.append({"speaker": speaker, "value": value, "absolute_change": abs(value - primary)})
        except ValueError as error:
            records.append({"speaker": speaker, "status": "failed", "error": type(error).__name__})
    changes = [record["absolute_change"] for record in records if "absolute_change" in record]
    return {"records": records, "maximum_absolute_change": max(changes) if changes else None}


def condition_dispersion(residuals: pd.DataFrame) -> list[dict]:
    speaker_condition = residuals.groupby(["speaker_id", "condition"], sort=True)["residual"].median().reset_index()
    mads: dict[str, float] = {}
    for condition in CONDITIONS:
        values = speaker_condition.loc[speaker_condition["condition"].eq(condition), "residual"].to_numpy(float)
        center = float(np.median(values))
        mads[condition] = float(np.median(np.abs(values - center)))
    neutral = mads["Neutral"]
    records = []
    for condition in CONDITIONS:
        ratio = None
        if condition != "Neutral" and neutral > 0 and mads[condition] > 0:
            ratio = float(math.log(mads[condition] / neutral))
        records.append({"condition": condition, "speaker_mad": mads[condition], "log_mad_ratio_to_neutral": ratio})
    return records


def analyze_feature(
    frame: pd.DataFrame,
    feature: str,
    expected_speakers: int,
    msp_reference: float,
) -> tuple[dict, dict]:
    residuals = residualize_once(frame, feature)
    primary_map = primary_prompt_map(residuals["prompt_id"])
    pairs, support = speaker_pairs(residuals, primary_map)
    point = correlation_record(pairs, expected_speakers)
    uncertainty = bootstrap(pairs, expected_speakers)
    alternatives = []
    for split_index in range(ALTERNATIVE_SPLITS):
        mapping = alternative_prompt_map(residuals["prompt_id"], split_index)
        alt_pairs, _ = speaker_pairs(residuals, mapping)
        alternatives.append(correlation_record(alt_pairs, expected_speakers)["spearman_brown"])
    balanced_pairs, _ = speaker_pairs(residuals, primary_map, condition_balanced=True)
    balanced = correlation_record(balanced_pairs, expected_speakers)
    conditions = []
    condition_technical = []
    for condition in CONDITIONS:
        condition_pairs, condition_support = speaker_pairs(residuals, primary_map, condition=condition)
        condition_point = correlation_record(condition_pairs, expected_speakers)
        condition_ci = bootstrap(condition_pairs, expected_speakers)
        conditions.append({"condition": condition, "point": condition_point, "bootstrap": condition_ci})
        condition_technical.append({"condition": condition, **condition_support, "bootstrap_successful": condition_ci["successful"]})
    private = {
        "feature": feature,
        "point": point,
        "bootstrap": uncertainty,
        "alternative_splits": split_summary(alternatives),
        "leave_one_speaker_out": leave_one_out(pairs),
        "condition_balanced": balanced,
        "condition_balanced_absolute_change": abs(balanced["spearman_brown"] - point["spearman_brown"]),
        "msp_reference_spearman_brown": float(msp_reference),
        "delta_esd_minus_msp": point["spearman_brown"] - float(msp_reference),
        "condition_specific_speaker_location": conditions,
        "condition_dispersion": condition_dispersion(residuals),
    }
    technical = {
        "feature": feature,
        **support,
        "bootstrap_successful": uncertainty["successful"],
        "bootstrap_failed": uncertainty["failed"],
        "alternative_splits_complete": len(alternatives),
        "condition_support": condition_technical,
    }
    return private, technical


def read_msp_reference(path: Path) -> dict[str, float]:
    record = json.loads(path.read_text(encoding="utf-8"))
    values = record.get("corrected_reliability_by_feature", record)
    if not isinstance(values, dict) or set(values) != set(FEATURES):
        raise ValueError("MSP reference feature set mismatch")
    output = {feature: float(values[feature]) for feature in FEATURES}
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("nonfinite MSP reference")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--msp-reference", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--technical-output", required=True, type=Path)
    parser.add_argument("--expected-speakers", type=int, default=DEFAULT_EXPECTED_SPEAKERS)
    args = parser.parse_args()
    for output in (args.private_output, args.technical_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    frame = load_tokens(args.tokens, args.expected_speakers)
    references = read_msp_reference(args.msp_reference)
    private_features = []
    technical_features = []
    for feature in FEATURES:
        private, technical = analyze_feature(frame, feature, args.expected_speakers, references[feature])
        private_features.append(private)
        technical_features.append(technical)
    private_report = {
        "analysis": "esd_msp_style_speaker_stability_v1",
        "status": "scientific_private_complete",
        "prospective_secondary": True,
        "condition_components_status": "exploratory_unless_authoritative_access_audit",
        "feature_order": list(FEATURES),
        "condition_order": list(CONDITIONS),
        "tokens_sha256": file_sha(args.tokens),
        "msp_reference_sha256": file_sha(args.msp_reference),
        "features": private_features,
    }
    args.private_output.write_text(json.dumps(private_report, indent=2), encoding="utf-8")
    technical_report = {
        "analysis": "esd_msp_style_speaker_stability_v1_technical",
        "status": "TECHNICAL_COMPLETE",
        "scientific_results_exposed": False,
        "tokens_sha256": private_report["tokens_sha256"],
        "msp_reference_sha256": private_report["msp_reference_sha256"],
        "private_output_sha256": file_sha(args.private_output),
        "rows": len(frame),
        "speakers": frame["speaker_id"].nunique(),
        "prompts": frame["prompt_id"].nunique(),
        "features": technical_features,
    }
    args.technical_output.write_text(json.dumps(technical_report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "TECHNICAL_COMPLETE", "scientific_results_exposed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
