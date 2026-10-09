#!/usr/bin/env python3
"""Amendment J2, second part: does the frozen component table predict the
frozen ESD split-half reliability at the token count that analysis actually had?

Refits nothing.  Reuses the frozen preprocessing by importing the frozen
stability script, counts the tokens the frozen split-half used, and
evaluates the closed form at that k.  Reports the comparison as it comes out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esd_msp_style_speaker_stability_v1 as frozen

FEATURE_MAP = {
    "f0_mean_hz": "f0_mean_hz",
    "f0_robust_range_hz": "f0_robust_range_hz",
    "energy_db": "energy_db",
    "syllable_adjusted_log_duration": "duration_log_syllable_residual",
}
MEDIAN_EFFICIENCY = math.pi / 2.0


def pooled_components(rows: list[dict], component_feature: str) -> dict[str, float]:
    selected = [r for r in rows if r["feature"] == component_feature]
    if len(selected) != len(frozen.CONDITIONS):
        raise ValueError(f"expected one row per condition for {component_feature}")
    out = {}
    for key in ("variance_S", "variance_W", "variance_SW", "variance_e"):
        out[key] = sum(float(r[key]) for r in selected) / len(selected)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.components.open(newline="", encoding="utf-8") as handle:
        component_rows = list(csv.DictReader(handle))
    observed = json.loads(args.observed.read_text(encoding="utf-8"))
    observed_by_feature = {
        record["feature"]: float(record["point"]["spearman_brown"])
        for record in observed["features"]
    }

    frame = frozen.load_tokens(args.tokens, frozen.DEFAULT_EXPECTED_SPEAKERS)
    mapping = frozen.primary_prompt_map(frame["prompt_id"])

    records = []
    for feature in frozen.FEATURES:
        residuals = frozen.residualize_once(frame, feature)
        residuals = residuals.copy()
        residuals["half"] = frozen.assign(residuals, mapping)
        counts = residuals.groupby(["speaker_id", "half"], sort=True).size()
        k_median = float(np.median(counts.to_numpy(float)))
        k_min = float(counts.min())

        components = pooled_components(component_rows, FEATURE_MAP[feature])
        s = components["variance_S"]
        sw = components["variance_SW"]
        e = components["variance_e"]
        within = sw + e

        def curve(k: float, inflation: float) -> float:
            return s / (s + inflation * within / k)

        records.append(
            {
                "feature": feature,
                "pooled_variance_S": s,
                "pooled_variance_W": components["variance_W"],
                "pooled_variance_SW": sw,
                "pooled_variance_e": e,
                "repeatability_closed_form": sw / within,
                "tokens_per_speaker_half_median": k_median,
                "tokens_per_speaker_half_min": k_min,
                "predicted_mean_estimator": curve(k_median, 1.0),
                "predicted_median_estimator": curve(k_median, MEDIAN_EFFICIENCY),
                "observed_spearman_brown": observed_by_feature[feature],
                "tokens_for_0.80": 0.80 * within / (0.20 * s),
                "tokens_for_0.90": 0.90 * within / (0.10 * s),
            }
        )

    payload = {
        "analysis": "esd_component_prediction_J2b_v1",
        "amendment": "MEASUREMENT_MODEL_amendment_J_20260904.md",
        "models_refit": 0,
        "tokens_sha256": frozen.file_sha(args.tokens),
        "approximations_declared": [
            "components are the equal-weight mean of the five frozen per-condition fits",
            "the frozen statistic uses speaker medians; the median column inflates the "
            "within-speaker term by pi/2, the asymptotic normal-theory penalty",
            "condition is pooled rather than modelled as a further crossed factor",
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"features": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
