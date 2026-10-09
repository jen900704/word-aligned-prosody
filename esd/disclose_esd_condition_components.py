#!/usr/bin/env python3
"""Disclose frozen ESD condition components without refitting any model."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FEATURE_ORDER = (
    "f0_mean_hz",
    "f0_robust_range_hz",
    "energy_db",
    "duration_log_syllable_residual",
)
CONDITION_ORDER = ("Angry", "Happy", "Neutral", "Sad", "Surprise")


def _condition_mapping(primary: dict) -> dict[str, float]:
    candidates = (
        primary.get("condition_estimates"),
        primary.get("conditions"),
        primary.get("condition_results"),
    )
    raw = next((item for item in candidates if item is not None), None)
    if raw is None:
        raise ValueError("condition-specific estimates are absent")
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for record in raw:
            if not isinstance(record, dict) or "condition" not in record:
                raise ValueError("malformed condition record")
            items.append((record["condition"], record))
    else:
        raise ValueError("unsupported condition-estimate container")
    output = {}
    for condition, value in items:
        if isinstance(value, dict):
            estimate = next(
                (value.get(key) for key in ("repeatability", "R", "estimate", "value") if value.get(key) is not None),
                None,
            )
        else:
            estimate = value
        if estimate is None or not math.isfinite(float(estimate)):
            raise ValueError(f"non-finite component for {condition}")
        output[str(condition).title()] = float(estimate)
    if set(output) != set(CONDITION_ORDER):
        raise ValueError(f"condition set mismatch: {sorted(output)}")
    return output


def _variance_components(primary: dict) -> dict[str, dict[str, float]]:
    raw = primary.get("condition_estimates", primary.get("conditions", primary.get("condition_results")))
    if not isinstance(raw, list):
        raise ValueError("variance components require list-form condition records")
    output: dict[str, dict[str, float]] = {}
    for record in raw:
        if not isinstance(record, dict) or "condition" not in record:
            raise ValueError("malformed condition record")
        condition = str(record["condition"]).title()
        components = record.get("variance_components")
        if not isinstance(components, dict) or set(components) != {"S", "W", "SW", "e"}:
            raise ValueError(f"variance-component set mismatch for {condition}")
        values = {key: float(components[key]) for key in ("S", "W", "SW", "e")}
        if not all(math.isfinite(value) and value >= 0 for value in values.values()):
            raise ValueError(f"invalid variance component for {condition}")
        output[condition] = values
    if set(output) != set(CONDITION_ORDER):
        raise ValueError(f"variance-component condition set mismatch: {sorted(output)}")
    return output


def extract(report: dict, tolerance: float = 5e-10) -> list[dict]:
    records = report.get("features")
    if not isinstance(records, list):
        raise ValueError("top-level features list is absent")
    by_feature = {record.get("feature"): record for record in records if isinstance(record, dict)}
    rows = []
    for feature in FEATURE_ORDER:
        record = by_feature.get(feature)
        if record is None or not isinstance(record.get("primary"), dict):
            raise ValueError(f"primary record absent for {feature}")
        primary = record["primary"]
        components = _condition_mapping(primary)
        variances = _variance_components(primary)
        frozen_mean = float(primary["equal_condition_mean"])
        disclosed_mean = sum(components.values()) / len(CONDITION_ORDER)
        if not math.isclose(frozen_mean, disclosed_mean, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"component mean mismatch for {feature}: {disclosed_mean} != {frozen_mean}"
            )
        for condition in CONDITION_ORDER:
            rows.append(
                {
                    "feature": feature,
                    "condition": condition,
                    "R_c": components[condition],
                    "variance_S": variances[condition]["S"],
                    "variance_W": variances[condition]["W"],
                    "variance_SW": variances[condition]["SW"],
                    "variance_e": variances[condition]["e"],
                    "frozen_equal_condition_mean": frozen_mean,
                    "status": "frozen_primary_component_no_refit",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = extract(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "models_refit": 0, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
