#!/usr/bin/env python3
"""Amendment J2: express both ESD estimands from the frozen variance components.

Refits nothing.  Reads the disclosed frozen per-condition components
(S, W, SW, e) and writes the two closed forms plus the token counts they
imply.  Any mismatch against the frozen R_c is reported, not repaired.
"""
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
K_GRID = (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
TARGETS = (0.7, 0.8, 0.9, 0.95)


def repeatability(sw: float, e: float) -> float:
    """Cross-context repeatability of one word token: no speaker term."""
    denominator = sw + e
    if denominator <= 0:
        raise ValueError("non-positive repeatability denominator")
    return sw / denominator


def speaker_location(s: float, sw: float, e: float, k: int) -> float:
    """Reliability of a speaker mean over k word tokens."""
    if k < 1:
        raise ValueError("k must be at least one token")
    denominator = s + (sw + e) / k
    if denominator <= 0:
        raise ValueError("non-positive speaker-location denominator")
    return s / denominator


def tokens_for(s: float, sw: float, e: float, target: float) -> float:
    """Smallest k with speaker-location reliability at or above target."""
    if s <= 0:
        return math.inf
    if not 0.0 < target < 1.0:
        raise ValueError("target outside the open unit interval")
    return target * (sw + e) / ((1.0 - target) * s)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = {(row["feature"], row["condition"]) for row in rows}
    expected = {(f, c) for f in FEATURE_ORDER for c in CONDITION_ORDER}
    if seen != expected:
        raise ValueError("frozen component table does not match the frozen grid")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    rows = load_rows(args.components)
    records = []
    mismatches = []
    for feature in FEATURE_ORDER:
        for condition in CONDITION_ORDER:
            row = next(
                r for r in rows if r["feature"] == feature and r["condition"] == condition
            )
            s = float(row["variance_S"])
            w = float(row["variance_W"])
            sw = float(row["variance_SW"])
            e = float(row["variance_e"])
            frozen_rc = float(row["R_c"])
            derived_rc = repeatability(sw, e)
            if not math.isclose(derived_rc, frozen_rc, rel_tol=0.0, abs_tol=args.tolerance):
                mismatches.append(
                    {
                        "feature": feature,
                        "condition": condition,
                        "frozen_R_c": frozen_rc,
                        "closed_form_R_c": derived_rc,
                    }
                )
            records.append(
                {
                    "feature": feature,
                    "condition": condition,
                    "variance_S": s,
                    "variance_W": w,
                    "variance_SW": sw,
                    "variance_e": e,
                    "frozen_R_c": frozen_rc,
                    "closed_form_R_c": derived_rc,
                    "speaker_share_of_token_variance": s / (s + w + sw + e),
                    "curve": {
                        str(k): speaker_location(s, sw, e, k) for k in K_GRID
                    },
                    "tokens_for_target": {
                        str(t): tokens_for(s, sw, e, t) for t in TARGETS
                    },
                    "repeatability_ceiling_note": (
                        "R_c does not depend on k; collecting more words cannot raise it."
                    ),
                }
            )

    payload = {
        "analysis": "esd_component_algebra_J2_v1",
        "amendment": "MEASUREMENT_MODEL_amendment_J_20260904.md",
        "models_refit": 0,
        "source_table": str(args.components),
        "closed_forms": {
            "cross_context_repeatability": "R_c = var_SW / (var_SW + var_e)",
            "speaker_location_at_k": "R(k) = var_S / (var_S + (var_SW + var_e) / k)",
            "tokens_for_target": "k* = t (var_SW + var_e) / ((1 - t) var_S)",
        },
        "closed_form_check": {
            "tolerance": args.tolerance,
            "mismatches": mismatches,
            "status": "reproduced" if not mismatches else "discrepancy_reported",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "mismatches": len(mismatches)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
