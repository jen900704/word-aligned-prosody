#!/usr/bin/env python3
"""Amendment O: aggregation curve on PTSD-STOP.

Reuses the frozen PTSD-STOP preprocessing: identity authority, frozen
nuisance models, frozen deterministic balanced whole-recording halves.
No label is read; only curve values leave the server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptsd_stop_aggregate_reliability_blind_v1 as frozen

K_GRID = (5, 10, 20, 50, 100, 200)
DRAWS = 200
SEED = 13


def spearman_brown(r: float) -> float:
    return 2.0 * r / (1.0 + r)


def one_way_icc(groups):
    sizes = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups], dtype=float)
    total = sizes.sum()
    grand = float((sizes * means).sum() / total)
    n = len(groups)
    ms_b = float((sizes * (means - grand) ** 2).sum() / (n - 1))
    ms_w = sum(float(((g - g.mean()) ** 2).sum()) for g in groups) / (total - n)
    n0 = float((total - (sizes ** 2).sum() / total) / (n - 1))
    den = ms_b + (n0 - 1.0) * ms_w
    return {"icc1": float((ms_b - ms_w) / den) if den > 0 else float("nan"),
            "groups": n, "tokens": int(total), "n_zero": n0}


def prediction(icc: float, k: int) -> float:
    return k * icc / (1.0 + (k - 1) * icc)


def summarise(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"draws": 0}
    return {"draws": int(a.size), "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}


def observed(halves, eligible, k, rng):
    usable = [p for p in eligible if len(halves[p][0]) >= k and len(halves[p][1]) >= k]
    if len(usable) < frozen.MIN_PAIRED_PARTICIPANTS:
        return {"tokens_per_half": k, "eligible_participants": len(usable),
                "status": "support_below_minimum"}
    raw_mean, raw_med = [], []
    for _ in range(DRAWS):
        m0, m1, d0, d1 = [], [], [], []
        for p in usable:
            a = rng.choice(halves[p][0], size=k, replace=False)
            b = rng.choice(halves[p][1], size=k, replace=False)
            m0.append(a.mean()); m1.append(b.mean())
            d0.append(np.median(a)); d1.append(np.median(b))
        for store, x, y in ((raw_mean, m0, m1), (raw_med, d0, d1)):
            x = np.asarray(x, float); y = np.asarray(y, float)
            if x.std() <= 0 or y.std() <= 0:
                continue
            store.append(float(np.corrcoef(x, y)[0, 1]))
    return {"tokens_per_half": k, "eligible_participants": len(usable),
            "reliability_of_a_k_token_mean": summarise(raw_mean),
            "reliability_of_a_k_token_median": summarise(raw_med),
            "status": "complete"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", required=True, type=Path)
    ap.add_argument("--identity-authority", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    frame = frozen.apply_identity_authority(frozen.load_tokens(args.token_dir),
                                            args.identity_authority)
    assignment = frozen.recording_assignment(frame, frozen.PRIMARY_SPLIT_SALT)
    rng = np.random.Generator(np.random.PCG64(SEED))

    features = []
    for feature in frozen.FEATURES:
        work = frozen.feature_frame(frame, feature, assignment)
        residuals = frozen.residualize_by_half(work, feature)
        halves = {}
        for (pid, half), grp in residuals.groupby(["participant_id", "half"], sort=True):
            halves.setdefault(str(pid), {})[int(half)] = grp["residual"].to_numpy(float)
        eligible = [p for p, s in halves.items()
                    if 0 in s and 1 in s
                    and len(s[0]) >= frozen.MIN_TOKENS_PER_HALF
                    and len(s[1]) >= frozen.MIN_TOKENS_PER_HALF]
        eligible.sort()
        icc = one_way_icc([np.concatenate([halves[p][0], halves[p][1]]) for p in eligible])
        curve = []
        for k in K_GRID:
            rec = observed(halves, eligible, k, rng)
            if rec["status"] == "complete":
                rec["prediction_at_k"] = prediction(icc["icc1"], k)
                rec["gap_at_k_prediction_minus_observed"] = (
                    rec["prediction_at_k"] - rec["reliability_of_a_k_token_mean"]["median"])
            curve.append(rec)
        features.append({
            "feature": feature, "token_level_icc": icc,
            "eligible_participants": len(eligible),
            "median_tokens_per_participant_half": float(
                np.median([len(halves[p][0]) for p in eligible])),
            "curve": curve,
        })
        del halves, work, residuals

    payload = {
        "analysis": "aggregation_curve_O_ptsd_v1",
        "amendment": "PTSD_aggregation_curve_amendment_O_20260905.md",
        "corpus": "PTSD-STOP", "labels_loaded": False,
        "outcome_analysis_performed": False,
        "split": "frozen deterministic balanced whole-recording halves",
        "draws_per_k": DRAWS, "seed": SEED, "generator": "PCG64",
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"features": len(features)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
