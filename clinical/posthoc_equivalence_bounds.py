#!/usr/bin/env python3
"""POST HOC equivalence bounds for the clinical association families (paper Sec. 5.4).

Two one-sided tests (TOST) at alpha = .05 on Spearman correlations, using the
Fisher z transform with the Bonett-Wright standard error for Spearman's rho,
SE = sqrt((1 + rho^2 / 2) / (n - 3)). For each feature the smallest symmetric
bound that the test excludes is tanh(|z| + z_.95 * SE); a family's bound is the
largest of its four features.

This file was written for the public release. It recomputes the bounds from the
aggregate correlations reported in the paper (Table 2 and the DAIC-WOZ result
file), so it needs no participant-level data. The within-person family is not
covered here: the paper bounds it with its own cluster bootstrap.
"""
from __future__ import annotations

import json
import math
from statistics import NormalDist

Z95 = NormalDist().inv_cdf(0.95)

# Spearman rho per feature (mean F0, F0 range, energy, log duration) and N.
FAMILIES = {
    # results/daic_woz/daic_phq8_correlation_unblinded_SAFE.json
    "DAIC-WOZ between-person (PHQ-8)": (189, [0.044395480012682134, 0.00014263567654819193,
                                               -0.034969396609322814, -0.00021038023698463399]),
    # paper Table 2, between-person column
    "PTSD-STOP between-person (PCL)": (213, [0.151, 0.032, 0.051, 0.056]),
    # paper Table 2, change column
    "PTSD-STOP change (PCL)": (178, [0.047, 0.110, -0.109, -0.013]),
}


def bonett_wright_se(rho: float, n: int) -> float:
    return math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))


def smallest_excluded_bound(rho: float, n: int) -> float:
    return math.tanh(abs(math.atanh(rho)) + Z95 * bonett_wright_se(rho, n))


def main() -> None:
    out = {}
    for family, (n, rhos) in FAMILIES.items():
        per_feature = [smallest_excluded_bound(r, n) for r in rhos]
        out[family] = {"n": n, "per_feature_bound": [round(b, 4) for b in per_feature],
                       "family_bound": round(max(per_feature), 4)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
