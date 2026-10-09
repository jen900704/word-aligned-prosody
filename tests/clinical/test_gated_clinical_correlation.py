from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from clinical.gated_clinical_correlation import FEATURE_ORDER, benjamini_hochberg, run_analysis


def gate_records(passing=()):
    return {
        "corpus": "SYNTHETIC",
        "features": [
            {
                "feature": feature,
                "status": "pass" if feature in passing else "fail",
                "corrected_reliability": 0.80 if feature in passing else 0.10,
                "ci_lower": 0.60 if feature in passing else 0.00,
                "max_robustness_delta": 0.05,
            }
            for feature in FEATURE_ORDER
        ],
    }


class FrozenCorrelationTests(unittest.TestCase):
    def test_bh_known_values(self):
        self.assertTrue(
            np.allclose(benjamini_hochberg([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
        )

    def test_no_pass_does_not_touch_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "gate.json"
            gate.write_text(json.dumps(gate_records()), encoding="utf-8")
            output = root / "result.json"
            result = run_analysis(
                root / "missing_acoustics.csv",
                root / "forbidden_missing_labels.csv",
                gate,
                output,
                "SYNTHETIC",
                "participant_id",
                "outcome",
                0.870,
            )
            self.assertFalse(result["labels_loaded"])
            self.assertEqual(result["status"], "not_run_no_gate_passing_features")

    def test_only_passing_feature_is_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n = 40
            identifiers = [f"p{i:03d}" for i in range(n)]
            acoustics = pd.DataFrame({"participant_id": identifiers})
            for index, feature in enumerate(FEATURE_ORDER):
                acoustics[feature] = np.arange(n, dtype=float) + index
            labels = pd.DataFrame(
                {"participant_id": identifiers, "outcome": np.arange(n, dtype=float)}
            )
            acoustics.to_csv(root / "acoustics.csv", index=False)
            labels.to_csv(root / "labels.csv", index=False)
            (root / "gate.json").write_text(
                json.dumps(gate_records(("f0_mean_hz",))), encoding="utf-8"
            )
            result = run_analysis(
                root / "acoustics.csv",
                root / "labels.csv",
                root / "gate.json",
                root / "result.json",
                "SYNTHETIC",
                "participant_id",
                "outcome",
                0.870,
            )
            self.assertTrue(result["labels_loaded"])
            self.assertEqual([item["feature"] for item in result["features"]], ["f0_mean_hz"])
            self.assertAlmostEqual(result["features"][0]["spearman_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()

