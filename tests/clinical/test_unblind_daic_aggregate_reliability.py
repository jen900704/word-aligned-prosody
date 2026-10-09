import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from clinical.unblind_daic_aggregate_reliability import EXPECTED_FEATURES, validate_and_expose


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnblindTests(unittest.TestCase):
    def make_files(self, root: Path, status: str = "pass") -> tuple[Path, Path]:
        features = []
        for feature in EXPECTED_FEATURES:
            features.append(
                {
                    "feature": feature,
                    "primary": {"paired_participants": 100, "raw_half_correlation": 0.5, "corrected_reliability": 2 / 3},
                    "bootstrap": {"successful_replicates": 1000, "failed_replicates": 0},
                    "gate": {"status": status, "corrected_reliability": 2 / 3, "ci_95": [0.5, 0.8], "max_robustness_delta": 0.1, "technical_failures": []},
                }
            )
        private = root / "reliability_PRIVATE.json"
        private.write_text(json.dumps({"analysis": "daic_aggregate_reliability_rq2", "labels_loaded": False, "outcome_analysis_performed": False, "features": features}))
        technical = root / "reliability_TECHNICAL.json"
        technical.write_text(json.dumps({"analysis": "daic_aggregate_reliability_rq2_blind_technical", "labels_loaded": False, "scientific_estimates_inspected": False, "all_features_complete": True, "private_output_sha256": sha(private), "features": [{"feature": feature, "status": "complete"} for feature in EXPECTED_FEATURES]}))
        return private, technical

    def test_exposes_consistent_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            private, technical = self.make_files(Path(directory))
            report = validate_and_expose(private, technical)
        self.assertEqual(len(report["features"]), 4)
        self.assertTrue(all(record["status"] == "pass" for record in report["features"]))

    def test_rejects_status_threshold_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            private, technical = self.make_files(Path(directory), status="fail")
            with self.assertRaises(ValueError):
                validate_and_expose(private, technical)


if __name__ == "__main__":
    unittest.main()
