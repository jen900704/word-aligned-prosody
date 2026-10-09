import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from unblind_ptsd_stop_aggregate_reliability_v1 import (
    EXPECTED_FEATURES,
    IDENTITY_AUTHORITY_SHA256,
    validate_and_expose,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PTSDStopAggregateUnblindingTests(unittest.TestCase):
    def make_inputs(self, root: Path, corrected: float = 0.80) -> tuple[Path, Path]:
        features = []
        technical_features = []
        for feature in EXPECTED_FEATURES:
            features.append({
                "feature": feature,
                "primary": {
                    "paired_participants": 225,
                    "raw_half_correlation": 0.70,
                    "corrected_reliability": corrected,
                },
                "bootstrap": {
                    "successful_replicates": 1000,
                    "failed_replicates": 0,
                    "ci_95": [0.60, 0.90],
                },
                "salted_balanced_recording_splits": [
                    {
                        "split": split,
                        "status": "complete",
                        "corrected_reliability": corrected,
                        "absolute_delta": 0.01,
                    }
                    for split in range(25)
                ],
                "participant_influence": {
                    "completed": 225,
                    "failed": 0,
                    "max_absolute_delta": 0.01,
                },
                "volume_weighting": {
                    "paired_participants": 225,
                    "corrected_reliability": corrected - 0.01,
                    "absolute_delta": 0.01,
                },
                "tie_exclusion_sensitivity": {
                    "paired_participants": 225,
                    "corrected_reliability": corrected - 0.02,
                    "absolute_delta": 0.02,
                },
                "max_robustness_delta": 0.02,
                "gate": {
                    "status": "pass",
                    "corrected_reliability": corrected,
                    "ci_95": [0.60, 0.90],
                    "max_robustness_delta": 0.02,
                    "technical_failures": [],
                },
            })
            technical_features.append({"feature": feature, "status": "complete"})
        private = root / "aggregate_PRIVATE.json"
        private.write_text(json.dumps({
            "analysis": "ptsd_stop_aggregate_recording_set_reliability_rq2",
            "identity_authority_sha256": IDENTITY_AUTHORITY_SHA256,
            "corpus_support": {"source_participants": 225, "analysis_participants": 219},
            "labels_loaded": False,
            "outcomes_loaded": False,
            "outcome_analysis_performed": False,
            "features": features,
        }), encoding="utf-8")
        technical = root / "aggregate_technical.json"
        technical.write_text(json.dumps({
            "analysis": (
                "ptsd_stop_aggregate_recording_set_reliability_rq2_blind_technical"
            ),
            "private_output_sha256": sha(private),
            "identity_authority_sha256": IDENTITY_AUTHORITY_SHA256,
            "corpus_support": {"source_participants": 225, "analysis_participants": 219},
            "labels_loaded": False,
            "outcomes_loaded": False,
            "scientific_estimates_inspected": False,
            "all_features_complete": True,
            "features": technical_features,
        }), encoding="utf-8")
        return private, technical

    def test_exposes_all_fixed_features_and_licenses_pcl(self):
        with tempfile.TemporaryDirectory() as directory:
            private, technical = self.make_inputs(Path(directory))
            report = validate_and_expose(private, technical)
        self.assertEqual(
            tuple(item["feature"] for item in report["features"]),
            EXPECTED_FEATURES,
        )
        self.assertEqual(report["gate_licensed_features"], list(EXPECTED_FEATURES))
        self.assertTrue(report["pcl_access_licensed"])
        self.assertFalse(report["outcomes_loaded"])

    def test_fails_on_private_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            private, technical = self.make_inputs(Path(directory))
            private.write_text(private.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_and_expose(private, technical)

    def test_fails_on_inconsistent_stored_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            private, technical = self.make_inputs(Path(directory))
            state = json.loads(private.read_text())
            state["features"][0]["gate"]["status"] = "fail"
            private.write_text(json.dumps(state), encoding="utf-8")
            tech = json.loads(technical.read_text())
            tech["private_output_sha256"] = sha(private)
            technical.write_text(json.dumps(tech), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stored gate status"):
                validate_and_expose(private, technical)


if __name__ == "__main__":
    unittest.main()
