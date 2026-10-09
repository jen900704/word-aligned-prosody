import unittest

from clinical.prepare_daic_gate_licensed_inputs import CORRELATION_FEATURES, build_gate_adapter


class PrepareOutcomeInputsTests(unittest.TestCase):
    def test_maps_duration_name_and_ci(self):
        source = {
            "analysis": "daic_aggregate_reliability_rq2_unblinded",
            "features": [
                {"feature": feature, "status": "pass", "corrected_reliability": 0.8, "ci_95": [0.6, 0.9], "max_robustness_delta": 0.1}
                for feature in (
                    "f0_mean_hz",
                    "f0_robust_range_hz",
                    "energy_db",
                    "syllable_adjusted_log_duration",
                )
            ]
        }
        adapted = build_gate_adapter(source)
        self.assertEqual(tuple(record["feature"] for record in adapted["features"]), CORRELATION_FEATURES)
        self.assertEqual(adapted["features"][-1]["feature"], "duration_log_syllable_residual")
        self.assertEqual(adapted["features"][0]["ci_lower"], 0.6)


if __name__ == "__main__":
    unittest.main()
