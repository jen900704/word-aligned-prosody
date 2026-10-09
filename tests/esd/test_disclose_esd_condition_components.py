import unittest

from esd.disclose_esd_condition_components import (
    CONDITION_ORDER,
    FEATURE_ORDER,
    extract,
)


class DiscloseConditionComponentsTests(unittest.TestCase):
    def make_report(self):
        features = []
        for feature_index, feature in enumerate(FEATURE_ORDER):
            estimates = []
            values = []
            for condition_index, condition in enumerate(CONDITION_ORDER):
                value = 0.01 * (feature_index + 1) + 0.001 * condition_index
                values.append(value)
                estimates.append(
                    {
                        "condition": condition,
                        "repeatability": value,
                        "variance_components": {
                            "S": 1.0 + condition_index,
                            "W": 2.0 + condition_index,
                            "SW": 3.0 + condition_index,
                            "e": 4.0 + condition_index,
                        },
                    }
                )
            features.append(
                {
                    "feature": feature,
                    "primary": {
                        "condition_estimates": estimates,
                        "equal_condition_mean": sum(values) / len(values),
                    },
                }
            )
        return {"features": features}

    def test_extracts_fixed_order_repeatability_and_variances(self):
        rows = extract(self.make_report())
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["feature"], FEATURE_ORDER[0])
        self.assertEqual(rows[0]["condition"], CONDITION_ORDER[0])
        self.assertEqual(rows[-1]["feature"], FEATURE_ORDER[-1])
        self.assertEqual(rows[-1]["condition"], CONDITION_ORDER[-1])
        self.assertEqual(rows[0]["variance_S"], 1.0)
        self.assertEqual(rows[0]["variance_W"], 2.0)
        self.assertEqual(rows[0]["variance_SW"], 3.0)
        self.assertEqual(rows[0]["variance_e"], 4.0)

    def test_rejects_missing_variance_component(self):
        report = self.make_report()
        del report["features"][0]["primary"]["condition_estimates"][0][
            "variance_components"
        ]["e"]
        with self.assertRaisesRegex(ValueError, "variance-component set mismatch"):
            extract(report)


if __name__ == "__main__":
    unittest.main()
