import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_ptsd_stop_within_person_pcl_v1 import (
    FEATURE_ORDER,
    outcome_longitudinal_support,
    run,
)


class PTSDStopWithinPersonTests(unittest.TestCase):
    def test_no_pass_returns_before_tokens_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = {
                "corpus": "PTSD-STOP",
                "features": [
                    {"feature": feature, "status": "fail"}
                    for feature in FEATURE_ORDER
                ],
            }
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            result = run(
                gate_path, root / "absent_tokens", root / "absent_identity.csv",
                root / "out.json", "absent_mysql",
            )
        self.assertEqual(result["status"], "not_run_no_gate_passing_features")
        self.assertFalse(result["labels_loaded"])

    def test_detects_nonlongitudinal_outcome_field(self):
        labels = pd.DataFrame({
            "participant_id": ["P1", "P1", "P2", "P2"],
            "PCL_SCORE": [10, 10, 5, 7],
        })
        support = outcome_longitudinal_support(labels)
        self.assertEqual(support["users"], 2)
        self.assertEqual(support["users_with_two_or_more_scores"], 1)
        self.assertEqual(support["minimum_unique_scores_per_user"], 1)
        self.assertEqual(support["maximum_unique_scores_per_user"], 2)


if __name__ == "__main__":
    unittest.main()
