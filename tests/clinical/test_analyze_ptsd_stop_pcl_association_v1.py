import json
import tempfile
import unittest
from pathlib import Path

from analyze_ptsd_stop_pcl_association_v1 import FEATURE_ORDER, run


class PTSDStopAssociationTests(unittest.TestCase):
    def test_no_pass_returns_before_opening_labels(self):
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
                root / "absent_acoustics.csv",
                root / "absent_labels.csv",
                gate_path,
                root / "result.json",
            )
        self.assertEqual(result["status"], "not_run_no_gate_passing_features")
        self.assertFalse(result["labels_loaded"])


if __name__ == "__main__":
    unittest.main()
