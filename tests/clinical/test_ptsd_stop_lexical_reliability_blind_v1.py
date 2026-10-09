import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ptsd_stop_lexical_reliability_blind_v1 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_ANALYSIS_PARTICIPANTS,
    MIN_LEXICAL_PARTICIPANTS,
    load_compute_decision,
    prepare_feature,
)
from select_ptsd_lexical_precision_mode_v1 import projection, select_mode


class PTSDStopLexicalReliabilityTests(unittest.TestCase):
    def synthetic_frame(self) -> pd.DataFrame:
        rows = []
        for participant in range(EXPECTED_ANALYSIS_PARTICIPANTS):
            for word in ("alpha", "beta"):
                for repeat in range(2):
                    rows.append({
                        "participant_id": f"P{participant}",
                        "normalized_word": word,
                        "f0_mean_hz": participant + repeat / 10,
                        "f0_robust_range_hz": participant + repeat / 10,
                        "energy_db": participant + repeat / 10,
                        "log_word_duration": 1.0 + repeat / 10,
                        "authoritative_syllable_count": 2,
                    })
        return pd.DataFrame(rows)

    def test_prepare_feature_keeps_repeated_cells_and_all_participants(self):
        work, support = prepare_feature(self.synthetic_frame(), "f0_mean_hz")
        self.assertEqual(
            support["participants"], EXPECTED_ANALYSIS_PARTICIPANTS
        )
        self.assertEqual(
            support["matched_participant_word_cells"],
            EXPECTED_ANALYSIS_PARTICIPANTS * 2,
        )
        self.assertEqual(len(work), EXPECTED_ANALYSIS_PARTICIPANTS * 4)
        self.assertTrue(np.isfinite(work["y"]).all())

    def test_rejects_fewer_than_fifty_supported_participants(self):
        frame = self.synthetic_frame()
        keep = {f"P{i}" for i in range(MIN_LEXICAL_PARTICIPANTS - 1)}
        frame = frame.loc[frame["participant_id"].isin(keep)]
        with self.assertRaisesRegex(ValueError, "fewer than 50"):
            prepare_feature(frame, "f0_mean_hz")

    def test_bootstrap_contract_is_inherited(self):
        self.assertEqual(BOOTSTRAP_REPLICATES, 1000)
        self.assertEqual(BOOTSTRAP_SEED, 13)

    def test_compute_decision_contract(self):
        decision = {
            "protocol": "ptsd_stop_precision_within_person_amendment_v1",
            "mode": "reduced_precision_B200",
            "bootstrap_target_per_feature": 200,
            "bootstrap_minimum_success": 190,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(decision), encoding="utf-8")
            mode, target, minimum, digest = load_compute_decision(path)
        self.assertEqual((mode, target, minimum), (
            "reduced_precision_B200", 200, 190
        ))
        self.assertEqual(len(digest), 64)

    def test_compute_decision_rejects_mode_target_mismatch(self):
        decision = {
            "protocol": "ptsd_stop_precision_within_person_amendment_v1",
            "mode": "full_precision_B1000",
            "bootstrap_target_per_feature": 200,
            "bootstrap_minimum_success": 190,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(decision), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target mismatch"):
                load_compute_decision(path)

    def test_projection_and_three_mode_ladder(self):
        full = select_mode(10.0, 10.0, 100.0, 10_000.0, 25_000.0)
        self.assertEqual(full["mode"], "full_precision_B1000")
        reduced_need = projection(10.0, 10.0, 4, 200)
        reduced = select_mode(
            10.0, 10.0, 100.0, 10_000.0, reduced_need + 1.0
        )
        self.assertEqual(reduced["mode"], "reduced_precision_B200")
        self.assertEqual(reduced["safe_parallel_workers"], 4)
        point = select_mode(10.0, 10.0, 100.0, 10_000.0, 1.0)
        self.assertEqual(point["mode"], "point_only_compute_limited")


if __name__ == "__main__":
    unittest.main()
