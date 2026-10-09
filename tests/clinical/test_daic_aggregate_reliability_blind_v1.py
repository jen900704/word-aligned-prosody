import unittest

import numpy as np
import pandas as pd

from clinical.daic_aggregate_reliability_blind_v1 import (
    GATE_CI_LOWER,
    GATE_MAX_DELTA,
    GATE_RELIABILITY,
    salted_assignment,
    spearman_brown,
)


class AggregateReliabilityTests(unittest.TestCase):
    def test_spearman_brown(self):
        self.assertAlmostEqual(spearman_brown(0.5), 2 / 3)
        self.assertAlmostEqual(spearman_brown(0.0), 0.0)

    def test_balanced_salted_assignment(self):
        rows = []
        for participant in ("a", "b"):
            for utterance in range(7):
                rows.append({"participant_id": participant, "utterance_index": utterance})
        frame = pd.DataFrame(rows)
        first = salted_assignment(frame, 0)
        second = salted_assignment(frame, 0)
        np.testing.assert_array_equal(first, second)
        for participant in ("a", "b"):
            counts = first.loc[frame["participant_id"].eq(participant)].value_counts()
            self.assertLessEqual(abs(int(counts.get(0, 0)) - int(counts.get(1, 0))), 1)

    def test_frozen_gate_constants(self):
        self.assertEqual(GATE_RELIABILITY, 0.60)
        self.assertEqual(GATE_CI_LOWER, 0.40)
        self.assertEqual(GATE_MAX_DELTA, 0.15)


if __name__ == "__main__":
    unittest.main()
