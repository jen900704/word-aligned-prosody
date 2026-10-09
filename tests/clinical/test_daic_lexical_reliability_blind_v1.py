import unittest

import numpy as np
import pandas as pd

from clinical.daic_lexical_reliability_blind_v1 import (
    bootstrap_draws,
    duration_fixed_effect_residual,
)


class LexicalReliabilityTests(unittest.TestCase):
    def test_duration_residual_removes_participant_intercepts_and_syllable_slope(self):
        rows = []
        for participant, intercept in (("a", 1.0), ("b", 3.0)):
            for syllables, noise in ((1, -0.2), (2, 0.1), (3, 0.1)):
                rows.append(
                    {
                        "participant_id": participant,
                        "log_word_duration": intercept + 0.5 * syllables + noise,
                        "authoritative_syllable_count": syllables,
                    }
                )
        residual = duration_fixed_effect_residual(pd.DataFrame(rows))
        self.assertTrue(np.isfinite(residual).all())
        for participant in ("a", "b"):
            mask = pd.Series([row["participant_id"] == participant for row in rows])
            self.assertAlmostEqual(float(residual.loc[mask].mean()), 0.0, places=12)

    def test_bootstrap_draws_are_fixed_pcg64(self):
        participants = ("a", "b", "c")
        first = bootstrap_draws(participants)
        second = bootstrap_draws(participants)
        self.assertEqual(first.shape, (1000, 3))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
