import unittest

import numpy as np
import pandas as pd

from esd.esd_msp_style_speaker_stability_v1 import (
    alternative_prompt_map,
    correlation_record,
    primary_prompt_map,
    residualize_once,
    speaker_pairs,
    spearman_brown,
)


class ESDMatchedAggregationTests(unittest.TestCase):
    def test_primary_prompt_split_is_global_and_balanced(self):
        prompts = pd.Series(["003", "001", "002", "004", "003", "001"])
        mapping = primary_prompt_map(prompts)
        self.assertEqual(mapping, {"001": 0, "002": 1, "003": 0, "004": 1})

    def test_alternative_split_is_deterministic_and_balanced(self):
        prompts = pd.Series([str(index) for index in range(11)])
        first = alternative_prompt_map(prompts, 4)
        second = alternative_prompt_map(prompts, 4)
        self.assertEqual(first, second)
        counts = pd.Series(first).value_counts()
        self.assertLessEqual(abs(int(counts[0]) - int(counts[1])), 1)

    def test_spearman_brown_and_correlations(self):
        self.assertAlmostEqual(spearman_brown(0.5), 2 / 3)
        pairs = pd.DataFrame(
            {
                "speaker_id": list("abcdefghi"),
                "half_0": np.arange(9, dtype=float),
                "half_1": np.arange(9, dtype=float) * 2 + 1,
            }
        )
        result = correlation_record(pairs, 9)
        self.assertAlmostEqual(result["raw_pearson"], 1.0)
        self.assertAlmostEqual(result["spearman_brown"], 1.0)
        self.assertAlmostEqual(result["raw_spearman"], 1.0)

    def test_nuisance_model_is_fit_once_across_all_rows(self):
        frame = pd.DataFrame(
            {
                "speaker_id": ["a"] * 6,
                "prompt_id": ["1", "1", "2", "2", "3", "3"],
                "condition": ["Neutral"] * 6,
                "relative_word_position": [0, 1, 0, 1, 0, 1],
                "f0_mean_hz": [10, 12, 10, 12, 10, 12],
            }
        )
        residuals = residualize_once(frame, "f0_mean_hz")
        np.testing.assert_allclose(residuals["residual"], 0.0, atol=1e-10)

    def test_condition_balanced_summary_retains_all_speakers(self):
        rows = []
        conditions = ("Angry", "Happy", "Neutral", "Sad", "Surprise")
        for speaker_index in range(9):
            for condition_index, condition in enumerate(conditions):
                for prompt in range(10):
                    for word in range(10):
                        rows.append(
                            {
                                "speaker_id": str(speaker_index),
                                "prompt_id": str(prompt),
                                "condition": condition,
                                "residual": speaker_index + condition_index / 10 + word / 1000,
                            }
                        )
        residuals = pd.DataFrame(rows)
        pairs, support = speaker_pairs(
            residuals,
            primary_prompt_map(residuals["prompt_id"]),
            condition_balanced=True,
        )
        self.assertEqual(len(pairs), 9)
        self.assertEqual(support["paired_speakers"], 9)


if __name__ == "__main__":
    unittest.main()
