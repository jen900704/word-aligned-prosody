import unittest

import numpy as np
import pandas as pd

from prepare_ptsd_stop_gate_licensed_inputs_v1 import (
    CORRELATION_FEATURES,
    EXPECTED_LABEL_PARTICIPANTS,
    validate_labeled_acoustic_support,
)


def fixtures():
    label_ids = [f"P{i:03d}" for i in range(EXPECTED_LABEL_PARTICIPANTS)]
    labels = pd.DataFrame({"participant_id": label_ids})
    acoustic_ids = label_ids + ["P999"]
    acoustics = pd.DataFrame({"participant_id": acoustic_ids})
    for feature in CORRELATION_FEATURES:
        acoustics[feature] = 0.0
    acoustics.loc[acoustics["participant_id"].eq("P999"), CORRELATION_FEATURES[0]] = np.nan
    return acoustics, labels


class LabeledAcousticSupportTests(unittest.TestCase):
    def test_allows_missing_support_outside_pcl_cohort(self):
        acoustics, labels = fixtures()
        validate_labeled_acoustic_support(acoustics, labels)

    def test_rejects_missing_support_inside_pcl_cohort(self):
        acoustics, labels = fixtures()
        acoustics.loc[
            acoustics["participant_id"].eq(labels.iloc[0]["participant_id"]),
            CORRELATION_FEATURES[0],
        ] = np.nan
        with self.assertRaisesRegex(ValueError, "labeled participant"):
            validate_labeled_acoustic_support(acoustics, labels)


if __name__ == "__main__":
    unittest.main()
