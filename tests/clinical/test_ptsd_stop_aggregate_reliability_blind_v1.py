import unittest
import tempfile
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ptsd_stop_aggregate_reliability_blind_v1 import (
    GATE_CI_LOWER,
    GATE_MAX_DELTA,
    GATE_RELIABILITY,
    PRIMARY_SPLIT_SALT,
    SOURCE_COLUMNS,
    apply_identity_authority,
    load_token_paths,
    participant_pairs,
    recording_assignment,
)


class PTSDStopAggregateReliabilityTests(unittest.TestCase):
    def test_identity_authority_reassigns_before_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity_PRIVATE.csv"
            pd.DataFrame([
                {
                    "recording_id": "R1",
                    "source_participant_id": "P1",
                    "authoritative_participant_id": "P9",
                    "identity_match_status": "one_authoritative_user",
                },
                {
                    "recording_id": "R2",
                    "source_participant_id": "P2",
                    "authoritative_participant_id": "P9",
                    "identity_match_status": "one_authoritative_user",
                },
            ]).to_csv(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = pd.DataFrame({
                "participant_id": ["P1", "P2"],
                "recording_id": ["R1", "R2"],
            })
            mapped = apply_identity_authority(
                frame, path, digest, expected_recordings=2,
                expected_participants=1,
            )
        self.assertEqual(set(mapped["participant_id"]), {"P9"})
        self.assertEqual(mapped.attrs["analysis_participants"], 1)

    def test_header_only_recording_is_counted_not_misread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populated = pd.DataFrame([{
                "participant_id": "P1",
                "recording_id": "R1",
                "role_assignment_tie": False,
                "utterance_index": 0,
                "word_index": 0,
                "normalized_word": "hello",
                "alignment_status": "aligned",
                "log_word_duration": 0.0,
                "f0_mean_hz": 100.0,
                "f0_robust_range_hz": 20.0,
                "energy_db": -10.0,
                "authoritative_syllable_count": 2,
            }], columns=SOURCE_COLUMNS)
            populated.to_csv(root / "R1.csv", index=False)
            pd.DataFrame(columns=SOURCE_COLUMNS).to_csv(root / "R2.csv", index=False)
            frame = load_token_paths(
                [root / "R1.csv", root / "R2.csv"], 2, 1
            )
        self.assertEqual(frame.attrs["token_files"], 2)
        self.assertEqual(frame.attrs["empty_token_files"], 1)
        self.assertEqual(frame["recording_id"].nunique(), 1)

    def test_opaque_recording_suffix_may_contain_pcl_letters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = "R_qG5pCln1"
            frame = pd.DataFrame([{
                "participant_id": "P1",
                "recording_id": recording,
                "role_assignment_tie": False,
                "utterance_index": 0,
                "word_index": 0,
                "normalized_word": "hello",
                "alignment_status": "aligned",
                "log_word_duration": 0.0,
                "f0_mean_hz": 100.0,
                "f0_robust_range_hz": 20.0,
                "energy_db": -10.0,
                "authoritative_syllable_count": 2,
            }], columns=SOURCE_COLUMNS)
            path = root / f"{recording}.csv"
            frame.to_csv(path, index=False)
            loaded = load_token_paths([path], 1, 1)
        self.assertEqual(loaded["recording_id"].iloc[0], recording)

    def test_recording_assignment_is_deterministic_and_balanced(self):
        rows = []
        for participant in ("a", "b"):
            for recording in range(7):
                for token in range(3):
                    rows.append({
                        "participant_id": participant,
                        "recording_id": f"{participant}-{recording}",
                        "token": token,
                    })
        frame = pd.DataFrame(rows)
        first = recording_assignment(frame, PRIMARY_SPLIT_SALT)
        second = recording_assignment(frame, PRIMARY_SPLIT_SALT)
        np.testing.assert_array_equal(first, second)
        for participant in ("a", "b"):
            unique = frame.loc[frame["participant_id"].eq(participant), ["recording_id"]].copy()
            unique["half"] = first.loc[unique.index]
            counts = unique.drop_duplicates()["half"].value_counts()
            self.assertLessEqual(
                abs(int(counts.get(0, 0)) - int(counts.get(1, 0))), 1
            )

    def test_recording_median_volume_mode(self):
        rows = []
        for participant, offset in (("a", 0.0), ("b", 1.0)):
            for half in (0, 1):
                for recording in ("x", "y"):
                    for token in range(60):
                        rows.append({
                            "participant_id": participant,
                            "half": half,
                            "recording_id": f"{participant}-{half}-{recording}",
                            "residual": offset + half + token / 1000,
                        })
        pairs = participant_pairs(pd.DataFrame(rows), "recording_medians")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(set(pairs.columns), {"participant_id", "half_0", "half_1"})

    def test_frozen_gate_constants(self):
        self.assertEqual(GATE_RELIABILITY, 0.60)
        self.assertEqual(GATE_CI_LOWER, 0.40)
        self.assertEqual(GATE_MAX_DELTA, 0.15)


if __name__ == "__main__":
    unittest.main()
