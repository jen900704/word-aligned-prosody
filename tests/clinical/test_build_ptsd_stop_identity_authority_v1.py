import unittest

from build_ptsd_stop_identity_authority_v1 import build_rows


class IdentityAuthorityTests(unittest.TestCase):
    def test_one_user_multiple_days_keeps_identity_but_blanks_time(self):
        stage = [{"audio_basename": "P100_1_1_code.wav"}]
        authority = [
            {"user_id": "100", "ud_id": "100_1", "AudioFilename": "P100_1_1_code.mp3", "Date": "d1"},
            {"user_id": "100", "ud_id": "100_2", "AudioFilename": "P100_1_1_code.mp3", "Date": "d2"},
        ]
        row = build_rows(stage, authority)[0]
        self.assertEqual(row["authoritative_participant_id"], "P100")
        self.assertEqual(row["authoritative_user_day_id"], "")
        self.assertEqual(row["temporal_match_status"], "ambiguous_excluded_from_longitudinal")

    def test_two_users_for_recording_fails_closed(self):
        stage = [{"audio_basename": "P100_1_1_code.wav"}]
        authority = [
            {"user_id": "100", "ud_id": "100_1", "AudioFilename": "P100_1_1_code.mp3", "Date": "d1"},
            {"user_id": "200", "ud_id": "200_1", "AudioFilename": "P100_1_1_code.mp3", "Date": "d1"},
        ]
        with self.assertRaises(RuntimeError):
            build_rows(stage, authority)


if __name__ == "__main__":
    unittest.main()
