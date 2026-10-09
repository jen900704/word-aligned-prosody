import json
import tempfile
import unittest
from pathlib import Path

from make_ptsd_stop_extraction_closure_v1 import (
    EXPECTED_ALIGNER,
    EXPECTED_CMU,
    EXPECTED_EXTRACTOR,
    EXPECTED_MANIFEST,
    EXPECTED_PARTICIPANTS,
    EXPECTED_PER_SHARD,
    EXPECTED_PROTOCOL,
    EXPECTED_RECORDINGS,
    FROZEN_GIT_COMMIT,
    FROZEN_GIT_TAG,
    build_closure,
)


class PTSDStopClosureTests(unittest.TestCase):
    def make_reports(self, root: Path) -> list[Path]:
        paths = []
        for shard in range(4):
            recordings = []
            for offset in range(EXPECTED_PER_SHARD):
                number = shard * EXPECTED_PER_SHARD + offset
                recordings.append({
                    "recording_id": f"R{number}",
                    "participant_id": f"P{number % EXPECTED_PARTICIPANTS}",
                    "role_assignment_tie": number % 2 == 0,
                    "valid_participant_intervals": 2,
                    "invalid_participant_rows": 0,
                    "empty_participant_rows": 0,
                    "intervals_beyond_audio": 0,
                    "normalized_tokens": 10,
                    "aligned_tokens": 9,
                    "unaligned_tokens": 1,
                    "utterance_alignment_failures": 0,
                    "private_output_sha256": "a" * 64,
                })
            report = {
                "protocol": EXPECTED_PROTOCOL,
                "extractor_version": EXPECTED_EXTRACTOR,
                "n": EXPECTED_PER_SHARD,
                "expected_full_n": EXPECTED_RECORDINGS,
                "labels_loaded": False,
                "outcomes_loaded": False,
                "scientific_results_inspected": False,
                "source_manifest_sha256": EXPECTED_MANIFEST,
                "shard": {"index": shard, "count": 4},
                "resource_hashes": {
                    "cmudict_sha256": EXPECTED_CMU,
                    "alignment_checkpoint_sha256": EXPECTED_ALIGNER,
                },
                "recordings": recordings,
                "total_s": 100.0,
            }
            path = root / f"technical_{shard}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        return paths

    def test_builds_accounted_four_shard_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            closure = build_closure(self.make_reports(Path(directory)))
        self.assertEqual(closure["recordings"], EXPECTED_RECORDINGS)
        self.assertEqual(closure["participants"], EXPECTED_PARTICIPANTS)
        self.assertEqual(closure["aligned_tokens"], EXPECTED_RECORDINGS * 9)
        self.assertEqual(closure["frozen_git_commit"], FROZEN_GIT_COMMIT)
        self.assertEqual(closure["frozen_git_tag"], FROZEN_GIT_TAG)
        self.assertFalse(closure["outcomes_loaded"])

    def test_rejects_duplicate_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_reports(Path(directory))
            with self.assertRaises(ValueError):
                build_closure([paths[0], paths[0], paths[2], paths[3]])

    def test_fails_closed_on_outcome_access(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_reports(Path(directory))
            report = json.loads(paths[0].read_text())
            report["outcomes_loaded"] = True
            paths[0].write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                build_closure(paths)


if __name__ == "__main__":
    unittest.main()
