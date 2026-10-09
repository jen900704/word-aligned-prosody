import json
import tempfile
import unittest
from pathlib import Path

from clinical.make_daic_extraction_closure import (
    EXPECTED_ALIGNER,
    EXPECTED_CMU,
    EXPECTED_EXTRACTOR,
    EXPECTED_N,
    EXPECTED_PROTOCOL,
    build_closure,
)


class ClosureTests(unittest.TestCase):
    def make_report(self, root: Path) -> Path:
        participants = [
            {
                "participant_id": str(index),
                "participant_utterances": 2,
                "normalized_tokens": 10,
                "aligned_tokens": 9,
                "unaligned_tokens": 1,
                "utterance_alignment_failures": 0,
            }
            for index in range(EXPECTED_N)
        ]
        report = {
            "protocol": EXPECTED_PROTOCOL,
            "extractor_version": EXPECTED_EXTRACTOR,
            "n": EXPECTED_N,
            "labels_loaded": False,
            "scientific_results_inspected": False,
            "resource_hashes": {
                "cmudict_sha256": EXPECTED_CMU,
                "alignment_checkpoint_sha256": EXPECTED_ALIGNER,
            },
            "participants": participants,
            "runtime": {},
            "total_s": 100.0,
            "median_participant_s": 1.0,
        }
        path = root / "technical_report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_builds_accounted_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            closure = build_closure(self.make_report(Path(directory)))
        self.assertEqual(closure["normalized_tokens"], EXPECTED_N * 10)
        self.assertEqual(closure["aligned_tokens"], EXPECTED_N * 9)
        self.assertFalse(closure["labels_loaded"])

    def test_fails_closed_on_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_report(Path(directory))
            report = json.loads(path.read_text())
            report["labels_loaded"] = True
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                build_closure(path)


if __name__ == "__main__":
    unittest.main()
