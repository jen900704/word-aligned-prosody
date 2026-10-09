#!/usr/bin/env python3
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import esd_contract as contract
import esd_stage2_adapter as adapter
import stage2b_full as full


def real_order_rows():
    rows = []
    for speaker in contract.ELIGIBLE_ESD_SPEAKERS:
        for prompt in range(1, 351):
            for condition in ("Angry", "Happy", "Neutral", "Sad", "Surprise"):
                rows.append({"sample_id": f"{speaker}:{condition}:{prompt:03d}", "speaker_id": speaker})
    return rows


class Stage2BFullTests(unittest.TestCase):
    def test_exact_32_shard_balance_and_frozen_plan_sha(self):
        rows = real_order_rows()
        self.assertEqual(len(rows), 15750)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(adapter, "read_manifest", return_value=rows):
            report = full.create_plan(Path("synthetic.csv"), Path(td))
        self.assertEqual(report["rows_per_shard"], {str(i): 493 if i < 6 else 492 for i in range(32)})
        self.assertEqual(report["total_rows"], 15750)
        self.assertEqual(report["duplicate_assignments"], 0)
        self.assertEqual(report["missing_assignments"], 0)
        self.assertEqual(report["shard_manifest_sha256"], "9b21e12bed480c554779f62cc9f3bbbedde895b6ff34e3ca2cf1c284a2420793")

    def small_fixture(self, root):
        speakers = contract.ELIGIBLE_ESD_SPEAKERS
        rows = [{"sample_id": f"sample:{i:02d}", "speaker_id": speakers[i % len(speakers)]} for i in range(32)]
        with mock.patch.object(full, "EXPECTED_UTTERANCES", 32), mock.patch.object(adapter, "read_manifest", return_value=rows):
            full.create_plan(Path("synthetic.csv"), root)
        (root / "runtime_verification_SAFE.json").write_text(json.dumps({"all_checks_passed": True}))
        return rows

    @staticmethod
    def write_unit(path, sample_id, duplicate=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {field: "" for field in adapter.FIELDS}
        row.update(sample_id=sample_id, word_index="0", alignment_status="aligned", feature_status="complete", adapter_version=adapter.ADAPTER_VERSION)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, adapter.FIELDS); writer.writeheader(); writer.writerow(row)
            if duplicate: writer.writerow(row)

    def test_completed_unit_is_skipped_without_audio_alignment_or_acoustics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = self.small_fixture(root)
            self.write_unit(full._unit_path(root / "shards/shard_00", rows[0]["sample_id"]), rows[0]["sample_id"])
            with mock.patch.object(full, "EXPECTED_UTTERANCES", 32), mock.patch.object(adapter, "read_manifest", return_value=rows), mock.patch.object(adapter, "verify_cmudict", side_effect=AssertionError("resource/audio path invoked")), mock.patch.object(full, "FrozenWordAligner", side_effect=AssertionError("alignment invoked")), mock.patch.object(adapter, "process_utterance", side_effect=AssertionError("acoustics invoked")):
                marker = full.run_shard(Path("synthetic.csv"), root, Path("cmu"), Path("model"), 0)
            self.assertTrue(marker["shard_complete"])
            self.assertEqual(marker["valid_utterances"], 1)

    def populate_complete_shards(self, root, rows, duplicate=False):
        plan_report = json.loads((root / full.PLAN_REPORT_NAME).read_text())
        record = full.contract_record(); digest = full._json_sha(record)
        for shard_id, row in enumerate(rows):
            shard_root = root / "shards" / f"shard_{shard_id:02d}"
            self.write_unit(full._unit_path(shard_root, row["sample_id"]), row["sample_id"], duplicate and shard_id == 0)
            contract.write_json(shard_root / full.COMPLETION_NAME, {
                "completion_version": "esd_stage2b_shard_completion_v1", "shard_id": shard_id,
                "assigned_utterances": 1, "valid_utterances": 1, "shard_complete": True,
                "shard_manifest_sha256": plan_report["shard_manifest_sha256"],
                "contract": record, "contract_sha256": digest,
                "gate_e_evaluated": False, "reliability_evaluated": False,
            })

    def test_aggregation_requires_complete_unique_schema_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = self.small_fixture(root); self.populate_complete_shards(root, rows)
            with mock.patch.object(full, "EXPECTED_UTTERANCES", 32), mock.patch.object(adapter, "read_manifest", return_value=rows):
                qa = full.aggregate(Path("synthetic.csv"), root)
            self.assertTrue(qa["qa_pass"]); self.assertTrue(qa["all_32_shards_complete"])
            self.assertEqual(qa["represented_utterances"], 32); self.assertEqual(qa["duplicate_token_identities"], 0)
            self.assertFalse(qa["gate_e_evaluated"]); self.assertFalse(qa["reliability_evaluated"])
            self.assertTrue((root / "acoustics_raw" / full.AGGREGATE_NAME).is_file())

    def test_aggregation_rejects_duplicate_token_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rows = self.small_fixture(root); self.populate_complete_shards(root, rows, duplicate=True)
            with mock.patch.object(full, "EXPECTED_UTTERANCES", 32), mock.patch.object(adapter, "read_manifest", return_value=rows):
                with self.assertRaisesRegex(contract.ContractError, "identity validation"):
                    full.aggregate(Path("synthetic.csv"), root)

    def test_static_array_and_no_inference_or_clinical_paths(self):
        here = Path(full.__file__).parent
        slurm = (here / "slurm_stage2b_full.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-31", slurm); self.assertIn("#SBATCH --partition=gpu", slurm); self.assertIn("#SBATCH --gres=gpu:1", slurm)
        combined = (Path(full.__file__).read_text() + slurm).casefold()
        for forbidden in ("reml", "bootstrap", "leave_one_speaker_out", "lexical_sensitivity", "gate_e_pass", "clinical", "msp"):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__": unittest.main()
