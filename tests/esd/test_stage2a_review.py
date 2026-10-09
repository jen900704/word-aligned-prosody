#!/usr/bin/env python3
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import esd_stage2_adapter as adapter
import review_stage2a_smoke as review
from esd_contract import ContractError


class Stage2ASmokeReviewTests(unittest.TestCase):
    def fixture(self, root: Path, mutate=None):
        (root / "extraction_PRIVATE" / "by_utterance").mkdir(parents=True)
        (root / "runtime_verification_SAFE.json").write_text(json.dumps({"all_checks_passed": True}))
        statuses = [{"sample_id": f"private-{i}", "status": "complete"} for i in range(10)]
        (root / "extraction_PRIVATE" / "adapter_run_SAFE.json").write_text(json.dumps({"mode": "smoke", "utterance_count": 10, "gate_e_evaluated": False, "statuses": statuses}))
        (root / "stage2_smoke_QA_SAFE.json").write_text(json.dumps({"metrics": {"utterance_outputs": 10, "word_tokens": 10}}))
        for i in range(10):
            row = {field: "" for field in adapter.FIELDS}
            row.update(sample_id=f"private-{i}", speaker_id="PRIVATE_SPEAKER", condition="Neutral", prompt_id="1", utterance_id=f"u{i}", word_index="0", lexical_occurrence_index="0", source_word=f"SECRET_TOKEN_{i}", normalized_word=f"secret_token_{i}", alignment_status="aligned", alignment_confidence=".9", word_start_sec="0.1", word_end_sec="0.3", word_duration_sec="0.2", log_word_duration=str(math.log(.2)), f0_mean_hz="100", f0_p10_hz="90", f0_p90_hz="110", f0_robust_range_hz="20", energy_db="-20", authoritative_syllable_count="1", syllable_lookup_status="first_canonical", feature_status="complete", adapter_version="test")
            fields = list(adapter.FIELDS)
            if mutate:
                fields, row = mutate(i, fields, row)
            with (root / "extraction_PRIVATE" / "by_utterance" / f"private-{i}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader(); writer.writerow(row)
        return root

    def run_fixture(self, mutate=None):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = self.fixture(Path(temporary.name), mutate)
        artifact = review.review(root)
        return root, artifact

    def test_exactly_ten_csvs_required(self):
        root, _ = self.run_fixture()
        (root / "extraction_PRIVATE" / "by_utterance" / "private-9.csv").unlink()
        with self.assertRaisesRegex(ContractError, "csv_count_exactly_10"):
            review.review(root)

    def test_runtime_smoke_mode_and_gate_e_contracts(self):
        for relative, key, value, message in (
            ("runtime_verification_SAFE.json", "all_checks_passed", False, "runtime_all_checks_passed"),
            ("extraction_PRIVATE/adapter_run_SAFE.json", "mode", "full", "adapter_mode_smoke"),
            ("extraction_PRIVATE/adapter_run_SAFE.json", "gate_e_evaluated", True, "gate_e_not_evaluated"),
        ):
            root, _ = self.run_fixture(); path = root / relative; data = json.loads(path.read_text()); data[key] = value; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ContractError, message): review.review(root)

    def test_schema_mismatch_is_hard_error(self):
        def mutate(i, fields, row):
            return (fields[:-1], row) if i == 0 else (fields, row)
        _, result = self.run_fixture(mutate)
        self.assertFalse(result["schema_checks"]["exact_schema_identity"])
        self.assertIn("schema_mismatch_files", {x["check"] for x in result["hard_structural_errors"]})

    def test_invalid_and_nonmonotonic_timing(self):
        def invalid(i, fields, row):
            if i == 0: row.update(word_start_sec=".3", word_end_sec=".2", word_duration_sec="-.1")
            return fields, row
        _, result = self.run_fixture(invalid)
        self.assertEqual(result["timing_checks"]["end_not_greater_than_start"], 1)
        self.assertEqual(result["timing_checks"]["nonpositive_word_duration"], 1)
        self.assertIn("invalid_start_end", {x["check"] for x in result["hard_structural_errors"]})
        def two_rows(i, fields, row):
            return fields, row
        root, _ = self.run_fixture(two_rows)
        path = root / "extraction_PRIVATE/by_utterance/private-0.csv"
        with path.open(newline="") as handle: rows = list(csv.DictReader(handle))
        second = dict(rows[0]); second.update(word_index="1", word_start_sec=".05", word_end_sec=".08", word_duration_sec=".03", log_word_duration=str(math.log(.03)))
        with path.open("w", newline="") as handle:
            writer=csv.DictWriter(handle, adapter.FIELDS); writer.writeheader(); writer.writerows([rows[0], second])
        qa = json.loads((root/"stage2_smoke_QA_SAFE.json").read_text()); qa["metrics"]["word_tokens"] = 11; (root/"stage2_smoke_QA_SAFE.json").write_text(json.dumps(qa))
        result = review.review(root)
        self.assertEqual(result["timing_checks"]["non_monotonic_starts"], 1)

    def test_duplicate_token_and_nonfinite_values(self):
        root, _ = self.run_fixture()
        path = root / "extraction_PRIVATE/by_utterance/private-1.csv"
        with path.open(newline="") as handle: row = next(csv.DictReader(handle))
        row.update(sample_id="private-0", f0_mean_hz="nan")
        with path.open("w", newline="") as handle:
            writer=csv.DictWriter(handle, adapter.FIELDS); writer.writeheader(); writer.writerow(row)
        result = review.review(root)
        self.assertEqual(result["duplicate_checks"]["duplicate_key_columns"], ["sample_id", "word_index"])
        self.assertEqual(result["duplicate_checks"]["duplicate_row_count"], 1)
        self.assertEqual(result["duplicate_checks"]["duplicate_key_count"], 1)
        self.assertTrue(result["duplicate_checks"]["duplicate_rows_present"])
        self.assertIn("duplicate_token_evidence", {x["check"] for x in result["hard_structural_errors"]})
        self.assertEqual(result["feature_missingness"]["nonfinite_numeric_by_column"]["f0_mean_hz"], 1)

    def test_zero_duplicate_case_is_explicit(self):
        _, result = self.run_fixture()
        self.assertEqual(result["duplicate_checks"], {
            "duplicate_key_columns": ["sample_id", "word_index"],
            "duplicate_row_count": 0, "duplicate_key_count": 0,
            "duplicate_rows_present": False,
        })

    def test_all_timing_counts_and_duration_tolerance_are_explicit(self):
        expected = {
            "missing_word_start", "missing_word_end", "negative_word_start",
            "end_not_greater_than_start", "nonpositive_word_duration",
            "nonfinite_word_start", "nonfinite_word_end", "nonfinite_word_duration",
            "duration_identity_mismatch_count", "non_monotonic_starts",
        }
        _, clean = self.run_fixture()
        self.assertTrue(expected <= set(clean["timing_checks"]))
        self.assertEqual(clean["timing_checks"]["duration_identity_absolute_tolerance_sec"], 1e-12)
        cases = (
            ({"word_start_sec": ""}, "missing_word_start"),
            ({"word_end_sec": ""}, "missing_word_end"),
            ({"word_start_sec": "-.1", "word_end_sec": ".1", "word_duration_sec": ".2"}, "negative_word_start"),
            ({"word_start_sec": ".3", "word_end_sec": ".3"}, "end_not_greater_than_start"),
            ({"word_duration_sec": "0"}, "nonpositive_word_duration"),
            ({"word_start_sec": "nan"}, "nonfinite_word_start"),
            ({"word_end_sec": "inf"}, "nonfinite_word_end"),
            ({"word_duration_sec": "-inf"}, "nonfinite_word_duration"),
            ({"word_duration_sec": ".21"}, "duration_identity_mismatch_count"),
        )
        for changes, check in cases:
            def mutate(i, fields, row, changes=changes):
                if i == 0: row.update(changes)
                return fields, row
            _, result = self.run_fixture(mutate)
            self.assertEqual(result["timing_checks"][check], 1, check)
            self.assertIn("invalid_start_end", {x["check"] for x in result["hard_structural_errors"]})

    def test_f0_robust_range_structural_rules(self):
        def zero(i, fields, row):
            if i == 0: row.update(f0_p10_hz="100", f0_p90_hz="100", f0_robust_range_hz="0")
            return fields, row
        _, result = self.run_fixture(zero)
        robust = result["feature_missingness"]["f0_robust_range"]
        self.assertEqual(robust["zero_range_count"], 1)
        self.assertEqual(robust["negative_range_count"], 0)
        self.assertNotIn("invalid_f0_range_order", {x["check"] for x in result["hard_structural_errors"]})
        for changes in (
            {"f0_robust_range_hz": "-1"},
            {"f0_p10_hz": "110", "f0_p90_hz": "100", "f0_robust_range_hz": "10"},
        ):
            def invalid(i, fields, row, changes=changes):
                if i == 0: row.update(changes)
                return fields, row
            _, invalid_result = self.run_fixture(invalid)
            self.assertIn("invalid_f0_range_order", {x["check"] for x in invalid_result["hard_structural_errors"]})

    def test_f0_and_syllable_missingness_are_not_automatic_defects(self):
        def missing(i, fields, row):
            if i == 0: row.update(f0_mean_hz="", authoritative_syllable_count="", syllable_lookup_status="missing")
            return fields, row
        _, result = self.run_fixture(missing)
        self.assertIsNone(result["confirmed_technical_defect"])
        self.assertEqual(result["human_review_status"], "pending")
        warning = next(x for x in result["descriptive_warnings"] if x["check"] == "f0_mean_missing")
        self.assertEqual(warning["classification"], "EXPECTED_MISSINGNESS"); self.assertFalse(warning["automatic_defect"])
        self.assertEqual(result["syllable_checks"]["missing_count"], 1)

    def test_safe_outputs_do_not_leak_private_values(self):
        root, _ = self.run_fixture()
        text = (root / "stage2_smoke_technical_review_SAFE.json").read_text()
        self.assertNotIn("SECRET_TOKEN", text); self.assertNotIn("PRIVATE_SPEAKER", text); self.assertNotIn("private-0", text)
        decision = json.loads((root / "stage2_smoke_human_review_SAFE.json").read_text())
        self.assertFalse(decision["stage2b_authorized"]); self.assertIsNone(decision["confirmed_technical_defect"])
        self.assertEqual(len(decision["source_machine_review_sha256"]), 64)

    def test_review_never_opens_audio_or_invokes_scientific_pipeline(self):
        root, _ = self.run_fixture()
        with mock.patch.object(adapter, "read_audio_interval", side_effect=AssertionError("audio opened")), mock.patch.object(adapter, "process_utterance", side_effect=AssertionError("alignment/acoustics invoked")):
            result = review.review(root)
        self.assertFalse(result["audio_reprocessed"]); self.assertFalse(result["alignment_reprocessed"]); self.assertFalse(result["acoustics_reprocessed"])
        self.assertFalse(result["gate_e_evaluated"]); self.assertFalse(result["reliability_evaluated"]); self.assertFalse(result["stage2b_authorized"])
        source = Path(review.__file__).read_text().casefold()
        for forbidden in ("stage2_reliability", "esd_reml_reliability", "soundfile", ".wav", "bootstrap", "leave_one_speaker_out", "lexical_composition_sensitivity"):
            self.assertNotIn(forbidden, source)

    def test_wrapper_locks_validated_interpreter(self):
        shell = (Path(review.__file__).parent / "run_stage2a_review.sh").read_text()
        self.assertIn("PYTHON_BIN=/cluster/env/bin/python", shell)
        self.assertNotIn("python3", shell)


if __name__ == "__main__": unittest.main()
