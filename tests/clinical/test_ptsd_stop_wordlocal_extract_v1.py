from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "clinical"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ptsd_stop_wordlocal_extract_v1 as mod


def test_clock_seconds_strict():
    assert mod.clock_seconds("00:00:09.100") == pytest.approx(9.1)
    assert mod.clock_seconds("01:02:03.500") == pytest.approx(3723.5)
    for value in ("9.1", "00:60:00", "00:00:60", "bad"):
        with pytest.raises(ValueError):
            mod.clock_seconds(value)


def test_participant_from_stem():
    assert mod.participant_from_stem("P20225_4_6_token") == "P20225"
    assert mod.participant_from_stem("PP2702_1_21_token") == "P22702"
    with pytest.raises(RuntimeError):
        mod.participant_from_stem("20225_4_6_token")


def test_read_participant_intervals_skips_invalid_and_empty(tmp_path: Path):
    path = tmp_path / "P1_x.csv"
    pd.DataFrame([
        {"speaker": "Speaker 0", "start_timestamp": "00:00:01.000", "end_timestamp": "00:00:02.000", "message": "hello world", "speaker_role": "participant"},
        {"speaker": "Speaker 1", "start_timestamp": "00:00:02.000", "end_timestamp": "00:00:03.000", "message": "question?", "speaker_role": "interviewer"},
        {"speaker": "Speaker 0", "start_timestamp": "00:00:04.000", "end_timestamp": "00:00:03.000", "message": "invalid", "speaker_role": "participant"},
        {"speaker": "Speaker 0", "start_timestamp": "00:00:05.000", "end_timestamp": "00:00:06.000", "message": "", "speaker_role": "participant"},
    ], columns=mod.TRANSCRIPT_COLUMNS).to_csv(path, index=False)
    intervals, report = mod.read_participant_intervals(path)
    assert len(intervals) == 1
    assert intervals[0]["utterance_index"] == 0
    assert report["participant_rows"] == 3
    assert report["invalid_participant_rows"] == 1
    assert report["empty_participant_rows"] == 1
    assert report["participant_interval_seconds"] == pytest.approx(1.0)
    assert report["technical_index_column_dropped"] is False


def test_read_participant_intervals_accepts_valid_technical_index(tmp_path: Path):
    path = tmp_path / "P1_x.csv"
    frame = pd.DataFrame([
        {"speaker": "Speaker 0", "start_timestamp": "00:00:01.000", "end_timestamp": "00:00:02.000", "message": "hello", "speaker_role": "participant"},
        {"speaker": "Speaker 1", "start_timestamp": "00:00:02.000", "end_timestamp": "00:00:03.000", "message": "question", "speaker_role": "interviewer"},
    ], columns=mod.TRANSCRIPT_COLUMNS)
    frame.to_csv(path, index=True)
    intervals, report = mod.read_participant_intervals(path)
    assert len(intervals) == 1
    assert report["technical_index_column_dropped"] is True


def test_read_participant_intervals_rejects_nonsequential_technical_index(tmp_path: Path):
    path = tmp_path / "P1_x.csv"
    frame = pd.DataFrame([
        {"speaker": "Speaker 0", "start_timestamp": "00:00:01.000", "end_timestamp": "00:00:02.000", "message": "hello", "speaker_role": "participant"},
        {"speaker": "Speaker 1", "start_timestamp": "00:00:02.000", "end_timestamp": "00:00:03.000", "message": "question", "speaker_role": "interviewer"},
    ], columns=mod.TRANSCRIPT_COLUMNS, index=[0, 2])
    frame.to_csv(path, index=True)
    with pytest.raises(RuntimeError, match="technical transcript index"):
        mod.read_participant_intervals(path)


def test_read_participant_intervals_requires_exact_schema(tmp_path: Path):
    path = tmp_path / "P1_x.csv"
    pd.DataFrame({"speaker": ["a"]}).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="schema"):
        mod.read_participant_intervals(path)


def test_choose_recordings_is_disjoint_and_complete():
    records = [{"recording_id": str(i)} for i in range(11)]
    pieces = [mod.choose_recordings(records, None, i, 4) for i in range(4)]
    flattened = [row["recording_id"] for piece in pieces for row in piece]
    assert sorted(flattened, key=int) == [str(i) for i in range(11)]
    assert len(flattened) == len(set(flattened))


def test_load_manifest_validates_hash_and_frozen_cardinality(tmp_path: Path):
    root = tmp_path / "stage_full"
    (root / "audio").mkdir(parents=True)
    (root / "transcripts").mkdir()
    rows = []
    for participant in ("P1", "P2"):
        stem = f"{participant}_1_1_x"
        (root / "audio" / f"{stem}.wav").write_bytes(b"RIFF")
        (root / "transcripts" / f"{stem}.csv").write_text("x", encoding="utf-8")
        rows.append({
            "audio_basename": f"{stem}.wav",
            "transcript_basename": f"{stem}_formatted.csv",
            "n_rows": "1",
            "n_speakers": "2",
            "participant_rows": "1",
            "tie": "False",
        })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    loaded = mod.load_manifest(
        manifest, root, expected_recordings=2, expected_participants=2,
        expected_sha256=digest,
    )
    assert len(loaded) == 2
    assert {row["participant_id"] for row in loaded} == {"P1", "P2"}
    with pytest.raises(RuntimeError, match="checksum"):
        mod.load_manifest(
            manifest, root, expected_recordings=2, expected_participants=2,
            expected_sha256="0" * 64,
        )


def test_forbidden_paths_are_rejected():
    for name in ("PCL5_scores.csv", "outcome.csv", "crosswalk_v8.csv"):
        with pytest.raises(RuntimeError):
            mod.reject_forbidden_path(Path(name))
    mod.reject_forbidden_path(Path("P20359_7_9_qG5pDXlpCln1ufhIBiI8KD8KHQi3rCCA.wav"))
    mod.reject_forbidden_path(Path("stage_full/audio"))


def test_contract_constants():
    assert mod.EXPECTED_RECORDINGS == 13600
    assert mod.EXPECTED_PARTICIPANTS == 225
    assert mod.MANIFEST_SHA256 == "ed6d568659946452c288e0fec0beb3f6b76ec0ed802f5671976e199a9553633d"
    assert mod.ROLE_LABELER_SHA256 == "49861f77e0ccbe8553b8bfa9938e35d5568c6635cd93e4b5cde88712aba5c840"
    assert mod.PARTICIPANT_ALIASES == {"PP2702": "P22702"}
