from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "clinical"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import make_ptsd_stop_role_audit_manifest_v1 as mod


def records() -> list[dict]:
    output = []
    for tie in ("False", "True"):
        for index in range(5):
            output.append(
                {
                    "recording_id": f"P{index}_{tie}",
                    "participant_id": f"P{index}",
                    "tie": tie,
                }
            )
    return output


def test_selection_is_deterministic_balanced_and_blinded():
    first = mod.select_records(records(), per_stratum=3)
    second = mod.select_records(list(reversed(records())), per_stratum=3)
    assert [row["recording_id"] for row in first] == [row["recording_id"] for row in second]
    assert sum(row["tie"] == "True" for row in first) == 3
    assert sum(row["tie"] == "False" for row in first) == 3
    with pytest.raises(RuntimeError):
        mod.select_records(records(), per_stratum=6)


def make_transcript(path: Path) -> None:
    pd.DataFrame(
        [
            {"speaker": "Speaker 0", "speaker_role": "interviewer"},
            {"speaker": "Speaker 1", "speaker_role": "participant"},
            {"speaker": "Speaker 1", "speaker_role": "participant"},
        ]
    ).to_csv(path, index=False)


def test_role_key_and_manifest_separation(tmp_path: Path):
    transcript = tmp_path / "x.csv"
    audio = tmp_path / "x.wav"
    make_transcript(transcript)
    record = {
        "recording_id": "P1_x",
        "participant_id": "P1",
        "tie": "True",
        "transcript_path": transcript,
        "audio_path": audio,
    }
    participant, interviewer, speakers = mod.role_key(transcript)
    assert (participant, interviewer) == ("Speaker 1", "Speaker 0")
    assert speakers == ("Speaker 0", "Speaker 1")
    review, key = mod.build_manifests([record])
    assert "operational_participant_speaker" not in review[0]
    assert "tie_stratum" not in review[0]
    assert key[0]["operational_participant_speaker"] == "Speaker 1"
    assert key[0]["tie_stratum"] == "True"


def test_role_key_rejects_inconsistent_roles(tmp_path: Path):
    transcript = tmp_path / "x.csv"
    pd.DataFrame(
        [
            {"speaker": "Speaker 0", "speaker_role": "participant"},
            {"speaker": "Speaker 0", "speaker_role": "interviewer"},
            {"speaker": "Speaker 1", "speaker_role": "participant"},
        ]
    ).to_csv(transcript, index=False)
    with pytest.raises(RuntimeError):
        mod.role_key(transcript)


def test_private_files_are_distinct_and_mode_600(tmp_path: Path):
    path = tmp_path / "review.csv"
    mod.write_private(path, ("x",), [{"x": "1"}])
    assert path.read_text(encoding="utf-8").splitlines() == ["x", "1"]
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
