#!/usr/bin/env python3
"""Create the frozen, reviewer-blinded PTSD-STOP role-audit manifests.

This program reads only the outcome-blind staging manifest and role columns.  It
does not print or copy transcript text.  The reviewer manifest and answer key
are deliberately separate PRIVATE files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import pandas as pd

import ptsd_stop_wordlocal_extract_v1 as ingestion


AUDIT_VERSION = "ptsd_stop_role_audit_v1"
AUDIT_PER_STRATUM = 100
AUDIT_SALT = "ptsd-stop-role-audit-v1-review-order"
REVIEW_FIELDS = (
    "audit_id",
    "recording_id",
    "participant_id",
    "audio_path",
    "transcript_path",
    "speaker_a",
    "speaker_b",
    "reviewer_participant_speaker",
    "reviewer_confidence",
    "reviewer_notes",
)
KEY_FIELDS = (
    "audit_id",
    "recording_id",
    "tie_stratum",
    "operational_participant_speaker",
    "operational_interviewer_speaker",
)


def audit_hash(recording_id: str) -> bytes:
    return hashlib.sha256(recording_id.encode()).digest()


def audit_id(recording_id: str) -> str:
    return hashlib.sha256(f"{AUDIT_VERSION}|{recording_id}".encode()).hexdigest()[:16]


def select_records(records: list[dict], per_stratum: int = AUDIT_PER_STRATUM) -> list[dict]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    selected = []
    for tie in (False, True):
        stratum = [row for row in records if (row["tie"] == "True") is tie]
        stratum.sort(key=lambda row: (audit_hash(row["recording_id"]), row["recording_id"]))
        if len(stratum) < per_stratum:
            raise RuntimeError("role-audit stratum lacks frozen support")
        selected.extend(stratum[:per_stratum])
    selected.sort(
        key=lambda row: (
            hashlib.sha256(f"{AUDIT_SALT}|{row['recording_id']}".encode()).digest(),
            row["recording_id"],
        )
    )
    return selected


def role_key(path: Path) -> tuple[str, str, tuple[str, str]]:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=["speaker", "speaker_role"],
    )
    pairs = frame[["speaker", "speaker_role"]].drop_duplicates()
    mapping: dict[str, set[str]] = {}
    for row in pairs.to_dict("records"):
        mapping.setdefault(row["speaker"].strip(), set()).add(row["speaker_role"].strip())
    if len(mapping) != 2 or any(len(values) != 1 for values in mapping.values()):
        raise RuntimeError("role map is not a two-speaker one-to-one assignment")
    inverse = {next(iter(values)): speaker for speaker, values in mapping.items()}
    if set(inverse) != {"participant", "interviewer"}:
        raise RuntimeError("participant/interviewer role is absent")
    speakers = tuple(sorted(mapping))
    return inverse["participant"], inverse["interviewer"], speakers


def build_manifests(records: list[dict]) -> tuple[list[dict], list[dict]]:
    review = []
    key = []
    for record in records:
        participant, interviewer, speakers = role_key(record["transcript_path"])
        identifier = audit_id(record["recording_id"])
        review.append(
            {
                "audit_id": identifier,
                "recording_id": record["recording_id"],
                "participant_id": record["participant_id"],
                "audio_path": str(record["audio_path"]),
                "transcript_path": str(record["transcript_path"]),
                "speaker_a": speakers[0],
                "speaker_b": speakers[1],
                "reviewer_participant_speaker": "",
                "reviewer_confidence": "",
                "reviewer_notes": "",
            }
        )
        key.append(
            {
                "audit_id": identifier,
                "recording_id": record["recording_id"],
                "tie_stratum": record["tie"],
                "operational_participant_speaker": participant,
                "operational_interviewer_speaker": interviewer,
            }
        )
    return review, key


def write_private(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-output", required=True, type=Path)
    parser.add_argument("--key-output", required=True, type=Path)
    args = parser.parse_args()
    if args.review_output.resolve() == args.key_output.resolve():
        raise ValueError("review and key outputs must differ")
    records = ingestion.load_manifest()
    selected = select_records(records)
    review, key = build_manifests(selected)
    write_private(args.review_output, REVIEW_FIELDS, review)
    write_private(args.key_output, KEY_FIELDS, key)
    print(
        {
            "audit_version": AUDIT_VERSION,
            "review_rows": len(review),
            "key_rows": len(key),
            "tie_rows": sum(row["tie_stratum"] == "True" for row in key),
            "non_tie_rows": sum(row["tie_stratum"] == "False" for row in key),
            "outcomes_loaded": False,
            "transcript_text_exposed": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
