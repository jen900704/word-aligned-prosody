#!/usr/bin/env python3
"""Build an outcome-blind recording-to-participant identity authority.

Only the frozen staging manifest and structural fields from the host lab
``crosswalk_v8`` table are read. No symptom or other outcome field is queried.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


EXPECTED_RECORDINGS = 13_600
EXPECTED_SOURCE_PARTICIPANTS = 225
EXPECTED_AUTHORITATIVE_PARTICIPANTS = 219
EXPECTED_TEMPORAL_AMBIGUITIES = 3
EXPECTED_SOURCE_REASSIGNED_RECORDINGS = 20
MANIFEST_SHA256 = "ed6d568659946452c288e0fec0beb3f6b76ec0ed802f5671976e199a9553633d"
PARTICIPANT_PATTERN = re.compile(r"^(P{1,2}\d+)_", re.IGNORECASE)
PARTICIPANT_ALIASES = {"PP2702": "P22702"}
FIELDS = (
    "recording_id",
    "source_participant_id",
    "authoritative_participant_id",
    "authoritative_user_day_id",
    "authoritative_date",
    "identity_match_status",
    "temporal_match_status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stem(value: str) -> str:
    name = re.split(r"[\\/]", str(value))[-1]
    return Path(name).stem.lower()


def source_participant(stem: str) -> str:
    match = PARTICIPANT_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"recording lacks participant prefix: {stem}")
    raw = match.group(1).upper()
    return PARTICIPANT_ALIASES.get(raw, raw)


def canonical_user(value: str) -> str:
    digits = re.sub(r"[^0-9]", "", str(value))
    if not digits:
        raise ValueError("authority user ID lacks digits")
    return f"P{digits}"


def build_rows(stage_rows: list[dict], authority_rows: list[dict]) -> list[dict]:
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for row in authority_rows:
        by_stem[file_stem(row["AudioFilename"])].append(row)
    output = []
    for stage in stage_rows:
        recording_id = Path(
            re.split(r"[\\/]", str(stage["audio_basename"]))[-1]
        ).stem
        candidates = by_stem.get(recording_id.lower(), [])
        users = {canonical_user(row["user_id"]) for row in candidates}
        if len(users) != 1:
            raise RuntimeError("recording does not resolve to exactly one authority user")
        authority_user = next(iter(users))
        temporal_pairs = {
            (str(row["ud_id"]).strip(), str(row["Date"]).strip())
            for row in candidates
        }
        if len(temporal_pairs) == 1:
            user_day_id, date = next(iter(temporal_pairs))
            temporal_status = "one_to_one"
        else:
            user_day_id, date = "", ""
            temporal_status = "ambiguous_excluded_from_longitudinal"
        output.append({
            "recording_id": recording_id,
            "source_participant_id": source_participant(recording_id),
            "authoritative_participant_id": authority_user,
            "authoritative_user_day_id": user_day_id,
            "authoritative_date": date,
            "identity_match_status": "one_authoritative_user",
            "temporal_match_status": temporal_status,
        })
    return output


def validate(rows: list[dict]) -> dict:
    if len(rows) != EXPECTED_RECORDINGS:
        raise RuntimeError("identity authority recording cardinality mismatch")
    if len({row["recording_id"] for row in rows}) != EXPECTED_RECORDINGS:
        raise RuntimeError("identity authority recording IDs are not unique")
    source = {row["source_participant_id"] for row in rows}
    authority = {row["authoritative_participant_id"] for row in rows}
    ambiguous = sum(
        row["temporal_match_status"] != "one_to_one" for row in rows
    )
    reassigned = sum(
        row["source_participant_id"] != row["authoritative_participant_id"]
        for row in rows
    )
    observed = (
        len(source), len(authority), ambiguous, reassigned,
    )
    expected = (
        EXPECTED_SOURCE_PARTICIPANTS,
        EXPECTED_AUTHORITATIVE_PARTICIPANTS,
        EXPECTED_TEMPORAL_AMBIGUITIES,
        EXPECTED_SOURCE_REASSIGNED_RECORDINGS,
    )
    if observed != expected:
        raise RuntimeError(f"identity authority structural mismatch: {observed}")
    return {
        "recordings": len(rows),
        "source_participants": len(source),
        "authoritative_participants": len(authority),
        "temporally_ambiguous_recordings": ambiguous,
        "source_reassigned_recordings": reassigned,
    }


def query_authority(mysql: str) -> list[dict]:
    query = "SELECT user_id,ud_id,AudioFilename,Date FROM crosswalk_v8"
    completed = subprocess.run(
        [mysql, "--batch", "--skip-column-names", "ptsd_stop", "-e", query],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = ("user_id", "ud_id", "AudioFilename", "Date")
    return [dict(zip(fields, line.split("\t"))) for line in completed.stdout.splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mysql", default="mysql")
    args = parser.parse_args()
    if sha256(args.stage_manifest) != MANIFEST_SHA256:
        raise RuntimeError("frozen staging manifest checksum mismatch")
    with args.stage_manifest.open(newline="", encoding="utf-8") as handle:
        stage_rows = list(csv.DictReader(handle))
    rows = build_rows(stage_rows, query_authority(args.mysql))
    summary = validate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(args.output, 0o600)
    print({
        **summary,
        "outcomes_loaded": False,
        "symptom_values_loaded": False,
        "output_sha256": sha256(args.output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
