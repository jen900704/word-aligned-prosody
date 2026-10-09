#!/usr/bin/env python3
"""Strict structural-only projector for MSP-Podcast v1.12 consensus metadata."""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from pathlib import Path

from tools.msp_structural_v1 import TEST3_RE, recording_unit_id, StructuralError

SOURCE_FIELDS = ("FileName", "SpkrID", "Split_Set")
OUTPUT_FIELDS = ("filename", "speaker_id", "split_set", "recording_unit_id")
EXPECTED_BASENAME = "labels_consensus.csv"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def project(source: Path, output: Path, summary: Path):
    if source.name != EXPECTED_BASENAME:
        raise StructuralError("unexpected metadata source basename")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    read_rows = projected = test3 = missing_speaker = 0
    speakers, units = set(), set()
    tmp = output.with_suffix(output.suffix + ".tmp")
    with source.open("r", encoding="utf-8-sig", newline="") as src, tmp.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise StructuralError("empty metadata source") from exc
        if len(header) != len(set(header)):
            raise StructuralError("duplicate source columns")
        missing = [x for x in SOURCE_FIELDS if x not in header]
        if missing:
            raise StructuralError("required structural columns absent: " + ",".join(missing))
        idx = {name: header.index(name) for name in SOURCE_FIELDS}
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in reader:
            read_rows += 1
            if len(row) != len(header):
                raise StructuralError(f"row-width mismatch at data row {read_rows}")
            filename = row[idx["FileName"]].strip()
            speaker = row[idx["SpkrID"]].strip()
            split_set = row[idx["Split_Set"]].strip()
            if TEST3_RE.fullmatch(filename):
                test3 += 1
                continue
            if not speaker:
                missing_speaker += 1
                continue
            unit = recording_unit_id(filename)
            writer.writerow({"filename": filename, "speaker_id": speaker, "split_set": split_set, "recording_unit_id": unit})
            projected += 1
            speakers.add(speaker)
            units.add(unit)
    os.chmod(tmp, 0o600)
    tmp.replace(output)
    record = {
        "audit": "msp_structural_projection_v1",
        "source_basename": source.name,
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "source_fields_projected": list(SOURCE_FIELDS),
        "output_fields": list(OUTPUT_FIELDS),
        "other_source_fields_emitted": False,
        "emotion_fields_used": False,
        "read_rows": read_rows,
        "projected_rows": projected,
        "test3_rows_excluded": test3,
        "missing_speaker_rows_excluded": missing_speaker,
        "unique_speakers": len(speakers),
        "unique_recording_units": len(units),
        "acoustic_values_accessed": False,
        "reliability_values_accessed": False,
        "clinical_outcomes_accessed": False,
    }
    summary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(summary, 0o600)
    return record

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    a = ap.parse_args(argv)
    project(a.labels_csv, a.output, a.summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
