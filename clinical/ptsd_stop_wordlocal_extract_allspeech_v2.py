#!/usr/bin/env python3
"""Outcome-blind PTSD-STOP word-local extraction under the frozen ESD contract.

The extractor opens only the frozen staging manifest, audio, and role-labeled
transcripts.  PCL, diagnosis, demographic, label, outcome, and crosswalk files
are forbidden.  Scientific acoustic rows are written only to PRIVATE files;
stdout and the technical report contain support and runtime information only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import daic_wordlocal_extract_v3 as base


ROOT = Path(os.environ.get('WAP_PTSD_ROOT', 'work/ptsd_stop')) / "ptsd_stop_pipeline/stage_full"
MANIFEST = Path(os.environ.get('WAP_PTSD_ROOT', 'work/ptsd_stop')) / "ptsd_stop_pipeline/stage_full_manifest.csv"
OUT = Path(os.environ.get('WAP_PTSD_ROOT', 'work/ptsd_stop')) / "ptsd_stop_wordlocal_allspeech_v2"
MODEL_DIR = Path(os.environ.get('WAP_PTSD_ROOT', 'work/ptsd_stop')) / "ptsd_stop_wordlocal_v1/resources"
CMUDICT = MODEL_DIR / "cmudict.dict"
EXPECTED_RECORDINGS = 13600
EXPECTED_PARTICIPANTS = 225
MANIFEST_SHA256 = "ed6d568659946452c288e0fec0beb3f6b76ec0ed802f5671976e199a9553633d"
ROLE_LABELER_SHA256 = "49861f77e0ccbe8553b8bfa9938e35d5568c6635cd93e4b5cde88712aba5c840"
PROTOCOL = "clinical_regime_amendment_v2_plus_v3_1_plus_ptsd_stop_ingestion_v1_3"
EXTRACTOR_VERSION = "ptsd_stop_allspeech_contract_v2_0"
ROLE_AUTHORITY = "none_single_speaker_diary_all_diarized_rows_retained"
FORBIDDEN_NAME_PARTS = base.FORBIDDEN_NAME_PARTS + ("pcl", "crosswalk")
TRANSCRIPT_COLUMNS = (
    "speaker",
    "start_timestamp",
    "end_timestamp",
    "message",
    "speaker_role",
)
FIELDS = base.FIELDS + (
    "recording_id",
    "role_assignment_tie",
    "role_authority",
    "source_manifest_sha256",
)
PARTICIPANT_PATTERN = re.compile(r"^(P{1,2}\d+)_")
PARTICIPANT_ALIASES = {"PP2702": "P22702"}


def configure_frozen_resources() -> None:
    base.RESOURCE_DIR = MODEL_DIR
    base.CMUDICT = CMUDICT


def reject_forbidden_path(path: Path) -> None:
    for component in path.parts:
        tokens = re.findall(r"[a-z]+", component.lower())
        if any(
            token == forbidden or token.startswith(forbidden)
            for token in tokens
            for forbidden in FORBIDDEN_NAME_PARTS
        ):
            raise RuntimeError(f"forbidden clinical path: {path}")


def participant_from_stem(stem: str) -> str:
    match = PARTICIPANT_PATTERN.match(stem)
    if match is None:
        raise RuntimeError(f"recording stem lacks frozen participant identity: {stem}")
    raw = match.group(1)
    return PARTICIPANT_ALIASES.get(raw, raw)


def clock_seconds(value: object) -> float:
    parts = str(value).strip().split(":")
    if len(parts) != 3:
        raise ValueError("timestamp is not HH:MM:SS.sss")
    hours, minutes = (int(parts[0]), int(parts[1]))
    seconds = float(parts[2])
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("timestamp component out of range")
    output = hours * 3600.0 + minutes * 60.0 + seconds
    if not math.isfinite(output):
        raise ValueError("timestamp is non-finite")
    return output


def load_manifest(
    path: Path = MANIFEST,
    root: Path = ROOT,
    expected_recordings: int = EXPECTED_RECORDINGS,
    expected_participants: int = EXPECTED_PARTICIPANTS,
    expected_sha256: str = MANIFEST_SHA256,
) -> list[dict]:
    reject_forbidden_path(path)
    if base.sha256(path) != expected_sha256:
        raise RuntimeError("frozen PTSD-STOP manifest checksum mismatch")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "audio_basename",
        "transcript_basename",
        "n_rows",
        "n_speakers",
        "participant_rows",
        "tie",
    }
    if set(frame.columns) != required:
        raise RuntimeError(f"unexpected manifest schema: {list(frame.columns)}")
    if len(frame) != expected_recordings or frame["audio_basename"].duplicated().any():
        raise RuntimeError("frozen manifest recording cardinality mismatch")
    records = []
    participants = set()
    for row in frame.to_dict("records"):
        audio_name = row["audio_basename"]
        stem = Path(audio_name).stem
        participant = participant_from_stem(stem)
        participants.add(participant)
        if row["n_speakers"] != "2" or row["tie"] not in {"True", "False"}:
            raise RuntimeError("manifest speaker/tie contract mismatch")
        audio_path = root / "audio" / audio_name
        transcript_path = root / "transcripts" / f"{stem}.csv"
        for selected in (audio_path, transcript_path):
            # Basenames contain opaque random identifiers and are never parsed
            # for clinical keywords.  Their fixed parent directories are the
            # allowlist boundary.
            reject_forbidden_path(selected.parent)
            if not selected.is_file():
                raise RuntimeError(f"missing staged input: {selected}")
        records.append(
            {
                **row,
                "recording_id": stem,
                "participant_id": participant,
                "audio_path": audio_path,
                "transcript_path": transcript_path,
            }
        )
    if len(participants) != expected_participants:
        raise RuntimeError("frozen manifest participant cardinality mismatch")
    records.sort(
        key=lambda row: (
            hashlib.sha256(row["recording_id"].encode()).digest(),
            row["recording_id"],
        )
    )
    return records


def choose_recordings(
    records: list[dict], count: int | None, shard_index: int, num_shards: int
) -> list[dict]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard specification")
    selected = records if count is None else records[:count]
    return [row for index, row in enumerate(selected) if index % num_shards == shard_index]


def read_participant_intervals(path: Path) -> tuple[list[dict], dict]:
    reject_forbidden_path(path.parent)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    observed_columns = tuple(frame.columns)
    technical_index_column_dropped = False
    if observed_columns == ("Unnamed: 0",) + TRANSCRIPT_COLUMNS:
        index_values = pd.to_numeric(frame["Unnamed: 0"], errors="coerce").to_numpy()
        expected_index = np.arange(len(frame), dtype=float)
        if not np.array_equal(index_values, expected_index):
            raise RuntimeError("unexpected values in technical transcript index column")
        frame = frame.drop(columns=["Unnamed: 0"])
        technical_index_column_dropped = True
    elif observed_columns != TRANSCRIPT_COLUMNS:
        raise RuntimeError(f"unexpected PTSD-STOP transcript schema: {list(frame.columns)}")
    speakers = {value.strip() for value in frame["speaker"] if value.strip()}
    roles = {value.strip() for value in frame["speaker_role"] if value.strip()}
    if not speakers:
        raise RuntimeError("transcript contains no speaker rows")
    intervals = []
    invalid_participant_rows = 0
    empty_participant_rows = 0
    for original_index, row in frame.iterrows():
        tokens = base.normalize_text(row["message"])
        if not tokens:
            empty_participant_rows += 1
            continue
        try:
            start = clock_seconds(row["start_timestamp"])
            end = clock_seconds(row["end_timestamp"])
        except (TypeError, ValueError):
            invalid_participant_rows += 1
            continue
        if end <= start:
            invalid_participant_rows += 1
            continue
        intervals.append(
            {
                "utterance_index": int(original_index),
                "start": start,
                "end": end,
                "tokens": tokens,
            }
        )
    intervals.sort(key=lambda row: (row["start"], row["end"], row["utterance_index"]))
    return intervals, {
        "transcript_rows": int(len(frame)),
        "technical_index_column_dropped": technical_index_column_dropped,
        "participant_rows": int(len(frame)),
        "valid_participant_intervals": len(intervals),
        "invalid_participant_rows": invalid_participant_rows,
        "empty_participant_rows": empty_participant_rows,
        "participant_interval_seconds": float(sum(row["end"] - row["start"] for row in intervals)),
    }


class FrozenDeviceSegmentAligner:
    def __init__(self, model_dir: Path, device: str):
        self.device = device
        self.model, self.dictionary = base.load_frozen_model(model_dir, device)

    def align(
        self,
        full_audio: np.ndarray,
        start: float,
        end: float,
        tokens: list[dict],
    ) -> list[dict]:
        left = max(0, round(start * base.SAMPLE_RATE))
        right = min(len(full_audio), round(end * base.SAMPLE_RATE))
        waveform = torch.from_numpy(full_audio[left:right]).unsqueeze(0)
        lengths = None
        if waveform.shape[-1] < 400:
            lengths = torch.as_tensor([waveform.shape[-1]], device=self.device)
            waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))
        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device), lengths=lengths)
            emission = torch.log_softmax(emissions, dim=-1)[0].cpu().detach()
        text = " ".join(token["normalized_word"] for token in tokens)
        return base.align_emission(emission, text, self.dictionary, start, end, 0)


def write_private_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_one(
    record: dict,
    aligner: FrozenDeviceSegmentAligner,
    cmudict: dict[str, list[int]],
    out: Path = OUT,
) -> dict:
    intervals, qa = read_participant_intervals(record["transcript_path"])
    audio, sample_rate = base.load_acoustic_audio(record["audio_path"])
    if sample_rate != base.SAMPLE_RATE:
        raise RuntimeError("audio resampling did not produce the frozen 16-kHz rate")
    audio_duration = len(audio) / sample_rate
    output_rows = []
    utterance_failures = 0
    unaligned_tokens = 0
    intervals_beyond_audio = 0
    tie = record["tie"] == "True"
    for interval in intervals:
        start = max(0.0, interval["start"])
        end = min(interval["end"], audio_duration)
        if end <= start:
            intervals_beyond_audio += 1
            continue
        tokens = interval["tokens"]
        sample_id = f"{record['recording_id']}:{interval['utterance_index']}"
        try:
            aligned = aligner.align(audio, start, end, tokens)
            alignment_error = ""
        except Exception as error:
            aligned = []
            alignment_error = type(error).__name__
            utterance_failures += 1
        for token_index, token in enumerate(tokens):
            common = {
                "sample_id": sample_id,
                "participant_id": record["participant_id"],
                "speaker_id": record["participant_id"],
                "condition": "PTSD-STOP",
                "utterance_index": int(interval["utterance_index"]),
                "utterance_start_sec": start,
                "utterance_end_sec": end,
                **token,
                "extractor_version": EXTRACTOR_VERSION,
                "recording_id": record["recording_id"],
                "role_assignment_tie": tie,
                "role_authority": ROLE_AUTHORITY,
                "source_manifest_sha256": MANIFEST_SHA256,
            }
            candidate = aligned[token_index] if token_index < len(aligned) else {}
            word_start = candidate.get("start")
            word_end = candidate.get("end")
            if (
                word_start is None
                or word_end is None
                or float(word_start) < start
                or float(word_end) <= float(word_start)
                or float(word_end) > end + 1e-6
            ):
                unaligned_tokens += 1
                output_rows.append(
                    {
                        **common,
                        "alignment_status": "unaligned",
                        "alignment_error_code": alignment_error or "INVALID_OR_MISSING_BOUNDS",
                        "feature_status": "not_attempted",
                        "feature_error_code": "ALIGNMENT_FAILED",
                    }
                )
                continue
            word_start = float(word_start)
            word_end = float(word_end)
            left = max(0, round(word_start * sample_rate))
            right = min(len(audio), round(word_end * sample_rate))
            feature = base.analyze_word(audio[left:right], sample_rate)
            syllables = cmudict.get(token["normalized_word"], [])
            duration = word_end - word_start
            output_rows.append(
                {
                    **common,
                    "alignment_status": "aligned",
                    "alignment_error_code": "",
                    "alignment_confidence": candidate.get("score", ""),
                    "word_start_sec": word_start,
                    "word_end_sec": word_end,
                    "word_duration_sec": duration,
                    "log_word_duration": math.log(duration),
                    **feature,
                    "authoritative_syllable_count": syllables[0] if syllables else "",
                    "syllable_lookup_status": "found" if syllables else "missing",
                    "feature_status": "complete",
                    "feature_error_code": "",
                }
            )
    private_path = out / "tokens_PRIVATE" / f"{record['recording_id']}.csv"
    write_private_csv(private_path, output_rows)
    return {
        "recording_id": record["recording_id"],
        "participant_id": record["participant_id"],
        "role_assignment_tie": tie,
        **qa,
        "audio_duration_seconds": audio_duration,
        "intervals_beyond_audio": intervals_beyond_audio,
        "normalized_tokens": len(output_rows),
        "aligned_tokens": sum(row.get("alignment_status") == "aligned" for row in output_rows),
        "unaligned_tokens": unaligned_tokens,
        "utterance_alignment_failures": utterance_failures,
        "private_output_sha256": base.sha256(private_path),
    }


def technical_report(
    reports: list[dict], setup_seconds: float, total_seconds: float, device: str,
    shard_index: int, num_shards: int,
) -> dict:
    elapsed = [float(record["elapsed_s"]) for record in reports]
    interval_seconds = sum(float(record["participant_interval_seconds"]) for record in reports)
    processing_seconds = sum(elapsed)
    seconds_per_interval_second = processing_seconds / interval_seconds if interval_seconds > 0 else math.nan
    median_seconds = float(np.median(elapsed)) if elapsed else math.nan
    projected_serial_seconds = setup_seconds + median_seconds * EXPECTED_RECORDINGS if elapsed else math.nan
    projected_four_gpu_seconds = projected_serial_seconds / 4.0 if elapsed else math.nan
    return {
        "protocol": PROTOCOL,
        "extractor_version": EXTRACTOR_VERSION,
        "mode": "technical_smoke",
        "n": len(reports),
        "expected_full_n": EXPECTED_RECORDINGS,
        "scientific_results_inspected": False,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "source_manifest_sha256": MANIFEST_SHA256,
        "role_labeler_sha256": ROLE_LABELER_SHA256,
        "role_authority": ROLE_AUTHORITY,
        "shard": {"index": shard_index, "count": num_shards},
        "feature_contract": {
            "frame_length_sec": base.FRAME_LENGTH_SEC,
            "hop_length_sec": base.HOP_LENGTH_SEC,
            "f0_min_hz": base.F0_MIN_HZ,
            "f0_max_hz": base.F0_MAX_HZ,
            "silence_rms_threshold": base.SILENCE_RMS_THRESHOLD,
            "voicing_correlation_threshold": base.VOICING_CORRELATION_THRESHOLD,
        },
        "resource_hashes": base.verify_resources(),
        "runtime": {
            "python_packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchaudio", "numpy", "pandas", "soundfile")
            },
            "device": device,
            "audio_decoder": "soundfile_float32_mono_linear_resample_to_16k",
        },
        "aligner_setup_s": setup_seconds,
        "total_s": total_seconds,
        "median_recording_s": median_seconds,
        "observed_processing_seconds_per_participant_interval_second": seconds_per_interval_second,
        "projected_full_serial_s": projected_serial_seconds,
        "projected_full_serial_days": projected_serial_seconds / 86400 if elapsed else math.nan,
        "projected_full_four_gpu_days": projected_four_gpu_seconds / 86400 if elapsed else math.nan,
        "recordings": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=int, default=1,
                        help="deterministic smoke size; 0 selects the full frozen manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.recordings < 0 or args.recordings > EXPECTED_RECORDINGS:
        raise ValueError(f"recordings must be in 0..{EXPECTED_RECORDINGS}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configure_frozen_resources()
    base.verify_resources()
    records = load_manifest()
    selected = choose_recordings(
        records, None if args.recordings == 0 else args.recordings,
        args.shard_index, args.num_shards,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    aligner = FrozenDeviceSegmentAligner(MODEL_DIR, args.device)
    setup_seconds = time.time() - started
    cmudict = base.load_cmudict()
    reports = []
    for record in selected:
        recording_started = time.time()
        report = extract_one(record, aligner, cmudict)
        report["elapsed_s"] = time.time() - recording_started
        reports.append(report)
        print(json.dumps({
            "recording_id": report["recording_id"],
            "participant_id": report["participant_id"],
            "role_assignment_tie": report["role_assignment_tie"],
            "valid_participant_intervals": report["valid_participant_intervals"],
            "invalid_participant_rows": report["invalid_participant_rows"],
            "normalized_tokens": report["normalized_tokens"],
            "aligned_tokens": report["aligned_tokens"],
            "utterance_alignment_failures": report["utterance_alignment_failures"],
            "elapsed_s": report["elapsed_s"],
        }), flush=True)
    report = technical_report(
        reports, setup_seconds, time.time() - started, args.device,
        args.shard_index, args.num_shards,
    )
    report_path = OUT / f"smoke_{len(selected)}_shard_{args.shard_index}_technical_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "n": report["n"],
        "projected_full_serial_days": report["projected_full_serial_days"],
        "projected_full_four_gpu_days": report["projected_full_four_gpu_days"],
        "labels_loaded": False,
        "scientific_results_inspected": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
