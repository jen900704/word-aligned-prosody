#!/usr/bin/env python3
"""Outcome-blind DAIC word-local extraction using the frozen ESD contract.

The script opens only participant archives, participant audio, and transcripts.
It never reads split, PHQ, diagnosis, or demographic files. Scientific acoustic
values are written to PRIVATE token files; stdout and the technical report expose
only counts, hashes, versions, elapsed time, and the wall-clock projection.
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
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from frozen_word_aligner import (
    ALIGN_CHECKPOINT_BASENAME,
    ALIGN_CHECKPOINT_SHA256,
    SAMPLE_RATE,
    align_emission,
    file_sha,
    load_frozen_model,
)

ROOT = Path(os.environ.get("WAP_DAIC_ROOT", "data/DAIC_WOZ"))
OUT = Path(os.environ.get("WAP_DAIC_OUT", "work/runtime_daic_v3"))
RESOURCE_DIR = Path(os.environ.get("WAP_RESOURCE_DIR", "resources"))
CMUDICT = RESOURCE_DIR / "cmudict.dict"
CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_ENTRIES = 135166
PROTOCOL = "clinical_regime_amendment_v2_plus_v3_1_addendum"
EXTRACTOR_VERSION = "daic_frozen_esd_contract_v3"

FRAME_LENGTH_SEC = 0.040
HOP_LENGTH_SEC = 0.010
F0_MIN_HZ = 60.0
F0_MAX_HZ = 500.0
SILENCE_RMS_THRESHOLD = 0.0001
VOICING_CORRELATION_THRESHOLD = 0.30
OUTER_PUNCT = re.compile(r"(^[^\w']+|[^\w']+$)", re.UNICODE)
FORBIDDEN_NAME_PARTS = (
    "phq",
    "label",
    "split",
    "gender",
    "diagnos",
    "demograph",
    "outcome",
)
FIELDS = (
    "sample_id",
    "participant_id",
    "speaker_id",
    "condition",
    "utterance_index",
    "utterance_start_sec",
    "utterance_end_sec",
    "word_index",
    "lexical_occurrence_index",
    "source_word",
    "normalized_word",
    "alignment_status",
    "alignment_error_code",
    "alignment_confidence",
    "word_start_sec",
    "word_end_sec",
    "word_duration_sec",
    "log_word_duration",
    "f0_mean_hz",
    "f0_p10_hz",
    "f0_p90_hz",
    "f0_robust_range_hz",
    "energy_db",
    "authoritative_syllable_count",
    "syllable_lookup_status",
    "feature_status",
    "feature_error_code",
    "extractor_version",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
        raise RuntimeError(f"forbidden clinical path: {path}")


def verify_resources() -> dict:
    if not CMUDICT.is_file() or sha256(CMUDICT) != CMUDICT_SHA256:
        raise RuntimeError("frozen CMUdict checksum mismatch")
    checkpoint = RESOURCE_DIR / ALIGN_CHECKPOINT_BASENAME
    if not checkpoint.is_file() or file_sha(checkpoint) != ALIGN_CHECKPOINT_SHA256:
        raise RuntimeError("frozen alignment checkpoint checksum mismatch")
    return {
        "cmudict_sha256": CMUDICT_SHA256,
        "alignment_checkpoint_sha256": ALIGN_CHECKPOINT_SHA256,
    }


def normalize_text(text: str) -> list[dict]:
    output = []
    occurrences: dict[str, int] = {}
    for position, source in enumerate(str(text).split()):
        normalized = source.replace("’", "'").lower()
        normalized = OUTER_PUNCT.sub("", normalized)
        normalized = re.sub(r"[^\w'-]", "", normalized)
        if not normalized:
            continue
        occurrence = occurrences.get(normalized, 0)
        occurrences[normalized] = occurrence + 1
        output.append(
            {
                "source_word": source,
                "normalized_word": normalized,
                "word_index": position,
                "lexical_occurrence_index": occurrence,
            }
        )
    return output


def load_cmudict() -> dict[str, list[int]]:
    dictionary: dict[str, list[int]] = defaultdict(list)
    entries = 0
    with CMUDICT.open(encoding="latin-1") as handle:
        for line in handle:
            if not line.strip() or line.startswith(";;;"):
                continue
            word, phones = line.split(maxsplit=1)
            base = re.sub(r"\(\d+\)$", "", word).lower()
            count = sum(bool(re.search(r"[012]$", phone)) for phone in phones.split())
            dictionary[base].append(count)
            entries += 1
    if entries != CMUDICT_ENTRIES:
        raise RuntimeError("frozen CMUdict entry count mismatch")
    return dictionary


def frames(signal: np.ndarray, size: int, hop: int) -> list[np.ndarray]:
    if not len(signal):
        return []
    if len(signal) <= size:
        return [np.pad(signal, (0, size - len(signal)))]
    return [signal[index : index + size] for index in range(0, len(signal) - size + 1, hop)]


def analyze_word(signal: np.ndarray, sample_rate: int) -> dict:
    size = round(FRAME_LENGTH_SEC * sample_rate)
    hop = round(HOP_LENGTH_SEC * sample_rate)
    chunks = frames(signal, size, hop)
    window = np.hanning(size)
    minimum_lag = max(1, int(sample_rate / F0_MAX_HZ))
    maximum_lag = min(size - 2, int(sample_rate / F0_MIN_HZ))
    pitches = []
    for chunk in chunks:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            continue
        centered = (chunk - chunk.mean()) * window
        nfft = 1 << (2 * size - 1).bit_length()
        spectrum = np.fft.rfft(centered, nfft)
        autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:size]
        if autocorrelation[0] <= 0:
            continue
        search = autocorrelation[minimum_lag : maximum_lag + 1] / autocorrelation[0]
        best = int(np.argmax(search))
        if search[best] >= VOICING_CORRELATION_THRESHOLD:
            pitches.append(sample_rate / float(best + minimum_lag))
    pitch = np.asarray(pitches, dtype=float)
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) if len(signal) else math.nan
    lower = float(np.percentile(pitch, 10)) if len(pitch) else math.nan
    upper = float(np.percentile(pitch, 90)) if len(pitch) else math.nan
    return {
        "f0_mean_hz": float(np.mean(pitch)) if len(pitch) else math.nan,
        "f0_p10_hz": lower,
        "f0_p90_hz": upper,
        "f0_robust_range_hz": upper - lower if len(pitch) else math.nan,
        "energy_db": max(-100.0, 20 * np.log10(max(rms, 1e-5)))
        if np.isfinite(rms)
        else math.nan,
    }


def load_acoustic_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    signal, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = signal.mean(axis=1)
    if sample_rate != SAMPLE_RATE and len(mono):
        count = max(1, round(len(mono) * SAMPLE_RATE / sample_rate))
        mono = np.interp(
            np.linspace(0, len(mono) - 1, count), np.arange(len(mono)), mono
        ).astype("float32")
        sample_rate = SAMPLE_RATE
    return mono, sample_rate


class FrozenSegmentAligner:
    def __init__(self, model_dir: Path):
        self.model, self.dictionary = load_frozen_model(model_dir, "cpu")

    def align(
        self,
        full_audio: np.ndarray,
        start: float,
        end: float,
        tokens: list[dict],
    ) -> list[dict]:
        left = max(0, round(start * SAMPLE_RATE))
        right = min(len(full_audio), round(end * SAMPLE_RATE))
        waveform = torch.from_numpy(full_audio[left:right]).unsqueeze(0)
        lengths = None
        if waveform.shape[-1] < 400:
            lengths = torch.as_tensor([waveform.shape[-1]], device="cpu")
            waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))
        with torch.inference_mode():
            emissions, _ = self.model(waveform, lengths=lengths)
            emission = torch.log_softmax(emissions, dim=-1)[0].cpu().detach()
        text = " ".join(token["normalized_word"] for token in tokens)
        return align_emission(emission, text, self.dictionary, start, end, 0)


def choose_archives(count: int | None) -> list[Path]:
    archives = []
    for path in ROOT.glob("*_P.zip"):
        reject_forbidden_path(path)
        archives.append(path)
    archives.sort(key=lambda path: (hashlib.sha256(path.stem.encode()).hexdigest(), path.name))
    if len(archives) != 189:
        raise RuntimeError(f"expected 189 participant archives, found {len(archives)}")
    return archives if count is None else archives[:count]


def _write_private_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_one(
    archive_path: Path,
    aligner: FrozenSegmentAligner,
    cmudict: dict[str, list[int]],
) -> dict:
    participant = archive_path.stem.split("_")[0]
    with tempfile.TemporaryDirectory(prefix=f"daic_{participant}_") as directory:
        temporary = Path(directory)
        audio_name = f"{participant}_AUDIO.wav"
        transcript_name = f"{participant}_TRANSCRIPT.csv"
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if audio_name not in names or transcript_name not in names:
                raise RuntimeError("participant archive lacks audio or transcript")
            if any(any(part in name.lower() for part in FORBIDDEN_NAME_PARTS) for name in (audio_name, transcript_name)):
                raise RuntimeError("forbidden archive member selected")
            archive.extract(audio_name, temporary)
            archive.extract(transcript_name, temporary)
        audio_path = temporary / audio_name
        transcript_path = temporary / transcript_name
        # DAIC distributes tab-delimited transcript tables despite the .csv
        # suffix. Fix the delimiter explicitly rather than using inference.
        transcript = pd.read_csv(transcript_path, sep="\t")
        required = {"start_time", "stop_time", "speaker", "value"}
        if not required.issubset(transcript.columns):
            raise RuntimeError("unexpected transcript schema")
        participant_rows = (
            transcript[
                transcript["speaker"].astype(str).str.lower().eq("participant")
            ]
            .sort_values("start_time", kind="mergesort")
            .reset_index(drop=True)
        )
        acoustic_audio, acoustic_rate = load_acoustic_audio(audio_path)
        if acoustic_rate != SAMPLE_RATE:
            raise RuntimeError("audio resampling did not produce the frozen 16-kHz rate")
        # DAIC audio is distributed as ordinary PCM WAV. Reuse the normalized
        # mono 16-kHz waveform for alignment because ffmpeg is not a dependency
        # of this isolated Windows runtime.
        alignment_audio = acoustic_audio
        output_rows = []
        utterance_failures = 0
        unaligned_tokens = 0
        for utterance_index, record in participant_rows.iterrows():
            tokens = normalize_text(record.get("value", ""))
            start = max(0.0, float(record["start_time"]))
            end = min(float(record["stop_time"]), len(alignment_audio) / SAMPLE_RATE)
            if not tokens or not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            sample_id = f"{participant}:{utterance_index}"
            try:
                aligned = aligner.align(alignment_audio, start, end, tokens)
                alignment_error = ""
            except Exception as error:
                aligned = []
                alignment_error = type(error).__name__
                utterance_failures += 1
            for token_index, token in enumerate(tokens):
                base = {
                    "sample_id": sample_id,
                    "participant_id": participant,
                    "speaker_id": participant,
                    "condition": "DAIC",
                    "utterance_index": int(utterance_index),
                    "utterance_start_sec": start,
                    "utterance_end_sec": end,
                    **token,
                    "extractor_version": EXTRACTOR_VERSION,
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
                            **base,
                            "alignment_status": "unaligned",
                            "alignment_error_code": alignment_error or "INVALID_OR_MISSING_BOUNDS",
                            "feature_status": "not_attempted",
                            "feature_error_code": "ALIGNMENT_FAILED",
                        }
                    )
                    continue
                word_start = float(word_start)
                word_end = float(word_end)
                left = max(0, round(word_start * acoustic_rate))
                right = min(len(acoustic_audio), round(word_end * acoustic_rate))
                clip = acoustic_audio[left:right]
                feature = analyze_word(clip, acoustic_rate)
                syllables = cmudict.get(token["normalized_word"], [])
                duration = word_end - word_start
                output_rows.append(
                    {
                        **base,
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
        private_path = OUT / "tokens_PRIVATE" / f"{participant}.csv"
        _write_private_csv(private_path, output_rows)
        aligned_count = sum(row.get("alignment_status") == "aligned" for row in output_rows)
        return {
            "participant_id": participant,
            "participant_utterances": len(participant_rows),
            "normalized_tokens": len(output_rows),
            "aligned_tokens": aligned_count,
            "unaligned_tokens": unaligned_tokens,
            "utterance_alignment_failures": utterance_failures,
            "private_output_sha256": sha256(private_path),
        }


def technical_report(reports: list[dict], setup_seconds: float, total_seconds: float) -> dict:
    elapsed = [float(report["elapsed_s"]) for report in reports]
    median_seconds = float(np.median(elapsed)) if elapsed else math.nan
    projected_full_seconds = setup_seconds + median_seconds * 189 if elapsed else math.nan
    projected_days = projected_full_seconds / 86400 if elapsed else math.nan
    maximum_within_four_days = (
        min(189, int(max(0.0, 4 * 86400 - setup_seconds) // median_seconds))
        if elapsed and median_seconds > 0
        else 0
    )
    return {
        "protocol": PROTOCOL,
        "extractor_version": EXTRACTOR_VERSION,
        "mode": "technical_smoke",
        "n": len(reports),
        "scientific_results_inspected": False,
        "labels_loaded": False,
        "feature_contract": {
            "frame_length_sec": FRAME_LENGTH_SEC,
            "hop_length_sec": HOP_LENGTH_SEC,
            "f0_min_hz": F0_MIN_HZ,
            "f0_max_hz": F0_MAX_HZ,
            "silence_rms_threshold": SILENCE_RMS_THRESHOLD,
            "voicing_correlation_threshold": VOICING_CORRELATION_THRESHOLD,
        },
        "resource_hashes": verify_resources(),
        "runtime": {
            "python_packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchaudio", "numpy", "pandas", "soundfile")
            },
            "device": "cpu",
            "audio_decoder": "soundfile_float32_mono_linear_resample_to_16k",
        },
        "aligner_setup_s": setup_seconds,
        "total_s": total_seconds,
        "median_participant_s": median_seconds,
        "projected_full_s": projected_full_seconds,
        "projected_full_days": projected_days,
        "maximum_participants_within_four_days": maximum_within_four_days,
        "participants": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--participants",
        type=int,
        default=10,
        help="deterministic technical smoke size; use 0 only after smoke authorization",
    )
    args = parser.parse_args()
    if args.participants < 0 or args.participants > 189:
        raise ValueError("participants must be in 0..189")
    verify_resources()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    aligner = FrozenSegmentAligner(RESOURCE_DIR)
    setup_seconds = time.time() - started
    cmudict = load_cmudict()
    archives = choose_archives(None if args.participants == 0 else args.participants)
    reports = []
    for archive in archives:
        participant_started = time.time()
        report = extract_one(archive, aligner, cmudict)
        report["elapsed_s"] = time.time() - participant_started
        reports.append(report)
        print(
            json.dumps(
                {
                    "participant_id": report["participant_id"],
                    "participant_utterances": report["participant_utterances"],
                    "normalized_tokens": report["normalized_tokens"],
                    "aligned_tokens": report["aligned_tokens"],
                    "utterance_alignment_failures": report["utterance_alignment_failures"],
                    "elapsed_s": report["elapsed_s"],
                }
            ),
            flush=True,
        )
    report = technical_report(reports, setup_seconds, time.time() - started)
    report_path = OUT / f"smoke_{len(reports)}_technical_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n", "projected_full_days", "maximum_participants_within_four_days", "labels_loaded")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
