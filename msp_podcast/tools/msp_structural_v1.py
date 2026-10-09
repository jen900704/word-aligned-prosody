#!/usr/bin/env python3
"""Outcome-blind MSP-Podcast structural helpers.

This module never opens corpus files. Callers may supply only a pre-projected
structural table with filename, speaker_id, and optional split_set.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import re

RECORDING_RE = re.compile(r"^(MSP-PODCAST_[0-9]+)_.*[.]wav$")
TEST3_RE = re.compile(r"^MSP-PODCAST_test3_.*[.]wav$")
PRIMARY_SALT = "cross-corpus-msp-primary"
ALT_SALTS = tuple(f"cross-corpus-msp-alt-{i:02d}" for i in range(25))
SPEAKER_ORDER_SALT = "cross-corpus-msp-v1"
ALLOWED_FIELDS = frozenset({"filename", "speaker_id", "split_set"})
MIN_UTTERANCES = 100
MIN_RECORDING_UNITS = 2
MAX_SPEAKERS = 100

class StructuralError(ValueError):
    pass

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def recording_unit_id(filename: str) -> str:
    if TEST3_RE.fullmatch(filename):
        raise StructuralError("Test3 is prospectively excluded")
    m = RECORDING_RE.fullmatch(filename)
    if not m:
        raise StructuralError("unrecognized MSP-Podcast filename")
    return m.group(1)

def speaker_order_key(speaker_id: str) -> str:
    if not isinstance(speaker_id, str) or not speaker_id.strip():
        raise StructuralError("speaker_id must be nonempty")
    return _digest(f"{SPEAKER_ORDER_SALT}|{speaker_id}")

def validate_structural_row(row):
    if not isinstance(row, dict) or not {"filename", "speaker_id"} <= set(row):
        raise StructuralError("structural row requires filename and speaker_id")
    if set(row) - ALLOWED_FIELDS:
        raise StructuralError("non-structural field supplied")
    if not isinstance(row["speaker_id"], str) or not row["speaker_id"].strip():
        raise StructuralError("speaker_id must be normalized known ID")
    recording_unit_id(row["filename"])

def select_fixed_scope(rows):
    counts=defaultdict(int); units=defaultdict(set)
    for row in rows:
        validate_structural_row(row)
        sid=row["speaker_id"].strip(); counts[sid]+=1
        units[sid].add(recording_unit_id(row["filename"]))
    eligible=[s for s in counts if counts[s] >= MIN_UTTERANCES and len(units[s]) >= MIN_RECORDING_UNITS]
    selected=tuple(sorted(eligible, key=speaker_order_key)[:MAX_SPEAKERS])
    diagnostics={s:{"utterances":counts[s], "recording_units":len(units[s])} for s in selected}
    return selected, diagnostics

def split_recording_units(speaker_id: str, recording_units, salt: str = PRIMARY_SALT):
    units = sorted(set(recording_units), key=lambda u: _digest(f"{salt}|{speaker_id}|{u}"))
    if len(units) < MIN_RECORDING_UNITS:
        raise StructuralError("speaker requires at least two recording units")
    half_a, half_b = tuple(units[::2]), tuple(units[1::2])
    if not half_a or not half_b or set(half_a) & set(half_b):
        raise StructuralError("invalid recording-unit split")
    return half_a, half_b