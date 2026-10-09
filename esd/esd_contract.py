#!/usr/bin/env python3
"""Frozen constants and shared deterministic I/O for ESD X0B/X1."""
from __future__ import annotations

import csv, hashlib, json, os, tempfile
from pathlib import Path

PRESPEC_TAG = "cross-corpus-reliability-prespec-v1"
X0A_TAG = "cross-corpus-x0a-audit-bundle-v1"
DESIGN_TAG = "cross-corpus-x0b-esd-design-v1"
APPROVED_STAGE1_MANIFEST_SHA256 = "948bcfa5439fb59f23982e8245049e81d298696035d3c6c692b91331e50234bb"
ENGLISH_SPEAKERS = tuple(f"{i:04d}" for i in range(11, 21))
ELIGIBLE_ESD_SPEAKERS = tuple(x for x in ENGLISH_SPEAKERS if x != "0014")
TECHNICAL_ACCESS_EXCLUSIONS = {"0014":"transcript_permission_denied"}
CHINESE_SPEAKERS = tuple(f"{i:04d}" for i in range(1, 11))
EXPECTED_CONDITIONS = 5
EXPECTED_PROMPTS = 350
COMPARABLE_FEATURES = ("f0_mean_hz", "f0_robust_range_hz", "energy_db", "duration_log_syllable_residual")
PAUSE_STATUS = "structurally_not_comparable_for_ESD_primary_gate"
GATE_E_THRESHOLDS={"speakers":8,"cells":500,"median_repetitions":3,"missingness_max":0.10,
                   "point_min":0.70,"bootstrap_lower_min":0.50,"loso_min":0.50}
FORBIDDEN_NAMES = ("daic", "e-daic", "ptsd-stop", "phq", "depression", "clinical", "msp-podcast", "sorted_by_emotion", "activations", "masks", "labels", "liwc", "emotion_prediction", "emotion-classification")

class ContractError(RuntimeError): pass

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f: f.write(text)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def write_json(path: Path, value) -> None:
    atomic_text(path,json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True)+"\n")

def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n",extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def ensure_output_outside_source(source: Path, output: Path) -> None:
    s=source.resolve(); o=output.resolve()
    if o==s or s in o.parents: raise ContractError("output must be outside the read-only ESD source root")

def reject_forbidden_path(path: Path) -> None:
    low=str(path).casefold()
    if any(x in low for x in FORBIDDEN_NAMES): raise ContractError("forbidden outcome/MSP/clinical path")
