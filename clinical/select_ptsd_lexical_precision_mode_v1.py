#!/usr/bin/env python3
"""Select the frozen PTSD-STOP RQ1 precision mode from blind runtimes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROTOCOL = "ptsd_stop_precision_within_person_amendment_v1"
CUTOFF = "2026-09-08T23:59:00-04:00"
IMPLEMENTED_CONCURRENT_FEATURE_PROCESSES = 4


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_ram_mib() -> float:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = float(value.strip().split()[0]) / 1024.0
    return values["MemAvailable"]


def projection(
    point_seconds: float,
    bootstrap_seconds: float,
    workers: int,
    replicates: int,
) -> float:
    return 2.0 * (
        math.ceil(4 / workers) * point_seconds
        + math.ceil(4 * replicates / workers) * bootstrap_seconds
    )


def select_mode(
    point_seconds: float,
    bootstrap_seconds: float,
    peak_rss_mib: float,
    available_mib: float,
    available_seconds: float,
) -> dict:
    numeric = (point_seconds, bootstrap_seconds, peak_rss_mib, available_mib)
    if any(not math.isfinite(value) or value <= 0 for value in numeric):
        raise ValueError("benchmark time and memory inputs must be finite and positive")
    workers = max(1, min(
        IMPLEMENTED_CONCURRENT_FEATURE_PROCESSES,
        48,
        math.floor(0.80 * available_mib / peak_rss_mib),
    ))
    full_seconds = projection(point_seconds, bootstrap_seconds, workers, 1000)
    reduced_seconds = projection(point_seconds, bootstrap_seconds, workers, 200)
    if full_seconds <= available_seconds:
        mode, target, minimum = "full_precision_B1000", 1000, 950
    elif reduced_seconds <= available_seconds:
        mode, target, minimum = "reduced_precision_B200", 200, 190
    else:
        mode, target, minimum = "point_only_compute_limited", 0, 0
    return {
        "safe_parallel_workers": workers,
        "projected_full_precision_seconds": full_seconds,
        "projected_reduced_precision_seconds": reduced_seconds,
        "mode": mode,
        "bootstrap_target_per_feature": target,
        "bootstrap_minimum_success": minimum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-technical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision-time", help="ISO time; default is current time")
    parser.add_argument("--available-ram-mib", type=float)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark_technical.read_text(encoding="utf-8"))
    exact = {
        "analysis": "ptsd_stop_lexical_rq1_outcome_blind_benchmark",
        "protocol": PROTOCOL,
        "feature": "f0_mean_hz",
        "bootstrap_replicate": 0,
        "bootstrap_status": "complete",
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_estimates_inspected": False,
        "scientific_estimates_exposed": False,
    }
    for key, expected in exact.items():
        if benchmark.get(key) != expected:
            raise ValueError(f"benchmark technical record mismatch for {key}")

    eastern = ZoneInfo("America/New_York")
    decision_time = (
        datetime.fromisoformat(args.decision_time)
        if args.decision_time else datetime.now(eastern)
    )
    if decision_time.tzinfo is None:
        raise ValueError("decision time must include a timezone")
    cutoff = datetime.fromisoformat(CUTOFF)
    seconds_available = max(0.0, (cutoff - decision_time).total_seconds())
    memory_available = (
        args.available_ram_mib
        if args.available_ram_mib is not None else available_ram_mib()
    )
    selected = select_mode(
        float(benchmark["point_fit_seconds"]),
        float(benchmark["bootstrap_fit_seconds"]),
        float(benchmark["peak_rss_mib"]),
        float(memory_available),
        seconds_available,
    )
    decision = {
        "analysis": "ptsd_stop_lexical_rq1_compute_precision_decision",
        "protocol": PROTOCOL,
        "decision_time": decision_time.isoformat(),
        "cutoff": CUTOFF,
        "available_seconds_at_decision": seconds_available,
        "benchmark_technical_sha256": file_sha(args.benchmark_technical),
        "point_fit_seconds": float(benchmark["point_fit_seconds"]),
        "bootstrap_replicate_0_seconds": float(benchmark["bootstrap_fit_seconds"]),
        "benchmark_peak_rss_mib": float(benchmark["peak_rss_mib"]),
        "available_ram_mib": float(memory_available),
        "contingency_multiplier": 2.0,
        "feature_count": 4,
        "cpu_ceiling": 48,
        "implemented_concurrent_feature_processes": (
            IMPLEMENTED_CONCURRENT_FEATURE_PROCESSES
        ),
        **selected,
        "labels_loaded": False,
        "outcomes_loaded": False,
        "scientific_estimates_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": decision["mode"],
        "bootstrap_target_per_feature": decision["bootstrap_target_per_feature"],
        "safe_parallel_workers": decision["safe_parallel_workers"],
        "scientific_estimates_inspected": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
