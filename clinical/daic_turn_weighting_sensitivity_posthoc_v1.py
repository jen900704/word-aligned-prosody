#!/usr/bin/env python3
"""POST HOC (2026-09-11). DAIC-WOZ: token-weighted vs turn-weighted participant summaries
and their PHQ-8 correlations. Reuses the frozen token loader and the frozen nuisance
model (pooled fit, position; plus syllables for duration). Nothing about extraction changes."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from daic_aggregate_reliability_blind_v1 import FEATURES as AGG, load_tokens
from prepare_daic_gate_licensed_inputs import FEATURE_MAP, MIN_FULL_PARTICIPANT_TOKENS
R = HERE.parent / "runtime_daic_v3"
OUT = R / "daic_turn_weighting_POSTHOC_SAFE.json"
SEED, B = 13, 1000

def residuals(frame, feature):
    work = frame[["participant_id", "utterance_index", "relative_word_position"]].copy()
    if feature == "syllable_adjusted_log_duration":
        work["y"] = pd.to_numeric(frame["log_word_duration"], errors="coerce")
        work["syl"] = pd.to_numeric(frame["authoritative_syllable_count"], errors="coerce")
        work = work.loc[np.isfinite(work["y"]) & np.isfinite(work["syl"]) & (work["syl"] > 0)].copy()
        X = np.column_stack([np.ones(len(work)), work["relative_word_position"].to_numpy(float), work["syl"].to_numpy(float)])
    else:
        work["y"] = pd.to_numeric(frame[feature], errors="coerce")
        work = work.loc[np.isfinite(work["y"])].copy()
        X = np.column_stack([np.ones(len(work)), work["relative_word_position"].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, work["y"].to_numpy(float), rcond=None)
    work["r"] = work["y"].to_numpy(float) - X @ beta
    return work

def boot_rho(x, y):
    rng = np.random.default_rng(np.random.PCG64(SEED)); n = len(x); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r): v.append(r)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def main():
    frame = load_tokens(R / "tokens_PRIVATE")
    frame["participant_id"] = frame["participant_id"].astype(str)
    labels = pd.read_csv(R / "daic_phq8_labels_PRIVATE.csv", dtype={"participant_id": str})
    frozen = pd.read_csv(R / "daic_participant_acoustic_summaries_PRIVATE.csv", dtype={"participant_id": str})
    res = {"protocol": "daic_turn_weighting_sensitivity_posthoc_v1", "post_hoc": True, "seed": SEED, "bootstrap_replicates": B,
           "definitions": {"token_weighted": "median of all token residuals per participant (frozen)",
                            "turn_weighted": "median over utterances of the within-utterance median residual",
                            "turn_weighted_mean": "mean over utterances of the within-utterance median residual"},
           "features": []}
    # speaking-amount vs PHQ-8
    counts = frame.groupby("participant_id").agg(tokens=("word_index", "size"), turns=("utterance_index", "nunique")).reset_index()
    cl = counts.merge(labels, on="participant_id")
    res["speaking_amount_vs_phq8"] = {"n": int(len(cl)),
        "tokens_rho": float(spearmanr(cl["tokens"], cl["PHQ8_Score"]).correlation),
        "turns_rho": float(spearmanr(cl["turns"], cl["PHQ8_Score"]).correlation),
        "median_tokens": float(cl["tokens"].median()), "median_turns": float(cl["turns"].median())}
    print("speaking amount vs PHQ-8:", res["speaking_amount_vs_phq8"], flush=True)
    for raw in AGG:
        feat = FEATURE_MAP[raw]; w = residuals(frame, raw)
        tok = w.groupby("participant_id")["r"].agg(["median", "size"]).reset_index()
        tok = tok.loc[tok["size"] >= MIN_FULL_PARTICIPANT_TOKENS].rename(columns={"median": "token_w"})
        per_turn = w.groupby(["participant_id", "utterance_index"])["r"].median().reset_index()
        turn = per_turn.groupby("participant_id")["r"].agg(turn_w="median", turn_w_mean="mean", n_turns="size").reset_index()
        d = tok.merge(turn, on="participant_id").merge(labels, on="participant_id").merge(frozen[["participant_id", feat]], on="participant_id")
        chk = float(np.nanmax(np.abs(d["token_w"] - d[feat])))
        sd = float(d["token_w"].std())
        shift = (d["turn_w"] - d["token_w"]) / sd
        r_tok = float(spearmanr(d["token_w"], d["PHQ8_Score"]).correlation)
        r_turn = float(spearmanr(d["turn_w"], d["PHQ8_Score"]).correlation)
        r_turnm = float(spearmanr(d["turn_w_mean"], d["PHQ8_Score"]).correlation)
        lo, hi, nb = boot_rho(d["turn_w"].to_numpy(), d["PHQ8_Score"].to_numpy())
        rec = {"feature": feat, "n": int(len(d)), "frozen_reproduction_max_abs_diff": chk,
               "token_vs_turn_spearman": float(spearmanr(d["token_w"], d["turn_w"]).correlation),
               "median_abs_shift_sd_units": float(shift.abs().median()), "p95_abs_shift_sd_units": float(shift.abs().quantile(0.95)),
               "rho_phq8_token_weighted": r_tok, "rho_phq8_turn_weighted": r_turn, "rho_phq8_turn_weighted_mean": r_turnm,
               "turn_weighted_ci": [lo, hi], "boot_successes": nb}
        res["features"].append(rec)
        print(f"{feat:32s} n={rec['n']} repro={chk:.2e} tok~turn r={rec['token_vs_turn_spearman']:.3f} "
              f"shift med={rec['median_abs_shift_sd_units']:.3f}SD p95={rec['p95_abs_shift_sd_units']:.3f}SD | "
              f"rho(PHQ8): token={r_tok:+.3f} turn={r_turn:+.3f} [{lo:+.3f},{hi:+.3f}] turn-mean={r_turnm:+.3f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1)); print("wrote", OUT)
if __name__ == "__main__": main()
