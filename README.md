# Word-aligned prosody reflects speaker differences more than depression or PTSD severity

Analysis code for the paper by Hsiang-Chen Yeh, Aurosweta Mahapatra, Shreeram Suresh Chandra, Ryan L. Boyd and Berrak Sisman (submitted to ICASSP 2027; preprint: arXiv link to be added).

The paper asks whether a prosodic value aligned to one word is a stable property of the speaker, and whether speaker-level summaries of such values track depression (PHQ-8) or PTSD severity (PCL). It uses four corpora: ESD and MSP-Podcast for the stability questions (RQ1), and DAIC-WOZ and PTSD-STOP for the clinical questions (RQ2).

## What this repository contains

Code only. It contains no audio, transcripts, word-level features, symptom scores or participant-level files. All four corpora are distributed by their owners under their own terms, and PTSD-STOP data and outputs never leave the host laboratory's servers. Scripts write participant-level or token-level outputs to files whose names contain `PRIVATE`, with restricted file permissions, and aggregate outputs to files whose names contain `SAFE`.

The scripts are the versions that produced the numbers in the paper. For the public release they were changed in three ways only: absolute paths and interpreter paths were replaced by environment variables, two cluster-specific runtime fingerprint guards were removed together with the job-script lines that called them, and test import paths were adjusted to this layout. `PROVENANCE.tsv` lists the SHA-256 of every file as it was run and as released.

## Layout

| Folder | Contents |
|---|---|
| `esd/` | Shared word alignment and acoustic features (`frozen_word_aligner.py`, `esd_stage2_adapter.py`), ESD extraction, the crossed REML estimator, and the ESD speaker-summary analysis |
| `msp_podcast/` | MSP-Podcast speaker selection, feature extraction, split-half consistency (Gate M), the unidentified-speaker exclusion, the aggregation curve and the within-sex rerun; `slurm/` holds the job scripts in run order |
| `clinical/` | DAIC-WOZ and PTSD-STOP extraction, reliability, symptom associations and post hoc analyses |
| `posthoc_sensitivity/` | Post hoc extraction checks on ESD and the nine-speaker MSP-Podcast word-level fits |
| `tests/` | Unit tests on synthetic data |

## Where each result comes from

**Shared features (Sec. 2.2).** `esd/frozen_word_aligner.py` is a scoped copy of the WhisperX 3.3.1 CTC alignment functions (torchaudio `WAV2VEC2_ASR_BASE_960H`). `analyze_word` in `esd/esd_stage2_adapter.py` computes mean F0 and the P10 to P90 F0 range by autocorrelation over voiced frames, energy as 20 log10 of word RMS, and log word duration with CMUdict syllable counts. MSP-Podcast word boundaries come from the forced alignments distributed with the corpus (`msp_podcast/tools/msp_textgrid_content_qa_v1.py`); its acoustic step calls the same `analyze_word`, and the runner checks that file's SHA-256 before use.

**ESD cross-context recurrence (Table 1, Fig. 1).** `esd/stage1_preflight.py` builds the manifest, `esd/stage2b_full.py` aligns and extracts in shards, and `esd/esd_reml_reliability.py` fits the speaker, word and speaker-by-word model per condition, with bootstrap and leave-one-speaker-out runs orchestrated by `esd/esd_stage2c_parallel.py` (`esd/launch_stage2c_parallel.template.sh` and the `slurm_stage2c_parallel_*.sbatch` files). `esd/disclose_esd_condition_components.py` writes the variance components shown in Fig. 1; the raw-token share of 0.23 to 0.37 in Sec. 3.1 is (S + S×W) / (S + S×W + residual) computed from those percentages. `esd/esd_component_algebra_J2_v1.py` writes the closed forms that link the components to both estimands (recurrence of one token, and consistency of a speaker mean over k tokens).

**ESD speaker summaries (Sec. 5.1).** `esd/esd_msp_style_speaker_stability_v1.py` applies the MSP-Podcast speaker-summary procedure to the same ESD recordings.

**Post hoc extraction checks and nine-speaker draws (Sec. 5.2).** `posthoc_sensitivity/esd_postreview_extract_v1.py` re-extracts ESD features under shifted, expanded, contracted, jittered and central-half word boundaries and with Praat F0 (parselmouth); `esd_postreview_refit_v1.py` refits the frozen estimator on each variant. `esd_voiced_sensitivity_v1.py` requires a minimum number of voiced frames, and `esd_utterance_facet_v1.py` adds the utterance random effect. `msp_postreview_crossed_v1.py` runs the same crossed model on 20 seeded draws of nine MSP-Podcast speakers.

**MSP-Podcast consistency (Table 1, Sec. 5.3, Fig. 2).** The job scripts in `msp_podcast/slurm/` run in numbered order: corpus alignment QA, speaker selection with the 25 prespecified alternative splits (`tools/msp_final_scope_manifest_v1.py`), extraction (`tools/msp_acoustic_extraction_runner_v3.py`), merging, and Gate M (`tools/msp_final_analysis_v1.py`, which uses `tools/msp_primary_residualization_v2.py` and `tools/msp_reliability_v2.py`). `msp_amendment_nounk_v1.py` removes the pooled unidentified-speaker label before any model is fitted and reruns Gate M, which gives the 99-speaker results in the paper. `tools/msp_aggregation_curve_J1_v1.py` on that input gives Fig. 2. `msp_withinsex_v1.py` reruns Gate M within each corpus-labelled sex.

**DAIC-WOZ (Sec. 5.4).** `clinical/daic_wordlocal_extract_v3.py` extracts participant speech only, `clinical/daic_aggregate_reliability_blind_v1.py` and `clinical/unblind_daic_aggregate_reliability.py` give the odd/even reliability, and `clinical/prepare_daic_gate_licensed_inputs.py` with `clinical/gated_clinical_correlation.py` gives the PHQ-8 correlations, bootstrap intervals and Benjamini-Hochberg correction. PHQ-8 scores are read only after a feature passes its reliability gate.

**PTSD-STOP (Table 2, Sec. 5.4).** `clinical/ptsd_stop_wordlocal_extract_allspeech_v2.py` extracts all diary speech (it reuses `clinical/ptsd_stop_wordlocal_extract_v1.py` and the DAIC-WOZ extractor), `clinical/build_ptsd_stop_identity_authority_v1.py` maps recordings to participants, and `clinical/ptsd_stop_aggregate_reliability_blind_v1.py` with `clinical/unblind_ptsd_stop_aggregate_reliability_v1.py` gives the recording-set reliability. `clinical/prepare_ptsd_stop_gate_licensed_inputs_v2.py` with `clinical/analyze_ptsd_stop_pcl_association_v1.py` gives the between-person correlations, and `clinical/analyze_ptsd_stop_within_person_pcl_v1.py` gives the earliest-to-latest change correlations. `clinical/analyze_ptsd_stop_multilevel_posthoc_v1.py` is the post hoc within-person daily model.

**Equivalence bounds (Sec. 5.4).** `clinical/posthoc_equivalence_bounds.py` recomputes the post hoc TOST bounds (0.16 for DAIC-WOZ, 0.26 for PTSD-STOP between persons, 0.23 for change) from the reported correlations. It was written for this release and needs no data.

**Not included.** The post hoc single-day ICC of the daily PCL (0.56 in the Limitations) was computed on the host laboratory's server and its script is not part of this release.

**Run but not reported in the paper.** Participant-by-word crossed fits on the clinical corpora (`clinical/daic_lexical_reliability_blind_v1.py`, `clinical/ptsd_stop_lexical_reliability_blind_v1.py`, `clinical/select_ptsd_lexical_precision_mode_v1.py`), the DAIC-WOZ and PTSD-STOP aggregation curves (`clinical/aggregation_curve_J1_daic_v1.py`, `clinical/aggregation_curve_J1_diagnostic_v1.py`, `clinical/aggregation_curve_O_ptsd_v1.py`), attenuation and detectable-effect bounds (`clinical/clinical_bounds_J4_J5_v1.py`), a DAIC-WOZ sex-adjusted analysis (`clinical/daic_sex_sensitivity_L_v1.py`), a DAIC-WOZ turn-weighting check (`clinical/daic_turn_weighting_sensitivity_posthoc_v1.py`), the role audit of the earlier role-filtered PTSD-STOP extraction (`clinical/make_ptsd_stop_role_audit_manifest_v1.py`), and an MSP-Podcast word-level secondary analysis (`msp_podcast/tools/msp_lexical_secondary_analysis_v1.py`).

## Running the code

The corpora are not included. To reproduce an analysis you need your own licensed copy of the corpus:

* ESD: https://hltsingapore.github.io/ESD/
* MSP-Podcast (release 1.12 was used): https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html
* DAIC-WOZ: https://dcapswoz.ict.usc.edu/
* PTSD-STOP is not publicly available; access is through the study team under its IRB.

Paths are set with environment variables:

| Variable | Meaning |
|---|---|
| `WAP_REPO` | this repository (used by the job scripts) |
| `WAP_WORK_ROOT` | working directory for ESD and MSP-Podcast inputs and outputs (default `work`) |
| `WAP_PTSD_ROOT` | PTSD-STOP workspace on the server that holds the data (default `work/ptsd_stop`) |
| `WAP_DAIC_ROOT`, `WAP_DAIC_OUT` | DAIC-WOZ participant archives and output folder |
| `WAP_RESOURCE_DIR` | CMUdict (`cmudict.dict`, checked by SHA-256) and the alignment checkpoint |
| `MSP_PODCAST_ROOT`, `MSP_LABELS_CSV`, `MSP_MODEL_LOCK` | MSP-Podcast release, its `labels_consensus.csv`, and the frozen model-lock file |
| `PYTHON`, `WAP_VENV` | interpreter or virtual environment for the job scripts |

Two Python environments were used. `requirements-analysis.txt` is the mixed-model environment for the ESD estimator, MSP-Podcast Gate M and the post hoc refits. `requirements-extraction.txt` is the environment used for DAIC-WOZ extraction and analysis and for the tests.

## Tests

```
python -m pytest tests
```

The suite runs on synthetic data. On Linux with Python 3.12 and without PyTorch, 197 tests pass and 9 are skipped. The skipped tests check provenance files from the original cluster runs that are not part of this release; `tests/conftest.py` gives the reason for each. The two PTSD-STOP extractor test modules need PyTorch and torchaudio; with `requirements-extraction.txt` installed they pass as well. On Windows, four further tests fail for platform reasons only: Windows keeps their temporary files locked during cleanup, treats file names case-insensitively, or has no `bash` for the job-script check.

## Citation

```
@misc{yeh2027wordaligned,
  title  = {Word-Aligned Prosody Reflects Speaker Differences More Than Depression or {PTSD} Severity},
  author = {Yeh, Hsiang-Chen and Mahapatra, Aurosweta and Suresh Chandra, Shreeram and Boyd, Ryan L. and Sisman, Berrak},
  year   = {2026},
  note   = {Submitted to ICASSP 2027}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The corpora keep their own licenses.
