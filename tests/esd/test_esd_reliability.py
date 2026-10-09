import csv, hashlib, json, os, subprocess, sys, tarfile, tempfile, types, unittest, wave
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parents[2]/"esd"; sys.path.insert(0,str(HERE))
import esd_contract as c
import stage1_preflight as s1
import stage2_reliability as s2
import esd_stage2_adapter as adapter
import esd_stage2_qa as stage2qa
import esd_reml_reliability as reml
import verify_stage2_runtime as runtime
import frozen_word_aligner as local_align

CONDITIONS=("Neutral","Angry","Happy","Sad","Surprise")
OFFSETS={"Neutral":0,"Angry":350,"Happy":700,"Sad":1050,"Surprise":1400}

def wav(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),"wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(b"")

def fixture(root, prompts=3, include_chinese=True):
    for speaker in c.ENGLISH_SPEAKERS:
        transcript=[]
        for condition in CONDITIONS:
            for prompt in range(1,prompts+1):
                index=OFFSETS[condition]+prompt; utterance=f"{speaker}_{index:06d}"
                wav(root/speaker/condition/f"{utterance}.wav")
                transcript.append(f"{utterance}\tSpeaker {speaker} lexical prompt {prompt}.\t{condition}\n")
        (root/speaker/f"{speaker}.txt").write_text("".join(transcript),encoding="utf-8")
    if include_chinese: wav(root/"0001"/"Neutral"/"0001_000001.wav")

@contextmanager
def permission_denied_0014():
    real=s1.parse_transcript
    def parse(path,speaker):
        if speaker=="0014": raise PermissionError("synthetic permission denial")
        return real(path,speaker)
    with mock.patch.object(s1,"parse_transcript",side_effect=parse): yield

def run_v3(source,out,prompts=3):
    with permission_denied_0014(): return s1.run(source,out,prompts)

def safe_text(out): return "\n".join(path.read_text(errors="replace") for path in out.glob("*SAFE*"))

class LayoutParsingTests(unittest.TestCase):
    def test_ranges_and_out_of_range_rejection(self):
        for condition,(lower,upper,_) in s1.CONDITION_RANGES.items():
            self.assertEqual(s1.prompt_id_for(condition,lower),1); self.assertEqual(s1.prompt_id_for(condition,upper),350)
            with self.assertRaises(c.ContractError): s1.prompt_id_for(condition,lower-1)
            with self.assertRaises(c.ContractError): s1.prompt_id_for(condition,upper+1)
        with self.assertRaises(c.ContractError): s1.prompt_id_for("Neutral",351)

    def test_condition_surrounding_whitespace_and_0016_line_1623(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"0016.txt"
            path.write_text(" 0016_001623 \t  lexical  content stays  \tSurprise  \n")
            _,rows=s1.parse_transcript(path,"0016"); row=rows[0]
            self.assertEqual(row["condition"],"Surprise"); self.assertEqual(row["prompt_id"],223)
            self.assertEqual(row["transcript_text"],"lexical  content stays")

    def test_internal_lexical_content_not_altered(self):
        text="Keep  INTERNAL punctuation: can't-change!"
        self.assertEqual(s1.normalize_parallel_text(f"  {text}  "),text)

    def test_wrong_speaker_condition_and_global_range_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"x.txt"; path.write_text("0012_000001\tText\tNeutral\n")
            with self.assertRaisesRegex(c.ContractError,"speaker mismatch"): s1.parse_transcript(path,"0011")
            path.write_text("0011_001623\tText\tNeutral\n")
            with self.assertRaisesRegex(c.ContractError,"condition range"): s1.parse_transcript(path,"0011")

    def test_complete_1750_row_transcript(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); fixture(root,350)
            _,rows=s1.parse_transcript(root/"0016"/"0016.txt","0016"); self.assertEqual(len(rows),1750)

class FinalManifestTests(unittest.TestCase):
    def test_final_selected_speakers_and_0014_permission_exclusion(self):
        self.assertEqual(c.ELIGIBLE_ESD_SPEAKERS,("0011","0012","0013","0015","0016","0017","0018","0019","0020"))
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; fixture(source); decision=run_v3(source,base/"out")
            self.assertTrue(decision["stage1_pass"],decision["errors"]); self.assertEqual(decision["selected_speakers"],list(c.ELIGIBLE_ESD_SPEAKERS))
            self.assertEqual(decision["technical_access_exclusions"],[{"speaker_id":"0014","excluded_reason":"transcript_permission_denied"}])

    def test_0014_never_substituted_and_expected_small_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; out=base/"out"; fixture(source); decision=run_v3(source,out)
            with (out/"esd_manifest_PRIVATE.csv").open() as handle: rows=list(csv.DictReader(handle))
            self.assertNotIn("0014",{row["speaker_id"] for row in rows}); self.assertEqual(len(rows),9*5*3)
            self.assertNotIn("Speaker 0014",(out/"esd_manifest_PRIVATE.csv").read_text())
            self.assertEqual(decision["manifest_rows"],135)

    def test_complete_15750_fixture_and_350_wavs_per_condition(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; fixture(source,350,include_chinese=False)
            output=base/"out"; decision=run_v3(source,output,350); self.assertTrue(decision["stage1_pass"],decision["errors"])
            self.assertEqual(decision["manifest_rows"],15750); self.assertEqual(decision["expected_manifest_rows"],15750)
            with mock.patch.object(adapter,"file_sha",return_value=c.APPROVED_STAGE1_MANIFEST_SHA256):
                mapped=adapter.read_manifest(output/"esd_manifest_PRIVATE.csv")
            self.assertEqual(len(mapped),15750); self.assertNotIn("0014",{row["speaker_id"] for row in mapped})
            for speaker in c.ELIGIBLE_ESD_SPEAKERS:
                for condition in CONDITIONS: self.assertEqual(len(list((source/speaker/condition).glob("*.wav"))),350)

    def test_exact_own_speaker_join_and_manifest_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; out=base/"out"; fixture(source)
            first=run_v3(source,out); self.assertTrue(first["stage1_pass"],first["errors"])
            with (out/"esd_manifest_PRIVATE.csv").open() as handle: rows=list(csv.DictReader(handle))
            self.assertTrue(all(row["transcript_text"].startswith(f"Speaker {row['speaker_id']} ") for row in rows))
            digest=hashlib.sha256((out/"esd_manifest_PRIVATE.csv").read_bytes()).hexdigest()
            second=run_v3(source,out); self.assertEqual(digest,second["manifest_sha256"])

    def test_missing_transcript_row_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; fixture(source); path=source/"0011"/"0011.txt"
            path.write_text("\n".join(path.read_text().splitlines()[1:])+"\n")
            decision=run_v3(source,base/"out"); self.assertFalse(decision["stage1_pass"])
            self.assertTrue(any("transcript_row_count_failed" in error or "missing_audio_transcript_join" in error for error in decision["errors"]))

    def test_duplicate_manifest_key_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; fixture(source); wav(source/"0011"/"Neutral"/"0011_000001.WAV")
            decision=run_v3(source,base/"out"); self.assertFalse(decision["stage1_pass"])
            self.assertTrue(any("duplicate_speaker_condition_prompt_key" in error or "prompt_completeness_failed" in error for error in decision["errors"]))

    def test_condition_mismatch_after_normalization_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; fixture(source); path=source/"0016"/"0016.txt"
            path.write_text(path.read_text().replace("0016_001401\tSpeaker 0016 lexical prompt 1.\tSurprise","0016_001401\tSpeaker 0016 lexical prompt 1.\tNeutral  "))
            decision=run_v3(source,base/"out"); self.assertFalse(decision["stage1_pass"]); self.assertTrue(any("transcript_parse_failed" in error for error in decision["errors"]))

    def test_parallel_text_mismatch_is_diagnostic_not_failure_and_safe_is_private(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); source=base/"ESD"; out=base/"out"; fixture(source); path=source/"0011"/"0011.txt"
            path.write_text(path.read_text().replace("0011_000351\tSpeaker 0011 lexical prompt 1.","0011_000351\tPRIVATE VARIANT WORDS",1))
            decision=run_v3(source,out); self.assertTrue(decision["stage1_pass"],decision["errors"])
            self.assertGreater(decision["parallel_prompt_diagnostics"]["within_speaker_exact_mismatch"],0)
            safe=safe_text(out); self.assertNotIn("PRIVATE VARIANT WORDS",safe); self.assertNotIn("lexical prompt",safe); self.assertNotIn(str(source),safe)

class FrozenDesignTests(unittest.TestCase):
    def test_pause_and_gate_thresholds_unchanged(self):
        self.assertEqual(c.PAUSE_STATUS,"structurally_not_comparable_for_ESD_primary_gate")
        self.assertEqual(c.COMPARABLE_FEATURES,("f0_mean_hz","f0_robust_range_hz","energy_db","duration_log_syllable_residual"))
        self.assertEqual(c.GATE_E_THRESHOLDS,{"speakers":8,"cells":500,"median_repetitions":3,"missingness_max":.10,"point_min":.70,"bootstrap_lower_min":.50,"loso_min":.50})

    def test_no_msp_or_clinical_access(self):
        for path in ("/tmp/DAIC-WOZ","/tmp/E-DAIC","/tmp/PHQ-8.csv","/tmp/MSP-PODCAST/Audios","/tmp/MSP/labels","/tmp/MSP/sorted_by_emotion","/tmp/MSP/activations","/tmp/MSP/masks"):
            with self.assertRaises(c.ContractError): c.reject_forbidden_path(Path(path))

    def test_stage2_scope_checksum_and_lexical_identity_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/"esd_manifest_PRIVATE.csv").write_text("sample_id\nx\n")
            c.write_json(root/"stage1_decision_SAFE.json",{"stage1_pass":True,"acoustic_processing_performed":False,"manifest_sha256":c.APPROVED_STAGE1_MANIFEST_SHA256,"selected_speakers":list(c.ELIGIBLE_ESD_SPEAKERS)})
            with mock.patch.object(s2,"sha256",return_value=c.APPROVED_STAGE1_MANIFEST_SHA256):
                self.assertEqual(s2.load_stage1(root,c.APPROVED_STAGE1_MANIFEST_SHA256)[1],c.APPROVED_STAGE1_MANIFEST_SHA256)
                with self.assertRaises(c.ContractError): s2.load_stage1(root,"0"*64)
        self.assertIn("normalized_word",s2.REQUIRED_TOKEN_FIELDS); self.assertIn("word_index",s2.REQUIRED_TOKEN_FIELDS); self.assertIn("prompt_id",s2.REQUIRED_TOKEN_FIELDS)

class Stage2ReadinessTests(unittest.TestCase):
    def synthetic_rows(self):
        rows=[]
        for speaker_index in range(4):
            for word_index in range(5):
                for repetition in range(4):
                    rows.append({"speaker_id":f"s{speaker_index}","normalized_word":f"word{word_index}",
                                 "condition":f"c{repetition%2}","word_index":repetition,
                                 "prompt_id":999-repetition,"x":speaker_index*.4+word_index*.2+repetition*.01,
                                 "alignment_status":"aligned"})
        return rows

    def test_approved_manifest_sha_is_literal_and_other_rejected(self):
        self.assertEqual(c.APPROVED_STAGE1_MANIFEST_SHA256,"948bcfa5439fb59f23982e8245049e81d298696035d3c6c692b91331e50234bb")
        approval=json.loads((HERE/"stage1_realdata_approval.json").read_text())
        self.assertTrue(approval["approved"]); self.assertEqual(approval["manifest_sha256"],c.APPROVED_STAGE1_MANIFEST_SHA256)

    def test_exact_gate_pause_and_four_features(self):
        self.assertEqual(len(c.COMPARABLE_FEATURES),4); self.assertNotIn("pause"," ".join(c.COMPARABLE_FEATURES))
        reports=[{"feature":feature,"passed":True} for feature in c.COMPARABLE_FEATURES]
        self.assertTrue(s2.gate_e_pass(reports)[0]); reports[0]["passed"]=False; self.assertFalse(s2.gate_e_pass(reports)[0])
        reports[0]["passed"]=True; reports[1]["passed"]=False; reports[2]["passed"]=False; self.assertFalse(s2.gate_e_pass(reports)[0])

    def test_missing_adapters_fail_closed_and_2a_never_runs_full_job(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); lock=root/"lock.json"; lock.write_text(json.dumps({"alignment_adapter":{"ready":False,"reason":"missing"}}))
            with mock.patch.object(s2,"load_stage1",return_value=(root/"manifest.csv",c.APPROVED_STAGE1_MANIFEST_SHA256)), mock.patch.object(s2,"analyze") as analysis:
                with self.assertRaises(c.ContractError):
                    s2.stage2a_preflight(root,c.APPROVED_STAGE1_MANIFEST_SHA256,root/"out",HERE/"stage1_realdata_approval.json",lock,10)
                analysis.assert_not_called()
                report=json.loads((root/"out"/"stage2a_readiness_SAFE.json").read_text()); self.assertFalse(report["full_acoustic_job_invoked"])
            with self.assertRaises(c.ContractError): s2.stage2a_preflight(root,c.APPROVED_STAGE1_MANIFEST_SHA256,root/"other",HERE/"stage1_realdata_approval.json",lock,21)

    def test_no_liwc_emotion_prediction_msp_or_clinical_execution(self):
        source=(HERE/"stage2_reliability.py").read_text().casefold()
        self.assertNotIn("compute_prosodic_differentiation",source); self.assertNotIn("emotion classifier",source)
        for path in ("/tmp/DAIC-WOZ","/tmp/PHQ-8","/tmp/MSP-PODCAST","/tmp/MSP/labels"):
            with self.assertRaises(c.ContractError): c.reject_forbidden_path(Path(path))

    def test_missing_syllables_exclude_duration_only(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"tokens.csv"
            row={field:"" for field in s2.REQUIRED_TOKEN_FIELDS}
            row.update(sample_id="s",speaker_id="0011",condition="Neutral",
                       prompt_id="1",utterance_id="0011_000001",word_index="0",
                       normalized_word="word",alignment_status="aligned",
                       f0_mean_hz="100",f0_p10_hz="90",f0_p90_hz="110",
                       energy_db="-20",word_duration_sec="0.2",
                       authoritative_syllable_count="")
            with path.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=sorted(s2.REQUIRED_TOKEN_FIELDS))
                writer.writeheader(); writer.writerow(row)
            parsed=s2.read_tokens(path)[0]
            self.assertNotIn("parse_error",parsed)
            self.assertIsNone(parsed["authoritative_syllable_count"])
            prepared=s2.prepare_features([parsed])[0]
            self.assertEqual(prepared["f0_mean_hz"],100.0)
            self.assertNotIn("duration_log_syllable_residual",prepared)

class AdapterSyntheticTests(unittest.TestCase):
    def test_provenance_hashes_and_normalized_occurrence_positions(self):
        self.assertEqual(adapter.PROVENANCE["alignment_normalization"]["sha256"],"b6b08ff87f81b743e46fc0d54867b375f15acaa3320001f9d45fc5dff9ddf099")
        self.assertEqual(adapter.PROVENANCE["word_acoustics"]["sha256"],"3d84c495bfb2b0cae6f9230183e79e4076280735996a6f3d9f1b59ed1ecd8dc7")
        tokens=adapter.normalize_text("Word, middle WORD!")
        self.assertEqual([x["normalized_word"] for x in tokens],["word","middle","word"])
        self.assertEqual([x["word_index"] for x in tokens],[0,1,2]); self.assertEqual([x["lexical_occurrence_index"] for x in tokens],[0,0,1])

    def test_cmudict_checksum_fails_closed(self):
        self.assertEqual(adapter.CMUDICT_SHA256,"81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22")
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"cmudict.dict"; path.write_text("WORD  W ER1 D\n")
            with self.assertRaises(c.ContractError): adapter.verify_cmudict(path)

    def test_exact_alignment_checkpoint_identity_and_fail_closed(self):
        self.assertEqual(adapter.ALIGN_MODEL,"WAV2VEC2_ASR_BASE_960H")
        self.assertEqual(adapter.ALIGN_CHECKPOINT_BASENAME,"wav2vec2_fairseq_base_ls960_asr_ls960.pth")
        self.assertEqual(adapter.ALIGN_CHECKPOINT_SHA256,"488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d")
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaisesRegex(c.ContractError,"absent"): adapter.verify_alignment_checkpoint(root)
            (root/adapter.ALIGN_CHECKPOINT_BASENAME).write_bytes(b"wrong")
            with self.assertRaisesRegex(c.ContractError,"checksum"): adapter.verify_alignment_checkpoint(root)

    def test_runtime_verifier_rejects_absent_and_wrong_checkpoint(self):
        fake_torch=types.SimpleNamespace(__version__="2.5.1+cu121",cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1))
        fake_torchaudio=types.SimpleNamespace(__version__="2.5.1+cu121")
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules,{"torch":fake_torch,"torchaudio":fake_torchaudio}), mock.patch.object(runtime,"verify_cmudict",return_value={}), mock.patch.object(runtime.shutil,"which",return_value="/usr/bin/ffmpeg"):
            root=Path(td); cmu=root/"cmudict.dict"; cmu.write_text("synthetic")
            absent=runtime.verify(cmu,root/"models",root/"absent.json",Path(sys.executable))
            self.assertFalse(absent["checks"]["frozen_alignment_checkpoint_verified"])
            models=root/"models"; models.mkdir(); (models/adapter.ALIGN_CHECKPOINT_BASENAME).write_bytes(b"wrong")
            wrong=runtime.verify(cmu,models,root/"wrong.json",Path(sys.executable))
            self.assertFalse(wrong["checks"]["frozen_alignment_checkpoint_verified"])

    def test_deterministic_exact_ten_smoke_selection(self):
        conditions=("Neutral","Angry","Happy","Sad","Surprise"); rows=[]
        for i in range(10):
            rows.append({"speaker_id":c.ELIGIBLE_ESD_SPEAKERS[i%9],"condition":conditions[i%5],"prompt_id":str(1+(i*37)%350),"sample_id":f"u{i}"})
        first=adapter.select_smoke(rows,10); second=adapter.select_smoke(list(reversed(rows)),10)
        self.assertEqual([x["sample_id"] for x in first],[x["sample_id"] for x in second]); self.assertEqual(len(first),10)
        with self.assertRaises(c.ContractError): adapter.select_smoke(rows,9)

    def test_alignment_failure_and_missing_syllable_are_explicit(self):
        class Failed:
            def align(self,path,tokens): raise RuntimeError("synthetic")
        item={"sample_id":"s","speaker_id":"0011","condition":"Neutral","prompt_id":"1","utterance_id":"u","audio_path":"/synthetic.wav","transcript_text":"same same"}
        rows=adapter.process_utterance(item,Failed(),{}); self.assertTrue(all(x["alignment_status"]=="failed" for x in rows)); self.assertTrue(all(x["feature_status"]=="not_attempted" for x in rows))
        class Aligned:
            def align(self,path,tokens): return [{"start":0.0,"end":.1,"score":1.0}]
        with mock.patch.object(adapter,"read_audio_interval",side_effect=RuntimeError("extract")):
            row=adapter.process_utterance({**item,"transcript_text":"unknownword"},Aligned(),{})[0]
        self.assertEqual(row["syllable_lookup_status"],"missing"); self.assertEqual(row["feature_status"],"failed")

    def test_invalid_word_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"a.wav"; wav(path)
            with self.assertRaises(c.ContractError): adapter.read_audio_interval(path,.2,.1)

    def test_qa_duplicate_failures_f0_and_syllable_missingness(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); directory=root/"by_utterance"; directory.mkdir()
            row={field:"" for field in adapter.FIELDS}; row.update(sample_id="s",word_index="0",alignment_status="unaligned",feature_status="failed",feature_error_code="RuntimeError")
            with (directory/"s.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,adapter.FIELDS); writer.writeheader(); writer.writerow(row); writer.writerow(row)
            report=stage2qa.audit(root,root/"qa.json",1)["metrics"]
            self.assertEqual(report["duplicate_acoustic_evidence"],1); self.assertEqual(report["alignment_failures"],2)
            self.assertEqual(report["extraction_exceptions"],2); self.assertEqual(report["f0_missing"],2); self.assertEqual(report["syllable_lookup_failures"],2)

    def test_full_extraction_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(adapter,"read_manifest",return_value=[]):
            with self.assertRaises(c.ContractError): adapter.run_scope(Path("m"),Path(td),Path("c"),Path("model"),"full",None)

    def test_slurm_smoke_gpu_no_hostname_and_full_array(self):
        smoke=(HERE/"slurm_stage2a_smoke.sbatch").read_text(); full=(HERE/"slurm_stage2b_full.sbatch").read_text()
        self.assertIn("#SBATCH --partition=gpu",smoke); self.assertIn("#SBATCH --gres=gpu:1",smoke)
        self.assertNotIn("c10",smoke); self.assertNotIn("--nodelist",smoke); self.assertIn("#SBATCH --array=0-31",full)
        self.assertIn('${CODE_DIR:?',smoke); self.assertIn('[[ "$CODE_DIR" == /* ]]',smoke)
        self.assertIn('"$CODE_DIR/run_stage2_smoke.sh"',smoke)
        self.assertNotIn("BASH_SOURCE",smoke); self.assertNotIn('dirname "$0"',smoke); self.assertNotIn("dirname --",smoke)

    def test_missing_code_dir_runner_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); output=root/"must_not_exist"
            environment={**os.environ,"CODE_DIR":str(root),"STAGE1_ROOT":"/synthetic/stage1","CMUDICT_PATH":"/synthetic/cmudict","ALIGN_MODEL_DIR":"/synthetic/model","SMOKE_OUTPUT_ROOT":str(output)}
            result=subprocess.run(["bash",str(HERE/"slurm_stage2a_smoke.sbatch")],env=environment,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0); self.assertIn("smoke runner absent",result.stderr)
            self.assertFalse(output.exists())

    def test_run2_submission_exports_shared_code_dir(self):
        handoff=(HERE/"../../notes/cross_corpus_x2_smoke_handoff.md").resolve().read_text()
        self.assertIn("CODE_DIR=/cluster/project/prosody_esd_x2/bundle_v1/clsp_esd_reliability",handoff)
        self.assertIn("SMOKE_OUTPUT_ROOT=/cluster/project/prosody_esd_x2/smoke_v1_run2",handoff)
        self.assertNotIn("SMOKE_OUTPUT_ROOT=/cluster/project/prosody_esd_x2/smoke_v1_run1",handoff)

    def test_direct_baseline_and_dependency_separation(self):
        smoke=(HERE/"slurm_stage2a_smoke.sbatch").read_text(); analysis=(HERE/"requirements-stage2c-analysis.txt").read_text().casefold()
        self.assertIn("/cluster/env/bin/python",smoke)
        self.assertFalse((HERE/"setup_project_env.sh").exists()); self.assertFalse((HERE/"requirements-stage2-whisperx-overlay.txt").exists())
        self.assertIn("statsmodels==0.14.4",analysis)
        self.assertEqual(runtime.VERSION_LOCK,{"python":"3.10.20","torch":"2.5.1+cu121","torchaudio":"2.5.1+cu121","numpy":"1.26.4","soundfile":"0.13.1","scipy":"1.15.3","pandas":"2.3.3"})

    def test_local_aligner_import_graph_and_provenance(self):
        imports=runtime.imported_roots(HERE/"frozen_word_aligner.py")|runtime.imported_roots(HERE/"esd_stage2_adapter.py")
        self.assertFalse(imports&{"whisperx","pyannote","faster_whisper","ctranslate2"})
        self.assertEqual(local_align.HISTORICAL_ALIGNMENT_SHA256,"2d5b999ea5fc4a2e0e449471baa954fe1d9af5919aed325515578c0a8a734660")
        self.assertEqual(local_align.ALIGN_MODEL,"WAV2VEC2_ASR_BASE_960H")
        source=(HERE/"frozen_word_aligner.py").read_text().casefold()
        self.assertNotIn("http://",source); self.assertNotIn("https://",source); self.assertNotIn("download_url",source)

    def test_local_aligner_deterministic_repeated_and_omitted_tokens(self):
        segments=[local_align.Segment("a",0,1,.9),local_align.Segment("|",1,2,.8),local_align.Segment("a",2,3,.7)]
        first=local_align.aggregate_words("a a",[0,1,2],segments,.2,0.0)
        second=local_align.aggregate_words("a a",[0,1,2],segments,.2,0.0)
        self.assertEqual(first,second); self.assertEqual([row["word"] for row in first],["a","a"])
        self.assertTrue(all("start" in row and "end" in row for row in first))
        omitted=local_align.aggregate_words("a ?",[0],[local_align.Segment("a",0,1,.9)],.2,0.0)
        self.assertEqual(omitted[1],{"word":"?"})

    def test_historical_nonclinical_parity_fixture(self):
        parity=json.loads((HERE/"tests/historical_alignment_parity.json").read_text())
        self.assertTrue(parity["passed"]); self.assertEqual(parity["historical_alignment_sha256"],local_align.HISTORICAL_ALIGNMENT_SHA256)
        self.assertEqual(parity["historical_timings"],parity["local_timings"])
        self.assertEqual(parity["historical_omissions"],parity["local_omissions"])
        self.assertEqual(parity["maximum_observed_timing_delta_sec"],0.0)
        self.assertEqual(parity["prospective_timing_tolerance_sec"],0.001)

    def test_noncode_resources_excluded_from_bundle(self):
        bundle=HERE.parent/"clsp_esd_reliability_bundle.tar.gz"
        if bundle.is_file():
            with tarfile.open(bundle,"r:gz") as archive: names=archive.getnames()
            self.assertFalse(any(name.endswith(".pth") or name.endswith("cmudict.dict") or name.endswith(".whl") for name in names))

    def test_smoke_path_has_no_gate_e(self):
        shell=(HERE/"run_stage2_smoke.sh").read_text(); self.assertNotIn("stage2_reliability.py",shell); self.assertNotIn("gate_e",shell.casefold())

class ProspectiveEstimatorTests(unittest.TestCase):
    @staticmethod
    def dummy_fit(rows,feature,condition):
        value=(reml.CONDITIONS.index(condition)+1)/10
        return {"condition":condition,"method":"REML","repeatability":value,"token_count":10**(reml.CONDITIONS.index(condition)+1),"matched_cell_count":1,"converged":True}

    def test_reml_five_conditions_and_unweighted_mean(self):
        summary=reml.five_condition_summary([],"x",self.dummy_fit)
        self.assertEqual(reml.ESTIMATION_METHOD,"REML"); self.assertEqual(len(summary["condition_estimates"]),5)
        self.assertAlmostEqual(summary["equal_condition_mean"],.3); self.assertFalse(summary["token_count_weighted"])

    def test_bootstrap_frozen_constants_and_duplicate_draw_relabeling(self):
        self.assertEqual((reml.BOOTSTRAP_REPLICATES,reml.BOOTSTRAP_SEED,reml.BOOTSTRAP_REVIEW_MIN_SUCCESS),(1000,13,950))
        rows=[{"speaker_id":"0011","condition":"Neutral","normalized_word":"w"}]
        class Same:
            def choice(self,items): return "0011"
        sampled,draws=reml.bootstrap_sample(rows,Same()); self.assertEqual(len(draws),9)
        self.assertEqual(len({row["speaker_id"] for row in sampled}),9)

    def test_bootstrap_950_review_rule_and_failure_accounting(self):
        rows=[{"speaker_id":speaker} for speaker in c.ELIGIBLE_ESD_SPEAKERS]
        success=reml.speaker_cluster_bootstrap(rows,"x",self.dummy_fit,1000,13)
        self.assertEqual(success["successful_replicates"],1000); self.assertFalse(success["technical_review_required"])
        def fail(rows,feature,condition): raise RuntimeError("nonconverged")
        failed=reml.speaker_cluster_bootstrap(rows,"x",fail,1000,13)
        self.assertEqual(failed["failed_replicates"],1000); self.assertTrue(failed["technical_review_required"])

    def test_exact_nine_loso_with_eight_speakers(self):
        rows=[{"speaker_id":speaker} for speaker in c.ELIGIBLE_ESD_SPEAKERS]
        result=reml.leave_one_speaker_out(rows,"x",self.dummy_fit)
        self.assertEqual(result["deletion_count"],9); self.assertTrue(all(x["remaining_speaker_count"]==8 for x in result["results"]))

    def test_lexical_sensitivity_support_and_first_three_order(self):
        rows=[]
        for speaker in c.ELIGIBLE_ESD_SPEAKERS[:8]:
            for utterance in ("u4","u2","u1","u3"):
                rows.append({"speaker_id":speaker,"normalized_word":"shared","condition":"Neutral","utterance_id":utterance,"word_index":2,"x":1.0})
        balanced,words=reml.lexical_balance(rows,"x","Neutral")
        self.assertEqual(words,{"shared"}); self.assertEqual(len(balanced),8*3)
        for speaker in c.ELIGIBLE_ESD_SPEAKERS[:8]: self.assertEqual([r["utterance_id"] for r in balanced if r["speaker_id"]==speaker],["u1","u2","u3"])
        sensitivity=reml.lexical_composition_sensitivity(rows*5,"x",self.dummy_fit,.3)
        self.assertFalse(sensitivity["part_of_gate_e"])

    def test_cross_condition_secondary_and_smoke_inference_isolation(self):
        analyzed=reml.analyze_feature([{"speaker_id":speaker} for speaker in c.ELIGIBLE_ESD_SPEAKERS],"x",self.dummy_fit,10)
        self.assertFalse(analyzed["cross_condition"]["part_of_gate_e"])
        smoke=(HERE/"run_stage2_smoke.sh").read_text(); adapter_source=(HERE/"esd_stage2_adapter.py").read_text()
        for forbidden in ("esd_reml_reliability","bootstrap","leave_one_speaker_out","lexical_composition_sensitivity"):
            self.assertNotIn(forbidden,smoke); self.assertNotIn(forbidden,adapter_source)

    def test_real_stage1_path_static_guard(self):
        paths=[HERE/"../../notes/cross_corpus_x2_smoke_handoff.md",HERE/"../../notes/cross_corpus_x2_clsp_execution_handoff.md"]
        text="\n".join(path.resolve().read_text() for path in paths)
        self.assertIn("stage1_final_v3_run1",text)
        self.assertNotRegex(text,r"stage1_final_v3(?:[\"/, ]|$)")

if __name__=="__main__": unittest.main()
