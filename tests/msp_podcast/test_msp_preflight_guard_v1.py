import unittest
from tools.msp_preflight_guard_v1 import evaluate_preflight

S64="a"*64; E64="b"*64; M64="c"*64; D64="d"*64

def structural():
    return {"audit":"msp_structural_projection_v1","projected_rows":203936,"emotion_fields_used":False,"acoustic_values_accessed":False,"reliability_values_accessed":False,"output_sha256":S64}

def alignment():
    return {"audit":"msp_textgrid_content_qa_v1","filename_coverage_exact":True,"files_passing_content_qa":203000,"files_failing_content_qa":936,"eligible_filename_list_sha256":E64,"private_token_manifest_sha256":M64,"transcript_text_accessed":False,"phone_marks_parsed_or_used":False,"word_marks_emitted":False,"emotion_fields_used":False,"audio_accessed":False,"acoustic_values_accessed":False,"reliability_values_accessed":False,"acoustic_processing_authorized":False}

def final_scope():
    return {"audit":"msp_final_scope_and_split_manifest_v1","final_scope_selected":True,"selected_speakers":100,"speaker_cap":100,"private_manifest_rows":12000,"primary_split_assigned":True,"alternative_splits_assigned":25,"emotion_fields_used":False,"audio_accessed":False,"acoustic_values_accessed":False,"reliability_values_accessed":False,"private_manifest_sha256":D64,"structural_sha256":S64,"eligible_list_sha256":E64}

def support_report():
    sm=lambda n: {"n":n,"min":2.0,"q1":3.0,"median":4.0,"q3":5.0,"max":6.0}
    return {"audit":"msp_structural_support_report_v1","status":"PASS","qualified_speakers":120,"selected_speakers":100,"speaker_cap":100,
        "qualified_eligible_utterances":sm(120),"selected_eligible_utterances":sm(100),"qualified_recording_units":sm(120),"selected_recording_units":sm(100),
        "selected_distinct_unordered_alt_partitions":sm(100),"reselection_performed":False,"selection_change_authorized":False,"significance_tests_run":False,
        "speaker_ids_emitted":False,"emotion_fields_used":False,"audio_accessed":False,"acoustic_values_accessed":False,"reliability_values_accessed":False,"clinical_outcomes_accessed":False,
        "structural_sha256":S64,"eligible_list_sha256":E64,"final_private_manifest_sha256":D64}

def model_lock():
    files=[
        "msp_primary_residualization_v2.py","msp_profile_reml_v1.py","msp_lexical_secondary_v1.py","msp_reliability_v2.py","msp_acoustic_adapter_v1.py",
        "test_msp_primary_residualization_v2.py","test_msp_profile_reml_v1.py","test_msp_lexical_secondary_v1.py","test_msp_reliability_v2.py","test_msp_acoustic_adapter_v1.py",
        "esd_frozen_acoustic_adapter_sha256","seamless_frozen_pipeline_sha256"]
    return {
        "status":"frozen","human_approved":True,
        "frozen_before_msp_acoustic_extraction":True,"frozen_before_msp_reliability_results":True,
        "primary_summary":"median residual","nuisance_model":{"formula":"explicit"},
        "primary_reliability_scalar":"corrected Pearson",
        "bootstrap":{"replicates":1000,"seed":13,"failure_rule":"no replacement","ci_rule":"linear percentiles"},
        "volume_domination_rule":"fixed before results",
        "transcript_alignment_content_qa_rule":"cross-corpus-msp-textgrid-content-qa-v1",
        "git_commit":"abc123","git_tag":"model-lock-v2",
        "implementation_status":"frozen",
        "implementation":{"files_sha256":{k:"f"*64 for k in files},
            "synthetic_test_total":43,"synthetic_test_passed":43,
            "real_msp_acoustic_values_accessed":False,"real_msp_reliability_values_accessed":False,
            "implementation_commit":"def456","implementation_tag":"impl-v2"}}

class Tests(unittest.TestCase):
    def test_actual_schema_chain_authorizes_with_frozen_lock(self):
        o=evaluate_preflight(structural(),alignment(),final_scope(),support_report(),model_lock()); self.assertTrue(o["msp_audio_processing_authorized"]); self.assertEqual(o["audit"],"msp_acoustic_preflight_guard_v4")
    def test_pending_implementation_fails(self):
        m=model_lock(); m["implementation_status"]="pending"; self.assertFalse(evaluate_preflight(structural(),alignment(),final_scope(),support_report(),m)["msp_audio_processing_authorized"])
    def test_incomplete_implementation_tests_fail(self):
        m=model_lock(); m["implementation"]["synthetic_test_passed"]=42; self.assertFalse(evaluate_preflight(structural(),alignment(),final_scope(),support_report(),m)["msp_audio_processing_authorized"])
    def test_missing_implementation_hash_fails(self):
        m=model_lock(); m["implementation"]["files_sha256"].pop("msp_reliability_v2.py"); self.assertFalse(evaluate_preflight(structural(),alignment(),final_scope(),support_report(),m)["msp_audio_processing_authorized"])
    def test_missing_support_report_fails(self):
        self.assertFalse(evaluate_preflight(structural(),alignment(),final_scope(),{},model_lock())["msp_audio_processing_authorized"])
    def test_support_hash_chain_mismatch_fails(self):
        r=support_report(); r["final_private_manifest_sha256"]="e"*64; o=evaluate_preflight(structural(),alignment(),final_scope(),r,model_lock()); self.assertFalse(o["checks"]["support_hash_chain"]); self.assertFalse(o["msp_audio_processing_authorized"])
    def test_support_reselection_fails(self):
        r=support_report(); r["reselection_performed"]=True; self.assertFalse(evaluate_preflight(structural(),alignment(),final_scope(),r,model_lock())["checks"]["structural_support_report"])
    def test_draft_lock_fails_closed(self):
        m=model_lock(); m["status"]="draft"; self.assertFalse(evaluate_preflight(structural(),alignment(),final_scope(),support_report(),m)["msp_audio_processing_authorized"])
    def test_alignment_audit_mismatch_fails(self):
        a=alignment(); a["audit"]="old_mock"; self.assertFalse(evaluate_preflight(structural(),a,final_scope(),support_report(),model_lock())["checks"]["transcript_alignment_qa"])
    def test_structural_hash_chain_mismatch_fails(self):
        f=final_scope(); f["structural_sha256"]="e"*64; o=evaluate_preflight(structural(),alignment(),f,support_report(),model_lock()); self.assertFalse(o["checks"]["structural_hash_chain"]); self.assertFalse(o["msp_audio_processing_authorized"])
    def test_eligible_hash_chain_mismatch_fails(self):
        f=final_scope(); f["eligible_list_sha256"]="e"*64; o=evaluate_preflight(structural(),alignment(),f,support_report(),model_lock()); self.assertFalse(o["checks"]["eligible_list_hash_chain"]); self.assertFalse(o["msp_audio_processing_authorized"])
    def test_more_than_100_speakers_fails(self):
        f=final_scope(); f["selected_speakers"]=101; self.assertFalse(evaluate_preflight(structural(),alignment(),f,support_report(),model_lock())["checks"]["final_scope"])
    def test_wrong_alt_split_count_fails(self):
        f=final_scope(); f["alternative_splits_assigned"]=24; self.assertFalse(evaluate_preflight(structural(),alignment(),f,support_report(),model_lock())["checks"]["final_scope"])
    def test_content_qa_must_not_have_used_audio(self):
        a=alignment(); a["audio_accessed"]=True; self.assertFalse(evaluate_preflight(structural(),a,final_scope(),support_report(),model_lock())["checks"]["transcript_alignment_qa"])
    def test_emotion_named_key_fails_anywhere(self):
        f=final_scope(); f["emotion_label"]="forbidden"; o=evaluate_preflight(structural(),alignment(),f,support_report(),model_lock()); self.assertFalse(o["checks"]["forbidden_keys_absent"]); self.assertFalse(o["msp_audio_processing_authorized"])
    def test_missing_manifest_hash_fails(self):
        f=final_scope(); f["private_manifest_sha256"]=""; self.assertFalse(evaluate_preflight(structural(),alignment(),f,support_report(),model_lock())["checks"]["final_scope"])

if __name__=='__main__': unittest.main()
