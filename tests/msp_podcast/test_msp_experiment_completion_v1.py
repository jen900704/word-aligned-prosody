import json,tempfile,unittest
from pathlib import Path
from tools.msp_experiment_completion_v1 import *
class Tests(unittest.TestCase):
 def make(self):
  td=tempfile.TemporaryDirectory(); d=Path(td.name); p=d/'p'; l=d/'l'; m=d/'m'; r=d/'r'
  p.write_text(json.dumps({'audit':'msp_final_analysis_technical_safe_v1','status':'PASS','features':{},'reliability_values_emitted':False,'confidence_interval_values_emitted':False,'gate_m_status_emitted':False,'gate_m_status_inspected_for_safe':False,'speaker_ids_emitted':False}))
  l.write_text(json.dumps({'audit':'msp_crossfitted_lexical_secondary_technical_safe_v1','status':'PASS','reliability_values_emitted':False,'gate_m_status_emitted':False,'secondary_only':True,'can_modify_gate_m':False}))
  m.write_text(json.dumps({'audit':'msp_acoustic_merge_v1','status':'PASS','merged_token_rows':10,'acoustic_failed_tokens':0,'reliability_values_accessed':False,'clinical_outcomes_accessed':False}))
  r.write_text(json.dumps({'audit':'msp_production_readiness_v1','status':'PASS','checks':{'a':True}})); return td,p,l,m,r
 def test_primary_ready_without_secondary(self):
  td,p,l,m,r=self.make(); z=validate(p,m,r); self.assertEqual(z['primary_result_review_state'],'READY_FOR_RESULT_REVIEW'); self.assertEqual(z['lexical_secondary_state'],'PENDING_OR_NOT_RUN'); td.cleanup()
 def test_secondary_optional_complete(self):
  td,p,l,m,r=self.make(); self.assertEqual(validate(p,m,r,l)['lexical_secondary_state'],'COMPLETE'); td.cleanup()
 def test_primary_review_blocks_ready(self):
  td,p,l,m,r=self.make(); x=json.loads(p.read_text()); x['status']='TECHNICAL_REVIEW'; p.write_text(json.dumps(x)); z=validate(p,m,r); self.assertEqual(z['status'],'TECHNICAL_REVIEW'); td.cleanup()
 def test_blinding_violation_fails(self):
  td,p,l,m,r=self.make(); x=json.loads(p.read_text()); x['gate_m_status_emitted']=True; p.write_text(json.dumps(x));
  with self.assertRaises(MSPCompletionError): validate(p,m,r)
  td.cleanup()
if __name__=='__main__': unittest.main()
