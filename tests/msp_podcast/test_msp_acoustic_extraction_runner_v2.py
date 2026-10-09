import csv,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.msp_acoustic_extraction_runner_v2 import *

def sh(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wc(p,fields,rows):
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
class Tests(unittest.TestCase):
    def make(self):
        td=tempfile.TemporaryDirectory(); d=Path(td.name); audio=d/'audio'; audio.mkdir(); (audio/'a.wav').write_bytes(b'x')
        final=d/'final.csv'; token=d/'token.csv'; fs=d/'fs.json'; als=d/'al.json'; pf=d/'pf.json'; out=d/'out.csv'; safe=d/'safe.json'
        ff=['filename','speaker_id','recording_unit_id','primary_half']+[f'alt_{i:02d}_half' for i in range(25)]
        r={'filename':'a.wav','speaker_id':'s1','recording_unit_id':'u1','primary_half':'A'}; r.update({f'alt_{i:02d}_half':'A' for i in range(25)}); wc(final,ff,[r])
        tf=list(TOKEN_REQ); rows=[]
        for i in range(3): rows.append({'filename':'a.wav','speaker_id':'s1','recording_unit_id':'u1','word_index':str(i),'normalized_word':f'w{i}','word_start_sec':str(i*.1),'word_end_sec':str(i*.1+.08),'word_duration_sec':'.08','authoritative_syllable_count':'1','syllable_lookup_status':'first_canonical'})
        wc(token,tf,rows); fs.write_text(json.dumps({'audit':'msp_final_scope_and_split_manifest_v1','private_manifest_sha256':sh(final)})); als.write_text(json.dumps({'audit':'msp_textgrid_content_qa_v1','private_token_manifest_sha256':sh(token)})); pf.write_text(json.dumps({'audit':'msp_acoustic_preflight_guard_v4','msp_audio_processing_authorized':True,'checks':{'a':True},'scientific_choices_made_by_guard':False,'reliability_values_accessed':False})); return td,audio,final,token,fs,als,pf,out,safe
    def reader(self,p): return np.ones((16000,2),dtype=np.float32),16000
    def analyzer(self,x,sr): return {'f0_mean_hz':100.,'f0_robust_range_hz':10.,'energy_db':-20.}
    def test_valid_run_and_position(self):
        td,a,f,t,fs,als,pf,o,s=self.make(); z=run(pf,als,fs,t,f,a,o,s,0,1,self.reader,self.analyzer);
        with o.open(newline='',encoding='utf-8') as fh: rows=list(csv.DictReader(fh)); self.assertEqual(len(rows),3); self.assertEqual([float(x['relative_position']) for x in rows],[0,.5,1]); self.assertEqual(z['files_failed'],0); self.assertEqual(z['acoustic_failed_tokens'],0); td.cleanup()
    def test_preflight_required(self):
        td,a,f,t,fs,als,pf,o,s=self.make(); x=json.loads(pf.read_text()); x['msp_audio_processing_authorized']=False; pf.write_text(json.dumps(x));
        with self.assertRaises(ExtractionRunnerError): run(pf,als,fs,t,f,a,o,s,0,1,self.reader,self.analyzer)
        td.cleanup()
    def test_token_hash_chain_required(self):
        td,a,f,t,fs,als,pf,o,s=self.make(); t.write_text(t.read_text()+'\n');
        with self.assertRaises(ExtractionRunnerError): run(pf,als,fs,t,f,a,o,s,0,1,self.reader,self.analyzer)
        td.cleanup()
    def test_metadata_mismatch_fails(self):
        td,a,f,t,fs,als,pf,o,s=self.make();
        with t.open(newline='',encoding='utf-8') as fh: rows=list(csv.DictReader(fh)); rows[0]['speaker_id']='bad'; wc(t,list(TOKEN_REQ),rows); als.write_text(json.dumps({'audit':'msp_textgrid_content_qa_v1','private_token_manifest_sha256':sh(t)}));
        with self.assertRaises(ExtractionRunnerError): run(pf,als,fs,t,f,a,o,s,0,1,self.reader,self.analyzer)
        td.cleanup()
    def test_noncontiguous_word_index_fails(self):
        td,a,f,t,fs,als,pf,o,s=self.make();
        with t.open(newline='',encoding='utf-8') as fh: rows=list(csv.DictReader(fh)); rows[-1]['word_index']='4'; wc(t,list(TOKEN_REQ),rows); als.write_text(json.dumps({'audit':'msp_textgrid_content_qa_v1','private_token_manifest_sha256':sh(t)}));
        with self.assertRaises(ExtractionRunnerError): run(pf,als,fs,t,f,a,o,s,0,1,self.reader,self.analyzer)
        td.cleanup()
    def test_missing_f0_is_blank_not_imputed(self):
        td,a,f,t,fs,als,pf,o,s=self.make(); ana=lambda x,sr:{'f0_mean_hz':None,'f0_robust_range_hz':None,'energy_db':-30.}; run(pf,als,fs,t,f,a,o,s,0,1,self.reader,ana);
        with o.open(newline='',encoding='utf-8') as fh: rows=list(csv.DictReader(fh)); self.assertEqual(rows[0]['f0_mean_hz'],''); self.assertEqual(rows[0]['f0_robust_range_hz'],''); self.assertEqual(rows[0]['energy_db'],'-30'); td.cleanup()
    def test_shard_assignment_deterministic(self):
        self.assertEqual(shard_for('a.wav',17),shard_for('a.wav',17)); self.assertTrue(0<=shard_for('a.wav',17)<17)
    def test_frozen_analyzer_loader(self):
        fn=load_frozen_analyzer(); self.assertIs(fn,frozen_analyzer.analyze_word)
    def test_bad_shard_fails(self):
        td,a,f,t,fs,als,pf,o,s=self.make();
        with self.assertRaises(ExtractionRunnerError): run(pf,als,fs,t,f,a,o,s,2,1,self.reader,self.analyzer)
        td.cleanup()
if __name__=='__main__': unittest.main()
