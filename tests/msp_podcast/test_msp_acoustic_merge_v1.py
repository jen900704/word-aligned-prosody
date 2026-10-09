import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.msp_acoustic_extraction_runner_v2 import TOKEN_REQ,run,sha256
from tools.msp_acoustic_merge_v1 import *

def wc(p,fields,rows):
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
class Tests(unittest.TestCase):
    def make(self):
        td=tempfile.TemporaryDirectory(); d=Path(td.name); audio=d/'audio'; audio.mkdir(); shard=d/'shards'; shard.mkdir(); final=d/'final.csv'; token=d/'token.csv'; fs=d/'fs.json'; als=d/'al.json'; pf=d/'pf.json'; out=d/'merged.csv'; safe=d/'merge_SAFE.json'
        ff=['filename','speaker_id','recording_unit_id','primary_half']+[f'alt_{i:02d}_half' for i in range(25)]; fr=[]; tr=[]
        for j,fn in enumerate(['a.wav','b.wav','c.wav']):
            (audio/fn).write_bytes(b'x'); r={'filename':fn,'speaker_id':f's{j}','recording_unit_id':f'u{j}','primary_half':'A'}; r.update({f'alt_{i:02d}_half':'A' for i in range(25)}); fr.append(r)
            for wi in range(2): tr.append({'filename':fn,'speaker_id':f's{j}','recording_unit_id':f'u{j}','word_index':str(wi),'normalized_word':f'w{wi}','word_start_sec':str(wi*.1),'word_end_sec':str(wi*.1+.08),'word_duration_sec':'.08','authoritative_syllable_count':'1','syllable_lookup_status':'first_canonical'})
        wc(final,ff,fr); wc(token,list(TOKEN_REQ),tr); fs.write_text(json.dumps({'audit':'msp_final_scope_and_split_manifest_v1','private_manifest_sha256':sha256(final)})); als.write_text(json.dumps({'audit':'msp_textgrid_content_qa_v1','private_token_manifest_sha256':sha256(token)})); pf.write_text(json.dumps({'audit':'msp_acoustic_preflight_guard_v4','msp_audio_processing_authorized':True,'checks':{'x':True},'scientific_choices_made_by_guard':False,'reliability_values_accessed':False}))
        reader=lambda p:(np.ones((16000,1),np.float32),16000); analyzer=lambda x,sr:{'f0_mean_hz':100.,'f0_robust_range_hz':10.,'energy_db':-20.}
        for i in range(2): run(pf,als,fs,token,final,audio,shard/f'shard_{i:02d}.csv',shard/f'shard_{i:02d}_SAFE.json',i,2,reader,analyzer)
        return td,shard,pf,als,fs,token,final,out,safe
    def test_exact_merge(self):
        td,sd,pf,al,fs,t,f,o,z=self.make(); q=merge(sd,2,pf,al,fs,t,f,o,z); self.assertEqual(q['expected_token_rows'],6); self.assertEqual(q['merged_token_rows'],6); self.assertEqual(q['acoustic_failed_tokens'],0); self.assertFalse(q['reliability_values_accessed']); td.cleanup()
    def test_missing_shard_fails(self):
        td,sd,pf,al,fs,t,f,o,z=self.make(); (sd/'shard_01_SAFE.json').unlink();
        with self.assertRaises(MergeError): merge(sd,2,pf,al,fs,t,f,o,z)
        td.cleanup()
    def test_tampered_output_fails(self):
        td,sd,pf,al,fs,t,f,o,z=self.make(); p=sd/'shard_00.csv'; p.write_text(p.read_text()+'\n');
        with self.assertRaises(MergeError): merge(sd,2,pf,al,fs,t,f,o,z)
        td.cleanup()
    def test_duplicate_token_fails(self):
        td,sd,pf,al,fs,t,f,o,z=self.make(); p=sd/'shard_00.csv'
        with p.open(newline='',encoding='utf-8') as fh: rows=list(csv.DictReader(fh))
        if rows:
            with p.open('a',newline='') as h: csv.DictWriter(h,fieldnames=OUT_FIELDS).writerow(rows[0])
            q=json.loads((sd/'shard_00_SAFE.json').read_text()); q['output_sha256']=sha256(p); q['token_rows']+=1; (sd/'shard_00_SAFE.json').write_text(json.dumps(q))
            with self.assertRaises(MergeError): merge(sd,2,pf,al,fs,t,f,o,z)
        td.cleanup()
if __name__=='__main__': unittest.main()
