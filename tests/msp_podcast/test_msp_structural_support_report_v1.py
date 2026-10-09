import unittest,tempfile,csv,json
from pathlib import Path
from tools.msp_structural_support_report_v1 import *
from tools.msp_structural_v1 import speaker_order_key,ALT_SALTS,split_recording_units

class Tests(unittest.TestCase):
    def make(self,nsp=105,nutt=100):
        td=tempfile.TemporaryDirectory(); root=Path(td.name); st=root/'s.csv'; el=root/'e.txt'; fp=root/'f.csv'
        rows=[]
        for si in range(nsp):
            s=f's{si:03d}'
            for i in range(nutt): rows.append({'filename':f'MSP-PODCAST_{si*2+1:04d}_{i:04d}.wav','speaker_id':s,'split_set':'Train','recording_unit_id':f'MSP-PODCAST_{si*2+1:04d}' if i<50 else f'MSP-PODCAST_{si*2+2:04d}'})
        with st.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['filename','speaker_id','split_set','recording_unit_id']); w.writeheader(); w.writerows(rows)
        el.write_text('\n'.join(r['filename'] for r in rows)+'\n')
        selected=sorted([f's{i:03d}' for i in range(nsp)],key=speaker_order_key)[:100]; fields=['filename','speaker_id','split_set','recording_unit_id','primary_half']+[f'alt_{i:02d}_half' for i in range(25)]; out=[]
        for r in rows:
            if r['speaker_id'] not in selected: continue
            units={x['recording_unit_id'] for x in rows if x['speaker_id']==r['speaker_id']}; a,b=split_recording_units(r['speaker_id'],units); x=dict(r); x['primary_half']='A' if r['recording_unit_id'] in a else 'B'
            for j,salt in enumerate(ALT_SALTS): aa,bb=split_recording_units(r['speaker_id'],units,salt); x[f'alt_{j:02d}_half']='A' if r['recording_unit_id'] in aa else 'B'
            out.append(x)
        with fp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
        return td,st,el,fp
    def test_cap_and_no_reselection(self):
        td,st,el,fp=self.make(); z=build_report(st,el,fp); self.assertEqual(z['qualified_speakers'],105); self.assertEqual(z['selected_speakers'],100); self.assertFalse(z['reselection_performed']); td.cleanup()
    def test_two_units_have_one_unordered_partition(self):
        td,st,el,fp=self.make(); z=build_report(st,el,fp); self.assertEqual(z['selected_exactly_two_units'],100); self.assertEqual(z['selected_distinct_unordered_alt_partitions']['min'],1); self.assertEqual(z['selected_distinct_unordered_alt_partitions']['max'],1); td.cleanup()
    def test_safe_has_no_ids_or_science(self):
        td,st,el,fp=self.make(); z=build_report(st,el,fp); self.assertFalse(z['speaker_ids_emitted']); self.assertFalse(z['acoustic_values_accessed']); self.assertFalse(z['reliability_values_accessed']); self.assertNotIn('selected_speaker_ids',z); td.cleanup()
    def test_selected_set_mismatch_fails(self):
        td,st,el,fp=self.make()
        with fp.open(newline='') as f: rows=list(csv.DictReader(f))
        rows=[r for r in rows if r['speaker_id']!=rows[0]['speaker_id']]
        with fp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        with self.assertRaises(SupportReportError): build_report(st,el,fp); td.cleanup()
    def test_99_utterance_speaker_not_qualified(self):
        td,st,el,fp=self.make(nsp=1,nutt=99)
        with self.assertRaises(SupportReportError): build_report(st,el,fp)
        td.cleanup()

if __name__=='__main__': unittest.main()
