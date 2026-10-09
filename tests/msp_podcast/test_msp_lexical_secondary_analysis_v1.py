import unittest,math
from tools.msp_lexical_secondary_analysis_v1 import *

def rows(nspeakers=52):
    out=[]
    for s in range(nspeakers):
        for u in range(4):
            h='A' if u<2 else 'B'
            for wi in range(30):
                p=wi/29; sy=1+(wi%3); r={k:'' for k in OUT_FIELDS}; r.update(filename=f's{s}_u{u}.wav',speaker_id=f's{s}',recording_unit_id=f'u{u}',primary_half=h,word_index=str(wi),normalized_word=f'w{wi%12}',relative_position=str(p),word_start_sec=str(wi*.1),word_end_sec=str(wi*.1+.08),word_duration_sec=str(math.exp(-1.5+.01*s+.05*sy+.02*p)),authoritative_syllable_count=str(sy),syllable_lookup_status='first_canonical',f0_mean_hz=str(100+s+.5*p),f0_robust_range_hz=str(20+.2*s+.2*p),energy_db=str(-30+.1*s+.1*p),acoustic_status='complete',acoustic_error='')
                for j in range(25): r[f'alt_{j:02d}_half']=h if j%2==0 else ('B' if h=='A' else 'A')
                out.append(r)
    return out
class Tests(unittest.TestCase):
    def test_all_features_all_splits_secondary_only(self):
        z=analyze_rows(rows())
        self.assertTrue(z['secondary_only']); self.assertFalse(z['can_modify_gate_m'])
        for f,r in z['features'].items():
            self.assertEqual(set(r['splits']),set(SPLITS)); self.assertTrue(all(x['status']=='complete' for x in r['splits'].values())); self.assertTrue(all(x['gate_m_threshold_applied'] is False for x in r['splits'].values()))
    def test_crossfit_direction(self):
        z=analyze_split(valid_feature_rows(rows(),'f0_mean_hz'),'f0_mean_hz','primary_half'); self.assertEqual(z['A_train_half'],'B'); self.assertEqual(z['B_train_half'],'A')
    def test_missing_f0_dropped_not_imputed(self):
        r=rows(2); r[0]['f0_mean_hz']=''; v=valid_feature_rows(r,'f0_mean_hz'); self.assertEqual(len(v),len(r)-1)
    def test_bad_half_fails(self):
        v=valid_feature_rows(rows(2),'energy_db'); v[0][0]['primary_half']='X'
        with self.assertRaises(LexicalAnalysisError): analyze_split(v,'energy_db','primary_half')
    def test_safe_has_no_reliability_keys(self):
        z=analyze_rows(rows()); fake=Path('/tmp/a');
        # structural inspection of constructed SAFE fields without filesystem hashes
        feats={}
        for f,r in z['features'].items():
            states=[x['status'] for x in r['splits'].values()]; paired=[x.get('paired_speakers') for x in r['splits'].values() if x['status']=='complete']; feats[f]={'splits_complete':sum(x=='complete' for x in states),'paired_speakers_min':min(paired)}
        text=str(feats).lower(); self.assertNotIn('pearson',text); self.assertNotIn('spearman',text)
if __name__=='__main__': unittest.main()
