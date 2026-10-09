import unittest, numpy as np
from tools.msp_reliability_v2 import *
import tools.msp_reliability_v2 as m

def rows(nsp=60,tokens=60,noisy=False):
    out=[]
    for s in range(nsp):
        base=s/10
        for h in ('A','B'):
            for i in range(tokens):
                val=base+.02*i/tokens+(.03 if h=='B' else 0)+(.2*np.sin(i+s) if noisy else 0)
                r={'speaker_id':f's{s:03d}','recording_unit_id':f'u{s}_{h}_{i%3}','primary_half':h,'residual':val}
                for j in range(25): r[f'alt_{j:02d}_half']=h
                out.append(r)
    return out

class Tests(unittest.TestCase):
    def test_token_support_threshold(self):
        self.assertEqual(reliability(rows(tokens=49))['paired_speakers'],0); self.assertEqual(reliability(rows(tokens=50))['paired_speakers'],60)
    def test_paired_speaker_threshold(self):
        self.assertFalse(reliability(rows(nsp=49))['support_adequate']); self.assertTrue(reliability(rows(nsp=50))['support_adequate'])
    def test_perfect_stability(self):
        z=reliability(rows()); self.assertAlmostEqual(z['corrected_pearson'],1,places=12); self.assertAlmostEqual(z['spearman'],1,places=12)
    def test_negative_not_clipped(self):
        r,c=corrected_pearson([1,2,3,4],[4,3,2,1]); self.assertAlmostEqual(r,-1); self.assertIsNone(c)
    def test_bootstrap_deterministic_no_replacement(self):
        p=reliability(rows(noisy=True))['pairs']; a=bootstrap_from_pairs(p,1000,13); b=bootstrap_from_pairs(p,1000,13); self.assertEqual(a,b); self.assertFalse(a['replacement_draws_generated'])
    def test_bootstrap_ci_withheld_below_950(self):
        p=[('s1',1.,1.,50,50,1,1),('s2',1.,1.,50,50,1,1)]; z=bootstrap_from_pairs(p); self.assertFalse(z['technical_adequacy']); self.assertIsNone(z['ci_95'])
    def test_alternative_exact_25(self):
        z=alternative_split_metrics(rows()); self.assertEqual(z['valid_splits'],25); self.assertTrue(z['passes']);
        with self.assertRaises(MSPReliabilityError): alternative_split_metrics(rows(),24)
    def test_loso_and_volume(self):
        p=reliability(rows(noisy=True))['pairs']; self.assertEqual(len(loso(p)['changes']),60); self.assertIsNotNone(recording_volume_sensitivity(rows(noisy=True))['abs_change'])
    def test_constant_undefined(self):
        r,c=corrected_pearson([1,1,1],[2,3,4]); self.assertIsNone(r); self.assertIsNone(c)
    def test_invalid_half_fails(self):
        r=rows(); r[0]['primary_half']='X'
        with self.assertRaises(MSPReliabilityError): reliability(r)
    def test_gate_contract_keys(self):
        primary=reliability(rows()); boot=bootstrap_from_pairs(primary['pairs']); alt=alternative_split_metrics(rows()); lo=loso(primary['pairs']); vol=recording_volume_sensitivity(rows())
        g=gate_m_feature_conditions(primary,boot,alt,lo,vol); self.assertEqual(set(g['conditions']),{'paired_speakers_at_least_50','speaker_half_support_at_least_50','primary_corrected_at_least_0_60','bootstrap_lower_at_least_0_40','alternative_split_median_iqr','loso_max_change_at_most_0_15','recording_volume_change_at_most_0_15'})
    def test_average_rank_ties(self): self.assertTrue(np.allclose(m._avg_ranks([1,1,3,2]),[1.5,1.5,4,3]))

if __name__=='__main__': unittest.main()
