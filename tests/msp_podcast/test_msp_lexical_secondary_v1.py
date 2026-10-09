import unittest, numpy as np
from tools.msp_lexical_secondary_v1 import *

class Tests(unittest.TestCase):
    def make(self,seed=13,nword=80,k=6,duration=False):
        rng=np.random.Generator(np.random.PCG64(seed)); words=np.repeat(np.array([f'w{i}' for i in range(nword)],object),k); n=len(words); pos=np.tile(np.linspace(0,1,k),nword); u=np.repeat(rng.normal(0,.3,nword),k)
        if duration:
            syll=1+(np.arange(n)%3); y=np.exp(-1+.15*syll+.1*pos+u+rng.normal(0,.2,n)); return y,pos,words,syll
        y=2+.2*pos+u+rng.normal(0,.2,n); return y,pos,words,None
    def test_crossfit_deterministic(self):
        av,ap,aw,_=self.make(1); bv,bp,bw,_=self.make(2)
        x=crossfit_two_halves('f0_mean_hz',av,ap,aw,bv,bp,bw); y=crossfit_two_halves('f0_mean_hz',av,ap,aw,bv,bp,bw)
        self.assertTrue(np.array_equal(x['A'].residuals,y['A'].residuals)); self.assertTrue(np.array_equal(x['B'].residuals,y['B'].residuals))
    def test_heldout_a_does_not_affect_a_nuisance_fit(self):
        av,ap,aw,_=self.make(3); bv,bp,bw,_=self.make(4)
        x=crossfit_two_halves('f0_mean_hz',av,ap,aw,bv,bp,bw); av2=av+np.linspace(-5,5,len(av)); y=crossfit_two_halves('f0_mean_hz',av2,ap,aw,bv,bp,bw)
        self.assertTrue(np.array_equal(x['A'].beta,y['A'].beta)); self.assertEqual(x['A'].tau2,y['A'].tau2)
    def test_heldout_b_does_not_affect_b_nuisance_fit(self):
        av,ap,aw,_=self.make(5); bv,bp,bw,_=self.make(6)
        x=crossfit_two_halves('f0_mean_hz',av,ap,aw,bv,bp,bw); bv2=bv*1.7; y=crossfit_two_halves('f0_mean_hz',av,ap,aw,bv2,bp,bw)
        self.assertTrue(np.array_equal(x['B'].beta,y['B'].beta)); self.assertEqual(x['B'].tau2,y['B'].tau2)
    def test_unseen_word_random_effect_zero(self):
        tv,tp,tw,_=self.make(7,nword=50); hv,hp,hw,_=self.make(8,nword=10); hw=np.array([f'UNSEEN_{i}' for i in range(len(hw))],object)
        z=fit_train_apply_heldout('energy_db',tv,tp,tw,hv,hp,hw,train_half='A',heldout_half='B'); self.assertEqual(z.unseen_word_count,len(hw)); self.assertTrue(np.isfinite(z.residuals).all())
    def test_duration_crossfit(self):
        av,ap,aw,asy=self.make(9,duration=True); bv,bp,bw,bsy=self.make(10,duration=True)
        z=crossfit_two_halves(DURATION_FEATURE,av,ap,aw,bv,bp,bw,asy,bsy); self.assertTrue(np.isfinite(z['A'].residuals).all()); self.assertTrue(np.isfinite(z['B'].residuals).all())
    def test_same_half_fails(self):
        v,p,w,_=self.make(11)
        with self.assertRaises(LexicalSecondaryError): fit_train_apply_heldout('f0_mean_hz',v,p,w,v,p,w,train_half='A',heldout_half='A')

if __name__=='__main__': unittest.main()
