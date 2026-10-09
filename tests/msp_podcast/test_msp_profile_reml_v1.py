import unittest
import numpy as np
from tools.msp_profile_reml_v1 import ProfileREMLError, fit_profile_reml_random_intercept

class Tests(unittest.TestCase):
    def sim(self,tau,seed=13,g=800,k=10,duration=False):
        rng=np.random.Generator(np.random.PCG64(seed)); n=g*k
        word=np.repeat(np.arange(g),k); pos=rng.random(n); u=rng.normal(0,tau,g)
        if duration:
            syll=1+(word%4); X=np.column_stack([np.ones(n),syll,pos]); beta=np.array([-.7,.18,.12])
        else:
            X=np.column_stack([np.ones(n),pos]); beta=np.array([.4,.25])
        y=X@beta+u[word]+rng.normal(0,.6,n)
        return y,X,word,beta
    def test_zero_boundary(self):
        # Deterministic construction with identical within-group patterns and exactly zero between-group mean variance.
        groups=np.repeat(np.arange(200),4); X=np.ones((len(groups),1)); y=np.tile(np.array([-1.0,-0.5,0.5,1.0]),200)
        r=fit_profile_reml_random_intercept(y,X,groups)
        self.assertEqual(r.tau2,0.0); self.assertTrue(np.isfinite(r.residuals).all())
    def test_positive_variance_recovery(self):
        y,X,g,b=self.sim(.35,seed=48); r=fit_profile_reml_random_intercept(y,X,g)
        self.assertAlmostEqual(np.sqrt(r.tau2),.35,delta=.04); self.assertAlmostEqual(r.beta[1],b[1],delta=.05)
    def test_duration_identifiable_with_random_word_effect(self):
        y,X,g,b=self.sim(.25,seed=22,duration=True); r=fit_profile_reml_random_intercept(y,X,g)
        self.assertAlmostEqual(np.sqrt(r.tau2),.25,delta=.05); self.assertTrue(np.isfinite(r.residuals).all())
    def test_permutation_invariant(self):
        y,X,g,b=self.sim(.3,seed=20); r1=fit_profile_reml_random_intercept(y,X,g)
        p=np.random.Generator(np.random.PCG64(2)).permutation(len(y)); r2=fit_profile_reml_random_intercept(y[p],X[p],g[p])
        self.assertAlmostEqual(r1.tau2,r2.tau2,places=7); self.assertTrue(np.allclose(r1.beta,r2.beta,atol=1e-7))
    def test_deterministic(self):
        y,X,g,b=self.sim(.2,seed=8); a=fit_profile_reml_random_intercept(y,X,g); c=fit_profile_reml_random_intercept(y,X,g)
        self.assertEqual(a.theta,c.theta); self.assertTrue(np.array_equal(a.residuals,c.residuals))
    def test_rank_deficiency_fails(self):
        y,X,g,b=self.sim(.2); X=np.column_stack([X,X[:,1]])
        with self.assertRaises(ProfileREMLError): fit_profile_reml_random_intercept(y,X,g)
    def test_nonfinite_fails(self):
        y,X,g,b=self.sim(.2); y[0]=np.nan
        with self.assertRaises(ProfileREMLError): fit_profile_reml_random_intercept(y,X,g)
    def test_all_singletons_fail(self):
        y=np.arange(20,dtype=float); X=np.column_stack([np.ones(20),np.linspace(0,1,20)]); g=np.arange(20)
        with self.assertRaises(ProfileREMLError): fit_profile_reml_random_intercept(y,X,g)

if __name__=='__main__': unittest.main()
