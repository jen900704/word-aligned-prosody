import unittest, numpy as np
from tools.msp_primary_residualization_v2 import *

class Tests(unittest.TestCase):
    def test_relative_position_exact(self):
        self.assertEqual(relative_position(0,1),0.0); self.assertEqual(relative_position(1,3),0.5); self.assertEqual(relative_position(2,3),1.0)
    def test_relative_position_bad_bounds(self):
        for a,b in [(-1,2),(2,2),(0,0)]:
            with self.assertRaises(PrimaryResidualizationError): relative_position(a,b)
    def test_f0_recovers_fixed_effects(self):
        p=np.linspace(0,1,500); y=10+2*p+np.sin(np.arange(500))*0.1
        r=fit_primary_residuals('f0_mean_hz',y,p)
        self.assertAlmostEqual(r.beta[0],10,delta=.02); self.assertAlmostEqual(r.beta[1],2,delta=.03); self.assertAlmostEqual(float(r.residuals.mean()),0,places=12)
    def test_energy_forbids_syllable_adjustment(self):
        p=np.linspace(0,1,20)
        with self.assertRaises(PrimaryResidualizationError): design_matrix('energy_db',p,np.ones(20))
    def test_duration_logs_and_adjusts_syllables(self):
        n=600; syll=1+(np.arange(n)%4); p=np.linspace(0,1,n); logd=-1+.2*syll+.1*p+0.03*np.sin(np.arange(n)); d=np.exp(logd)
        r=fit_primary_residuals(DURATION_FEATURE,d,p,syll)
        self.assertAlmostEqual(r.beta[1],.2,delta=.01); self.assertAlmostEqual(r.beta[2],.1,delta=.01); self.assertTrue(np.allclose(r.transformed_response,np.log(d)))
    def test_duration_requires_positive_integer_syllables(self):
        p=np.linspace(0,1,10); d=np.ones(10)
        for s in [None,[1]*9+[0],[1]*9+[1.5]]:
            with self.assertRaises(PrimaryResidualizationError): fit_primary_residuals(DURATION_FEATURE,d,p,s)
    def test_rank_deficiency_fails(self):
        p=np.zeros(20); y=np.arange(20,dtype=float)
        with self.assertRaises(PrimaryResidualizationError): fit_primary_residuals('f0_mean_hz',y,p)
    def test_nonfinite_fails(self):
        p=np.linspace(0,1,20); y=np.arange(20,dtype=float); y[2]=np.nan
        with self.assertRaises(PrimaryResidualizationError): fit_primary_residuals('f0_mean_hz',y,p)

if __name__=='__main__': unittest.main()
