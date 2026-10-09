import unittest, numpy as np
from tools.msp_acoustic_adapter_v1 import *

class Tests(unittest.TestCase):
    def test_relative_position(self):
        self.assertEqual(relative_position(0,1),0.0); self.assertEqual(relative_position(0,3),0.0); self.assertEqual(relative_position(1,3),.5); self.assertEqual(relative_position(2,3),1.0)
    def test_position_fails_closed(self):
        for args in [(-1,3),(3,3),(0,0)]:
            with self.assertRaises(AdapterPrototypeError): relative_position(*args)
    def test_16k_exact_slice(self):
        x=np.arange(16000,dtype=np.float32); y,sr=slice_resample_mono(x,16000,.25,.5); self.assertEqual(sr,16000); self.assertEqual(len(y),4000); self.assertTrue(np.array_equal(y,x[4000:8000]))
    def test_8k_resample_length(self):
        x=np.sin(np.linspace(0,20,8000)).astype(np.float32); y,sr=slice_resample_mono(x,8000,.1,.3); self.assertEqual(sr,16000); self.assertEqual(len(y),3200)
    def test_48k_resample_length(self):
        x=np.sin(np.linspace(0,20,48000)).astype(np.float32); y,sr=slice_resample_mono(x,48000,.1,.3); self.assertEqual(sr,16000); self.assertEqual(len(y),3200)
    def test_stereo_mean(self):
        a=np.arange(1000,dtype=np.float32); x=np.column_stack([a,a+2]); y,sr=slice_resample_mono(x,1000,.1,.2,1000); self.assertTrue(np.allclose(y,a[100:200]+1))
    def test_bad_bounds_fail(self):
        x=np.zeros(1600,dtype=np.float32)
        for a,b in [(-.1,.1),(.2,.1),(0,2)]:
            with self.assertRaises(AdapterPrototypeError): slice_resample_mono(x,16000,a,b)
    def test_injected_analyzer(self):
        x=np.zeros(1600,dtype=np.float32)
        def fake(seg,sr): return {'f0_mean_hz':1.,'f0_robust_range_hz':2.,'energy_db':3.}
        self.assertEqual(analyze_interval(x,16000,0,.05,fake)['energy_db'],3.)
    def test_output_schema_fails(self):
        x=np.zeros(1600,dtype=np.float32)
        with self.assertRaises(AdapterPrototypeError): analyze_interval(x,16000,0,.05,lambda a,s:{'energy_db':1})

if __name__=='__main__': unittest.main()
