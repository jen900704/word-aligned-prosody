import inspect,sys,unittest
from pathlib import Path
import numpy as np
from tools import msp_esd_analyze_word_v1 as standalone
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"esd"))
import esd_stage2_adapter as original

class Tests(unittest.TestCase):
    def test_source_frames_exact(self):
        self.assertEqual(inspect.getsource(standalone.frames),inspect.getsource(original.frames))
    def test_source_analyze_word_exact(self):
        self.assertEqual(inspect.getsource(standalone.analyze_word),inspect.getsource(original.analyze_word))
    def assertSame(self,x):
        a=standalone.analyze_word(x.copy(),16000); b=original.analyze_word(x.copy(),16000)
        self.assertEqual(a.keys(),b.keys())
        for k in a:
            if a[k] is None or b[k] is None: self.assertIs(a[k],b[k])
            else: self.assertEqual(a[k],b[k])
    def test_silence_exact(self): self.assertSame(np.zeros(1600,dtype=np.float32))
    def test_sine_exact(self):
        t=np.arange(8000)/16000.; self.assertSame((.2*np.sin(2*np.pi*180*t)).astype(np.float32))
    def test_deterministic_noise_exact(self):
        x=np.random.Generator(np.random.PCG64(13)).normal(0,.03,5000).astype(np.float32); self.assertSame(x)
    def test_short_signal_exact(self):
        t=np.arange(200)/16000.; self.assertSame((.1*np.sin(2*np.pi*220*t)).astype(np.float32))
if __name__=='__main__': unittest.main()
