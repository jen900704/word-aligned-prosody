import hashlib,json,tempfile,unittest
from pathlib import Path
from tools.msp_implementation_integrity_v1 import *

def sh(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Tests(unittest.TestCase):
    def make(self):
        td=tempfile.TemporaryDirectory(); d=Path(td.name); (d/'tools').mkdir(); (d/'tests').mkdir(); expected={}
        for name in REQUIRED_MODEL_FILES:
            p=d/('tests' if name.startswith('test_') else 'tools')/name; p.write_text('frozen:'+name); expected[name]=sh(p)
        esd=d/'esd.py'; sm=d/'seamless.py'; esd.write_text('esd'); sm.write_text('seamless'); expected['esd_frozen_acoustic_adapter_sha256']=sh(esd); expected['seamless_frozen_pipeline_sha256']=sh(sm)
        lock=d/'lock.json'; lock.write_text(json.dumps({'status':'frozen','human_approved':True,'implementation_status':'frozen','implementation':{'files_sha256':expected,'synthetic_test_total':43,'synthetic_test_passed':43}}))
        return td,d,lock,esd,sm
    def test_exact_hashes_pass(self):
        td,d,l,e,s=self.make(); z=verify(l,d,e,s); self.assertTrue(z['all_hashes_match']); self.assertEqual(z['verified_hash_count'],12); td.cleanup()
    def test_modified_tool_fails(self):
        td,d,l,e,s=self.make(); (d/'tools/msp_reliability_v2.py').write_text('changed')
        with self.assertRaises(IntegrityError): verify(l,d,e,s)
        td.cleanup()
    def test_modified_reference_fails(self):
        td,d,l,e,s=self.make(); e.write_text('changed')
        with self.assertRaises(IntegrityError): verify(l,d,e,s)
        td.cleanup()
    def test_missing_file_fails(self):
        td,d,l,e,s=self.make(); (d/'tests/test_msp_reliability_v2.py').unlink()
        with self.assertRaises(IntegrityError): verify(l,d,e,s)
        td.cleanup()
    def test_not_frozen_fails(self):
        td,d,l,e,s=self.make(); o=json.loads(l.read_text()); o['implementation_status']='draft'; l.write_text(json.dumps(o))
        with self.assertRaises(IntegrityError): verify(l,d,e,s)
        td.cleanup()
    def test_test_count_mismatch_fails(self):
        td,d,l,e,s=self.make(); o=json.loads(l.read_text()); o['implementation']['synthetic_test_passed']=42; l.write_text(json.dumps(o))
        with self.assertRaises(IntegrityError): verify(l,d,e,s)
        td.cleanup()

if __name__=='__main__': unittest.main()
