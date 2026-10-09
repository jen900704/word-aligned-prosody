#!/usr/bin/env python3
"""SAFE, no-audio verification of the direct cluster baseline and local aligner."""
import argparse,ast,shutil,sys
import os
from pathlib import Path
from esd_contract import write_json
from esd_stage2_adapter import CMUDICT_SHA256,verify_cmudict
from frozen_word_aligner import (ALIGN_MODEL,ALIGN_CHECKPOINT_BASENAME,
    ALIGN_CHECKPOINT_SHA256,HISTORICAL_ALIGNMENT_SHA256,verify_checkpoint)

BASELINE_PYTHON=Path(os.environ.get("WAP_BASELINE_PYTHON", sys.executable))
LOCAL_ALIGNER=Path(__file__).resolve().with_name("frozen_word_aligner.py")
LOCAL_ALIGNER_SHA256="a0d642efe72c89d624cbb4f13c1520e590febdcd69df9621c980d73f043ceab9"
VERSION_LOCK={"python":"3.10.20","torch":"2.5.1+cu121","torchaudio":"2.5.1+cu121",
              "numpy":"1.26.4","soundfile":"0.13.1","scipy":"1.15.3","pandas":"2.3.3"}
FORBIDDEN_IMPORT_ROOTS={"whisperx","pyannote","faster_whisper","ctranslate2"}

def file_sha(path):
    import hashlib
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def imported_roots(path):
    roots=set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Import): roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: roots.add(node.module.split(".")[0])
    return roots

def verify(cmudict:Path,model_dir:Path,output:Path,expected_python:Path=BASELINE_PYTHON):
    versions={"python":".".join(map(str,sys.version_info[:3]))}; checks={}
    checks["expected_interpreter"]=Path(sys.executable).resolve()==expected_python.resolve()
    checks["python_version"]=versions["python"]==VERSION_LOCK["python"]
    try:
        import numpy,soundfile,scipy,pandas,torch,torchaudio
        versions.update(torch=torch.__version__,torchaudio=torchaudio.__version__,numpy=numpy.__version__,soundfile=soundfile.__version__,scipy=scipy.__version__,pandas=pandas.__version__)
        for name in ("torch","torchaudio","numpy","soundfile","scipy","pandas"): checks[f"version_{name}"]=versions[name]==VERSION_LOCK[name]
        checks["cuda_available"]=bool(torch.cuda.is_available()); checks["gpu_count_one_or_more"]=torch.cuda.device_count()>=1
    except Exception as exc:
        checks["required_imports_complete"]=False; versions["import_error_type"]=type(exc).__name__
    else: checks["required_imports_complete"]=True
    try: verify_cmudict(cmudict); checks["frozen_cmudict_verified"]=True
    except Exception: checks["frozen_cmudict_verified"]=False
    try: verify_checkpoint(model_dir); checks["frozen_alignment_checkpoint_verified"]=True
    except Exception: checks["frozen_alignment_checkpoint_verified"]=False
    checks["local_aligner_source_hash"]=file_sha(LOCAL_ALIGNER)==LOCAL_ALIGNER_SHA256
    imports=imported_roots(LOCAL_ALIGNER)|imported_roots(Path(__file__).resolve().with_name("esd_stage2_adapter.py"))
    checks["forbidden_runtime_imports_absent"]=not bool(imports&FORBIDDEN_IMPORT_ROOTS)
    source=LOCAL_ALIGNER.read_text().casefold()
    checks["network_or_download_path_absent"]="http://" not in source and "https://" not in source and "download_url" not in source
    checks["ffmpeg_present"]=shutil.which("ffmpeg") is not None
    artifact={"runtime_version":"esd_stage2_direct_baseline_v1","expected_interpreter":str(expected_python),
              "historical_alignment_sha256":HISTORICAL_ALIGNMENT_SHA256,"local_aligner_sha256":file_sha(LOCAL_ALIGNER),
              "alignment_model":ALIGN_MODEL,"alignment_checkpoint":{"basename":ALIGN_CHECKPOINT_BASENAME,"sha256":ALIGN_CHECKPOINT_SHA256},
              "cmudict_sha256":CMUDICT_SHA256,"versions":versions,"checks":checks,"all_checks_passed":all(checks.values()),
              "forbidden_import_roots":sorted(FORBIDDEN_IMPORT_ROOTS),"global_pip_check_used":False,
              "audio_read":False,"model_loaded":False,"model_download_attempted":False,"network_access_attempted":False}
    write_json(output,artifact); return artifact

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cmudict",type=Path,required=True); parser.add_argument("--model-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--expected-python",type=Path,default=BASELINE_PYTHON); args=parser.parse_args()
    return 0 if verify(args.cmudict,args.model_dir,args.output,args.expected_python)["all_checks_passed"] else 2

if __name__=="__main__": raise SystemExit(main())
