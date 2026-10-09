#!/usr/bin/env python3
"""Prepare/verify a deterministic local transport of final-scope MSP audio files."""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from pathlib import Path
class AudioTransportError(ValueError): pass

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def selected_filenames(final_private:Path,final_safe:Path):
    z=json.loads(final_safe.read_text(encoding='utf-8'))
    if z.get('audit')!='msp_final_scope_and_split_manifest_v1': raise AudioTransportError('final_safe_audit')
    if z.get('private_manifest_sha256')!=sha256(final_private): raise AudioTransportError('final_private_hash')
    out=[]
    with final_private.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if 'filename' not in (r.fieldnames or []): raise AudioTransportError('filename_missing')
        for row in r:
            fn=str(row['filename']).strip()
            if not fn or Path(fn).name!=fn or '/' in fn or '\\' in fn or not fn.lower().endswith('.wav'): raise AudioTransportError('unsafe_filename')
            out.append(fn)
    if not out or len(out)!=len(set(out)): raise AudioTransportError('empty_or_duplicate_filename')
    return sorted(out)

def write_manifest(final_private:Path,final_safe:Path,private_list:Path,safe_json:Path):
    names=selected_filenames(final_private,final_safe)
    private_list.parent.mkdir(parents=True,exist_ok=True); private_list.write_text('\n'.join(names)+'\n',encoding='utf-8'); os.chmod(private_list,0o600)
    safe={'audit':'msp_selected_audio_transport_manifest_v1','status':'PASS','selected_audio_files':len(names),'private_filename_list_sha256':sha256(private_list),'final_private_manifest_sha256':sha256(final_private),'final_scope_safe_sha256':sha256(final_safe),'audio_values_accessed':False,'reliability_values_accessed':False,'speaker_ids_emitted':False,'scientific_choices_made_by_tool':False}
    safe_json.parent.mkdir(parents=True,exist_ok=True); safe_json.write_text(json.dumps(safe,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.chmod(safe_json,0o600); return safe

def verify_destination(private_list:Path,manifest_safe:Path,dest_root:Path,safe_json:Path):
    m=json.loads(manifest_safe.read_text(encoding='utf-8'))
    if m.get('audit')!='msp_selected_audio_transport_manifest_v1' or m.get('status')!='PASS': raise AudioTransportError('manifest_safe_invalid')
    if m.get('private_filename_list_sha256')!=sha256(private_list): raise AudioTransportError('filename_list_hash')
    expected=[x.strip() for x in private_list.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(expected)!=len(set(expected)) or not expected: raise AudioTransportError('expected_invalid')
    actual=sorted(p.name for p in dest_root.iterdir() if p.is_file() and p.suffix.lower()=='.wav') if dest_root.is_dir() else []
    if actual!=sorted(expected): raise AudioTransportError('destination_filename_set_mismatch')
    missing=[fn for fn in expected if not (dest_root/fn).is_file()]
    if missing: raise AudioTransportError('destination_missing_file')
    total_bytes=sum((dest_root/fn).stat().st_size for fn in expected)
    if total_bytes<=0: raise AudioTransportError('destination_empty_bytes')
    safe={'audit':'msp_selected_audio_transport_verify_v1','status':'PASS','selected_audio_files':len(expected),'destination_files':len(actual),'total_bytes':int(total_bytes),'filename_set_exact':True,'private_filename_list_sha256':sha256(private_list),'manifest_safe_sha256':sha256(manifest_safe),'audio_values_accessed':False,'reliability_values_accessed':False,'speaker_ids_emitted':False,'scientific_choices_made_by_tool':False}
    safe_json.parent.mkdir(parents=True,exist_ok=True); safe_json.write_text(json.dumps(safe,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.chmod(safe_json,0o600); return safe

def main(argv=None):
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('prepare'); a.add_argument('--final-private',type=Path,required=True); a.add_argument('--final-safe',type=Path,required=True); a.add_argument('--private-list',type=Path,required=True); a.add_argument('--safe-json',type=Path,required=True)
    b=sp.add_parser('verify'); b.add_argument('--private-list',type=Path,required=True); b.add_argument('--manifest-safe',type=Path,required=True); b.add_argument('--dest-root',type=Path,required=True); b.add_argument('--safe-json',type=Path,required=True)
    q=p.parse_args(argv)
    if q.cmd=='prepare': write_manifest(q.final_private,q.final_safe,q.private_list,q.safe_json)
    else: verify_destination(q.private_list,q.manifest_safe,q.dest_root,q.safe_json)
    return 0
if __name__=='__main__': raise SystemExit(main())
