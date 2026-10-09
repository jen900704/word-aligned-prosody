#!/usr/bin/env python3
"""Full-corpus MSP TextGrid content QA under the frozen v1 contract."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path

CMUDICT_SHA256="81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
OUTER_PUNCT=re.compile(r"(^[^\w']+|[^\w']+$)",re.UNICODE)
FLOAT=r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STRUCT_REQ={"filename","speaker_id","recording_unit_id"}
STRUCT_ALLOWED=STRUCT_REQ|{"split_set"}
TOKEN_FIELDS=["filename","speaker_id","split_set","recording_unit_id","word_index","normalized_word",
              "word_start_sec","word_end_sec","word_duration_sec","authoritative_syllable_count","syllable_lookup_status"]

class QAError(ValueError): pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def normalize_text(text):
    out=[]
    for source in text.split():
        value=source.replace("’","'").lower()
        value=OUTER_PUNCT.sub("",value)
        value=re.sub(r"[^\w'-]","",value)
        if value: out.append(value)
    return out

def load_cmudict(path, expected_sha=CMUDICT_SHA256):
    actual=sha(path)
    if expected_sha is not None and actual!=expected_sha: raise QAError("CMUdict checksum mismatch")
    result=defaultdict(list)
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        if not line or line.startswith(";;;"): continue
        word,phones=line.split(maxsplit=1)
        base=re.sub(r"\(\d+\)$","",word).lower()
        result[base].append(sum(bool(re.search(r"[012]$",p)) for p in phones.split()))
    return dict(result),actual

def parse_tier_block(block, need_marks=False):
    ints=list(re.finditer(r"(?m)^\s*intervals \[(\d+)\]:\s*$",block))
    out=[]
    for j,im in enumerate(ints):
        ib=block[im.end():ints[j+1].start() if j+1<len(ints) else len(block)]
        mm=re.search(rf"xmin\s*=\s*({FLOAT}).*?xmax\s*=\s*({FLOAT})",ib,re.S)
        if not mm: raise QAError("interval_missing_bounds")
        x0,x1=map(float,mm.groups())
        if not math.isfinite(x0) or not math.isfinite(x1): raise QAError("interval_nonfinite")
        mark=None
        if need_marks:
            tm=re.search(r'(?m)^\s*text\s*=\s*"(.*)"\s*$',ib)
            if not tm: raise QAError("word_interval_missing_text")
            mark=tm.group(1).replace('""','"')
        out.append((x0,x1,mark))
    return out

def parse_textgrid(text):
    if "ooTextFile" not in text or "TextGrid" not in text: raise QAError("not_long_textgrid")
    head=re.search(rf"xmin\s*=\s*({FLOAT}).*?xmax\s*=\s*({FLOAT})",text,re.S)
    if not head: raise QAError("missing_global_bounds")
    gx0,gx1=map(float,head.groups())
    if not all(map(math.isfinite,(gx0,gx1))) or gx0<0 or gx1<=gx0: raise QAError("invalid_global_bounds")
    items=list(re.finditer(r"(?m)^\s*item \[(\d+)\]:\s*$",text))
    if not items: raise QAError("no_tiers")
    required={"words":[],"phones":[]}
    for i,m in enumerate(items):
        block=text[m.end():items[i+1].start() if i+1<len(items) else len(text)]
        cm=re.search(r'class\s*=\s*"([^"]+)"',block); nm=re.search(r'name\s*=\s*"([^"]*)"',block)
        if not cm or not nm: raise QAError("tier_missing_class_or_name")
        cls,name=cm.group(1),nm.group(1)
        if name in required:
            if cls!="IntervalTier": raise QAError("required_tier_not_interval")
            required[name].append(parse_tier_block(block,need_marks=(name=="words")))
    if len(required["words"])!=1: raise QAError("words_tier_not_unique")
    if len(required["phones"])!=1: raise QAError("phones_tier_not_unique")
    words,phones=required["words"][0],required["phones"][0]
    for tier_name,ints in (("words",words),("phones",phones)):
        prev=None
        for x0,x1,_ in ints:
            if x0<gx0-1e-9 or x1>gx1+1e-9: raise QAError(tier_name+"_bounds_outside_global")
            if x1<=x0: raise QAError(tier_name+"_nonpositive_interval")
            if prev is not None and x0<prev-1e-9: raise QAError(tier_name+"_overlap_or_nonmonotonic")
            prev=x1
    return words

def load_structural(path):
    rows=[]; seen=set()
    with Path(path).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f); fields=set(r.fieldnames or [])
        if not STRUCT_REQ<=fields or fields-STRUCT_ALLOWED: raise QAError("structural_schema_mismatch")
        for x in r:
            fn=x["filename"]
            if fn in seen: raise QAError("duplicate_structural_filename")
            seen.add(fn); rows.append({k:x.get(k,"") for k in ("filename","speaker_id","split_set","recording_unit_id")})
    return rows

def load_member_map(path):
    members=[x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    out={}
    for m in members:
        b=Path(m).name
        if not b.lower().endswith(".textgrid"): raise QAError("inventory_non_textgrid")
        fn=b[:-9]+".wav"
        if fn in out: raise QAError("duplicate_inventory_filename")
        out[fn]=m
    return out

def run(zip_path, inventory_path, structural_csv, cmudict_path, token_csv, eligible_txt, safe_json, expected_cmudict_sha=CMUDICT_SHA256):
    structural=load_structural(structural_csv); members=load_member_map(inventory_path)
    sf={x["filename"] for x in structural}
    if sf!=set(members): raise QAError("structural_inventory_filename_set_mismatch")
    cmu,cmu_sha=load_cmudict(cmudict_path,expected_cmudict_sha)
    file_fail=Counter(); token_excl=Counter(); syllable_missing=0; eligible_files=[]; token_count=0
    Path(token_csv).parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path,"r") as z, Path(token_csv).open("w",newline="",encoding="utf-8") as tf:
        names=set(z.namelist()); w=csv.DictWriter(tf,fieldnames=TOKEN_FIELDS); w.writeheader()
        for meta in structural:
            fn=meta["filename"]; member=members[fn]
            if member not in names: raise QAError("inventory_member_absent_from_zip")
            try:
                text=z.read(member).decode("utf-8","strict")
                words=parse_textgrid(text)
                pending=[]; word_index=0
                for x0,x1,mark in words:
                    if mark is None or not mark.strip():
                        token_excl["empty_gap"]+=1; continue
                    norm=normalize_text(mark)
                    if len(norm)==0:
                        token_excl["normalizes_to_zero"]+=1; continue
                    if len(norm)!=1:
                        token_excl["normalizes_to_multiple"]+=1; continue
                    word=norm[0]; counts=cmu.get(word); syl=counts[0] if counts else ""
                    status="first_canonical" if counts else "missing"
                    syllable_missing+=int(not counts)
                    pending.append({**meta,"word_index":word_index,"normalized_word":word,
                                    "word_start_sec":format(x0,".9g"),"word_end_sec":format(x1,".9g"),
                                    "word_duration_sec":format(x1-x0,".9g"),
                                    "authoritative_syllable_count":syl,"syllable_lookup_status":status})
                    word_index+=1
                if not pending:
                    file_fail["no_eligible_normalized_word"]+=1; continue
                eligible_files.append(fn)
                w.writerows(pending); token_count+=len(pending)
            except UnicodeDecodeError:
                file_fail["utf8_decode_error"]+=1
            except QAError as e:
                file_fail[str(e)]+=1
    Path(eligible_txt).write_text("\n".join(eligible_files)+("\n" if eligible_files else ""),encoding="utf-8")
    out={
        "audit":"msp_textgrid_content_qa_v1",
        "structural_files":len(structural),
        "filename_coverage_exact":True,
        "files_passing_content_qa":len(eligible_files),
        "files_failing_content_qa":len(structural)-len(eligible_files),
        "file_failure_reason_counts":dict(sorted(file_fail.items())),
        "eligible_word_tokens":token_count,
        "token_exclusion_reason_counts":dict(sorted(token_excl.items())),
        "syllable_lookup_missing_tokens":syllable_missing,
        "cmudict_sha256":cmu_sha,
        "private_token_manifest_sha256":sha(token_csv),
        "eligible_filename_list_sha256":sha(eligible_txt),
        "textgrid_content_accessed":True,
        "transcript_text_accessed":False,
        "phone_marks_parsed_or_used":False,
        "word_marks_accessed_for_normalization":True,
        "word_marks_emitted":False,
        "emotion_fields_used":False,
        "audio_accessed":False,
        "acoustic_values_accessed":False,
        "reliability_values_accessed":False,
        "clinical_outcomes_accessed":False,
        "acoustic_processing_authorized":False,
    }
    Path(safe_json).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return out

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--zip",type=Path,required=True); p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--structural-csv",type=Path,required=True); p.add_argument("--cmudict",type=Path,required=True)
    p.add_argument("--private-token-manifest",type=Path,required=True); p.add_argument("--eligible-filenames",type=Path,required=True)
    p.add_argument("--safe-json",type=Path,required=True); a=p.parse_args(argv)
    run(a.zip,a.inventory,a.structural_csv,a.cmudict,a.private_token_manifest,a.eligible_filenames,a.safe_json)
    return 0
if __name__=="__main__": raise SystemExit(main())
