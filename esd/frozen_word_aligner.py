#!/usr/bin/env python3
"""Minimal frozen WhisperX-3.3.1-compatible English CTC word aligner.

This is a scoped adaptation of WhisperX 3.3.1 alignment.py functions
load_align_model, align, get_trellis, backtrack, and merge_repeats. It accepts
only externally supplied, whitespace-tokenized English text and the frozen
TorchAudio bundle. It contains no ASR, VAD, diarization, or network path.
"""
from __future__ import annotations
import hashlib,subprocess
from dataclasses import dataclass
from pathlib import Path

HISTORICAL_PACKAGE="whisperx 3.3.1"
HISTORICAL_ALIGNMENT_SHA256="2d5b999ea5fc4a2e0e449471baa954fe1d9af5919aed325515578c0a8a734660"
HISTORICAL_FUNCTIONS=("load_align_model","align","get_trellis","backtrack","merge_repeats")
ALIGN_MODEL="WAV2VEC2_ASR_BASE_960H"
ALIGN_CHECKPOINT_BASENAME="wav2vec2_fairseq_base_ls960_asr_ls960.pth"
ALIGN_CHECKPOINT_SHA256="488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d"
SAMPLE_RATE=16000

class AlignmentContractError(RuntimeError): pass

def file_sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def verify_checkpoint(model_dir:Path)->Path:
    checkpoint=model_dir/ALIGN_CHECKPOINT_BASENAME
    if not checkpoint.is_file(): raise AlignmentContractError("frozen alignment checkpoint absent")
    if file_sha(checkpoint)!=ALIGN_CHECKPOINT_SHA256: raise AlignmentContractError("frozen alignment checkpoint checksum mismatch")
    return checkpoint

def load_audio(path:Path):
    """Match WhisperX 3.3.1 mono PCM16/16-kHz ffmpeg decoding."""
    import numpy as np
    command=["ffmpeg","-nostdin","-threads","0","-i",str(path),"-f","s16le","-ac","1","-acodec","pcm_s16le","-ar",str(SAMPLE_RATE),"-"]
    try: raw=subprocess.run(command,capture_output=True,check=True).stdout
    except subprocess.CalledProcessError as exc: raise AlignmentContractError("ffmpeg audio decode failed") from exc
    return np.frombuffer(raw,np.int16).flatten().astype(np.float32)/32768.0

@dataclass
class Point:
    token_index:int; time_index:int; score:float

@dataclass
class Segment:
    label:str; start:int; end:int; score:float

def get_trellis(emission,tokens,blank_id=0):
    import torch
    num_frame=emission.size(0); num_tokens=len(tokens)
    trellis=torch.empty((num_frame+1,num_tokens+1))
    trellis[0,0]=0; trellis[1:,0]=torch.cumsum(emission[:,0],0)
    trellis[0,-num_tokens:]=-float("inf"); trellis[-num_tokens:,0]=float("inf")
    for time in range(num_frame):
        trellis[time+1,1:]=torch.maximum(trellis[time,1:]+emission[time,blank_id],trellis[time,:-1]+emission[time,tokens])
    return trellis

def backtrack(trellis,emission,tokens,blank_id=0):
    import torch
    token_index=trellis.size(1)-1; start=torch.argmax(trellis[:,token_index]).item(); path=[]
    for time in range(start,0,-1):
        stayed=trellis[time-1,token_index]+emission[time-1,blank_id]
        changed=trellis[time-1,token_index-1]+emission[time-1,tokens[token_index-1]]
        code=tokens[token_index-1] if changed>stayed else blank_id
        path.append(Point(token_index-1,time-1,emission[time-1,code].exp().item()))
        if changed>stayed:
            token_index-=1
            if token_index==0: break
    else: return None
    return path[::-1]

def merge_repeats(path,transcript):
    first=second=0; segments=[]
    while first<len(path):
        while second<len(path) and path[first].token_index==path[second].token_index: second+=1
        score=sum(path[index].score for index in range(first,second))/(second-first)
        segments.append(Segment(transcript[path[first].token_index],path[first].time_index,path[second-1].time_index+1,score)); first=second
    return segments

def prepare_text(text,dictionary):
    clean_char=[]; clean_indices=[]
    for index,char in enumerate(text):
        candidate=char.lower().replace(" ","|")
        if candidate in dictionary: clean_char.append(candidate); clean_indices.append(index)
    return clean_char,clean_indices

def aggregate_words(text,clean_indices,char_segments,ratio,offset):
    """Match WhisperX word boundary, omission, score, and 3-ms precision rules."""
    char_rows=[]; clean_lookup={source:index for index,source in enumerate(clean_indices)}; word_index=0
    for index,char in enumerate(text):
        start=end=score=None
        if index in clean_lookup:
            segment=char_segments[clean_lookup[index]]
            start=round(segment.start*ratio+offset,3); end=round(segment.end*ratio+offset,3); score=round(segment.score,3)
        char_rows.append((word_index,char,start,end,score))
        if index==len(text)-1 or text[index+1]==" ": word_index+=1
    words=[]
    for current in range(word_index):
        rows=[row for row in char_rows if row[0]==current]; word="".join(row[1] for row in rows).strip()
        if not word: continue
        lexical=[row for row in rows if row[1]!=" "]; starts=[row[2] for row in lexical if row[2] is not None]; ends=[row[3] for row in lexical if row[3] is not None]; scores=[row[4] for row in lexical if row[4] is not None]
        result={"word":word}
        if starts: result["start"]=min(starts)
        if ends: result["end"]=max(ends)
        if scores: result["score"]=round(sum(scores)/len(scores),3)
        words.append(result)
    return words

def align_emission(emission,text,dictionary,start,end,blank_id=0):
    clean,indices=prepare_text(text,dictionary)
    if not clean: return [{"word":word} for word in text.split(" ") if word]
    tokens=[dictionary[char] for char in clean]; trellis=get_trellis(emission,tokens,blank_id); path=backtrack(trellis,emission,tokens,blank_id)
    if path is None: return []
    characters=merge_repeats(path,"".join(clean)); ratio=(end-start)/(trellis.size(0)-1)
    return aggregate_words(text,indices,characters,ratio,start)

def load_frozen_model(model_dir:Path,device="cuda"):
    """Instantiate the exact TorchAudio pipeline from a verified local state dict."""
    import torch,torchaudio
    from torchaudio.pipelines._wav2vec2 import utils
    checkpoint=verify_checkpoint(model_dir); bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model=utils._get_model(bundle._model_type,bundle._params)
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if bundle._remove_aux_axis: utils._remove_aux_axes(state,bundle._remove_aux_axis)
    model.load_state_dict(state)
    if bundle._normalize_waveform: model=utils._extend_model(model,normalize_waveform=True)
    return model.eval().to(device),{label.lower():index for index,label in enumerate(bundle.get_labels())}

class FrozenWordAligner:
    def __init__(self,model_dir:Path,device="cuda"):
        self.device=device; self.model,self.dictionary=load_frozen_model(model_dir,device)
    def align(self,audio_path:Path,tokens):
        import torch
        if not tokens: return []
        waveform=torch.from_numpy(load_audio(audio_path)).unsqueeze(0); duration=waveform.shape[1]/SAMPLE_RATE
        lengths=None
        if waveform.shape[-1]<400:
            lengths=torch.as_tensor([waveform.shape[-1]],device=self.device); waveform=torch.nn.functional.pad(waveform,(0,400-waveform.shape[-1]))
        with torch.inference_mode(): emissions,_=self.model(waveform.to(self.device),lengths=lengths); emission=torch.log_softmax(emissions,dim=-1)[0].cpu().detach()
        return align_emission(emission," ".join(token["normalized_word"] for token in tokens),self.dictionary,0.0,duration,0)
