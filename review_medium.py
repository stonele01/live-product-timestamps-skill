"""Re-transcribe explicit uncertainty intervals using a local Whisper model."""
import argparse, json, os, time, wave
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audio',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--ranges',required=True,help='JSON list of {start,end,reason}')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    ranges=json.loads(Path(args.ranges).read_text(encoding='utf-8'))
    with wave.open(args.audio,'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):
            raise ValueError('Expected mono 16kHz PCM16 WAV')
        duration=w.getnframes()/16000
    for r in ranges:
        if not 0<=r['start']<r['end']<=duration:
            raise ValueError(f'Invalid review range: {r}')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    import numpy as np
    from faster_whisper import WhisperModel,BatchedInferencePipeline
    began=time.perf_counter()
    model=WhisperModel(args.model,device='cuda',compute_type='int8_float16',local_files_only=True)
    loaded=time.perf_counter();results=[]
    with wave.open(args.audio,'rb') as source:
        for i,r in enumerate(ranges):
            source.setpos(round(r['start']*16000))
            audio=np.frombuffer(source.readframes(round((r['end']-r['start'])*16000)),dtype='<i2').astype(np.float32)/32768
            ts=time.perf_counter()
            segments,_=BatchedInferencePipeline(model=model).transcribe(audio,language='zh',beam_size=1,batch_size=1,vad_filter=True,word_timestamps=False,without_timestamps=False)
            rows=[dict(start=r['start']+s.start,end=r['start']+s.end,text=s.text) for s in segments]
            result=dict(**r,seconds=time.perf_counter()-ts,segments=rows)
            results.append(result)
            (out/f'review-{i:03}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f"Reviewed {i+1}/{len(ranges)} in {result['seconds']:.2f}s",flush=True)
    d=dict(engine='faster-whisper-medium',batch_size=1,beam_size=1,compute_type='int8_float16',model_load_seconds=loaded-began,transcribe_seconds=sum(r['seconds'] for r in results),total_seconds=time.perf_counter()-began,ranges=results)
    (out/'result.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in d.items() if k!='ranges'}),flush=True)

if __name__=='__main__':
    main()
