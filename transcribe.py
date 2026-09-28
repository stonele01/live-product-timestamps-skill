"""Bounded-memory, local-only SenseVoice transcription with explicit window timing."""
import argparse
import json
import re
import os
import time
import wave
from pathlib import Path
PROCESS_STARTED = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--duration', type=float)
    ap.add_argument('--window', type=float, default=20)
    ap.add_argument('--overlap', type=float, default=2)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    ap.add_argument('--timestamps', action='store_true')
    args = ap.parse_args()
    if args.window <= 0 or args.overlap < 0 or args.window + 2 * args.overlap > 30 or args.batch_size < 1:
        ap.error('Use positive windows/batches and at most 30 seconds including context')
    model_path = Path(args.model).resolve()
    if not (model_path / 'model.pt').is_file():
        ap.error('A downloaded local SenseVoiceSmall model is required')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    import torch
    from funasr import AutoModel
    if os.name == 'nt':
        # SentencePiece's native file loader can fail on Windows Unicode paths.
        import sentencepiece
        from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
        def load_tokenizer_bytes(self):
            if self.sp is None:
                self.sp = sentencepiece.SentencePieceProcessor(model_proto=Path(self.bpemodel).read_bytes())
        SentencepiecesTokenizer._build_sentence_piece_processor = load_tokenizer_bytes
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-enabled PyTorch installation and NVIDIA GPU are required')
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; test --dtype float32 --batch-size 1')
    started = time.perf_counter()
    # Avoid FunASR's full checkpoint deepcopy, which exhausts Windows commit
    # headroom when several desktop apps are open. Strict loading checks keys.
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(model_path / 'config.yaml'), resolve=True)
    config['tokenizer_conf']['bpemodel'] = str(model_path / 'chn_jpn_yue_eng_ko_spectok.bpe.model')
    config['frontend_conf']['cmvn_file'] = str(model_path / 'am.mvn')
    config.update(device='cuda:0', bf16=args.dtype=='bfloat16', disable_update=True, disable_pbar=True, ncpu=4,
                  trust_remote_code=False)
    model = AutoModel(**config)
    state = torch.load(str(model_path / 'model.pt'), map_location='cpu', mmap=True, weights_only=True)
    for key in ('state_dict', 'model_state_dict', 'model'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
    model.model.load_state_dict(state, strict=True)
    del state
    import gc
    gc.collect()
    model.model.eval()
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    print(f'Model loaded: {loaded-started:.2f}s', flush=True)
    rows = []
    with wave.open(str(Path(args.audio).resolve()), 'rb') as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (1, 16000, 2):
            raise ValueError('Expected mono 16kHz PCM16 WAV')
        total = source.getnframes() / 16000
        stop = total if args.duration is None else min(total, args.start + args.duration)
        if not 0 <= args.start < stop:
            raise ValueError('Requested interval is outside the audio')
        cursor = args.start
        with (out / 'segments.jsonl').open('w', encoding='utf-8') as log:
            while cursor < stop:
                batch, windows = [], []
                for _ in range(args.batch_size):
                    if cursor >= stop:
                        break
                    core_end = min(stop, cursor + args.window)
                    left, right = max(0, cursor - args.overlap), min(total, core_end + args.overlap)
                    source.setpos(round(left * 16000))
                    data = np.frombuffer(source.readframes(round((right-left)*16000)), dtype='<i2').astype(np.float32) / 32768
                    batch.append(data)
                    windows.append((cursor, core_end, left, right))
                    cursor = core_end
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.dtype=='bfloat16'):
                    predictions = model.generate(input=batch, language='zh', use_itn=True,
                                                 device='cuda:0', batch_size=len(batch), disable_pbar=True,
                                                 output_timestamp=args.timestamps)
                if len(predictions) != len(windows):
                    raise ValueError('Model did not return one result per input window')
                for pred, (core_start, core_end, left, right) in zip(predictions, windows):
                    row = dict(core_start=core_start, core_end=core_end, start=left, end=right,
                               timing_kind='input_window_not_speech_alignment',
                               text=re.sub(r'<\|[^|]*\|>', '', pred['text']).strip(), raw_text=pred['text'])
                    if 'timestamp' in pred:
                        row['words'] = pred['words']
                        row['word_timestamps'] = [[round(left+a/1000,3), round(left+b/1000,3)] for a,b in pred['timestamp']]
                    rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
                log.flush()
                if len(rows) % 20 < args.batch_size:
                    print(f'Covered {cursor:.1f}/{stop:.1f}s; elapsed {time.perf_counter()-loaded:.1f}s', flush=True)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    meta = dict(engine='SenseVoiceSmall', model_path=str(model_path), audio=str(Path(args.audio).resolve()),
                start=args.start, end=stop, audio_seconds=stop-args.start, window=args.window,
                overlap=args.overlap, batch_size=args.batch_size, device='cuda:0', dtype=args.dtype,
                output_timestamp=args.timestamps,
                model_load_seconds=loaded-started, transcribe_seconds=finished-loaded,
                total_seconds=finished-started, segments=rows)
    meta['process_seconds'] = finished - PROCESS_STARTED
    (out / 'result.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    def stamp(v):
        return f'{int(v)//3600:02d}:{int(v)%3600//60:02d}:{v%60:05.2f}'
    (out / 'transcript.txt').write_text('\n'.join(f"[{stamp(r['start'])}–{stamp(r['end'])}] {r['text']}" for r in rows), encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='segments'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
