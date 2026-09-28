"""Make complete, time-labelled reading packs from aligned SenseVoice JSONL."""
import argparse
import json
import math
from pathlib import Path


def clock(seconds):
    whole = int(seconds)
    return f'{whole // 3600:02}:{whole // 60 % 60:02}:{whole % 60:02}'


def prepare(source, output, chunk_seconds=1800, context_seconds=60):
    if chunk_seconds <= 0 or context_seconds < 0:
        raise ValueError('Invalid chunk/context duration')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not rows:
        raise ValueError('No complete segments')
    words, warnings = [], []
    previous = float(rows[0]['core_start'])
    for index, row in enumerate(rows):
        start, end = float(row['core_start']), float(row['core_end'])
        if abs(start - previous) > 0.05 or end <= start:
            raise ValueError(f'Non-contiguous core windows at row {index + 1}')
        previous = end
        tokens, times = row.get('words', []), row.get('word_timestamps', [])
        if not tokens or len(tokens) != len(times):
            raise ValueError(f'Missing/mismatched word alignment at row {index + 1}')
        kept = []
        last_word_start = float('-inf')
        for token, (a, b) in zip(tokens, times):
            if not all(math.isfinite(float(v)) for v in (a, b)) or b < a or a < row['start'] - .1 or b > row['end'] + .1:
                raise ValueError(f'Invalid timestamp at row {index + 1}')
            if a < last_word_start:
                raise ValueError(f'Unordered word alignment at row {index + 1}')
            last_word_start = a
            if start <= (a + b) / 2 < end:
                kept.append({'text': token, 'start': a, 'end': b})
        if not kept:
            warnings.append({'core_start': start, 'core_end': end, 'reason': 'No owned aligned words; inspect raw window/audio'})
        words.extend(kept)
    words.sort(key=lambda word: (word['start'], word['end']))
    sentences, pending = [], []
    for word in words:
        pending.append(word)
        if word['text'] in '。！？!?\n' or word['end'] - pending[0]['start'] >= 15:
            sentences.append({'start': pending[0]['start'], 'end': pending[-1]['end'], 'text': ''.join(w['text'] for w in pending)})
            pending = []
    if pending:
        sentences.append({'start': pending[0]['start'], 'end': pending[-1]['end'], 'text': ''.join(w['text'] for w in pending)})
    output.mkdir(parents=True, exist_ok=False)
    first, last = rows[0]['core_start'], rows[-1]['core_end']
    manifest = {'start': first, 'end': last, 'windows': len(rows), 'owned_words': len(words), 'warnings': warnings, 'packs': []}
    for number in range(1, math.ceil((last - first) / chunk_seconds) + 1):
        core_start = first + (number - 1) * chunk_seconds
        core_end = min(core_start + chunk_seconds, last)
        start, end = max(first, core_start - context_seconds), min(last, core_end + context_seconds)
        selected = [s for s in sentences if s['end'] >= start and s['start'] < end]
        name = f'part-{number:02}.txt'
        header = f'Core: {core_start:.2f}–{core_end:.2f} seconds; context: {start:.2f}–{end:.2f}.\nAll times relative to input WAV. Lines are ASR alignment, not product boundaries.\n'
        output.joinpath(name).write_text(header + '\n'.join(f"[{s['start']:.2f}–{s['end']:.2f} | {clock(s['start'])}] {s['text']}" for s in selected), encoding='utf-8')
        manifest['packs'].append({'file': name, 'core_start': core_start, 'core_end': core_end, 'lines': len(selected)})
    output.joinpath('words.json').write_text(json.dumps(words, ensure_ascii=False), encoding='utf-8')
    output.joinpath('manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-seconds', type=int, default=1800)
    parser.add_argument('--context-seconds', type=int, default=60)
    args = parser.parse_args()
    result = prepare(args.segments, args.output, args.chunk_seconds, args.context_seconds)
    print(json.dumps({key: value for key, value in result.items() if key != 'packs'}, ensure_ascii=False))
