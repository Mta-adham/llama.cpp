#!/usr/bin/env python3
"""Throughput + quality harness for llama.cpp (Windows-friendly).

Quality follows journey-video/review/llamacpp-config-discovery/harness-v2.py:
  smoke       8 prompts with exact expected answers (data/smoke.txt), must pass >= 7
  tasks       64 ARC multiple-choice questions (data/screen.*), answered by greedy decoding
  perplexity  8 x 512-token chunks of data/ppl.txt

Modes:
  reference  run the quality checks and save them as this eval's reference
  benchmark  run the quality checks, gate them against the reference, run batched-bench
             throughput, then write metrics to artemis_results.json

Each eval is a JSON config overriding keys in DEFAULT; 'name' keeps its reference and
results separate under ../perf-eval-results/<name>/ (outside the checkout).

  python scripts/perf-eval.py reference scripts/perf-eval/gemma-e4b.json
  python scripts/perf-eval.py benchmark scripts/perf-eval/gemma-e4b.json

The probe is built from scripts/perf-eval/probe.cpp, see scripts/perf-eval/build-probe.sh.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'build' / 'bin'
DATA = ROOT / 'scripts' / 'perf-eval' / 'data'
# kept outside the checkout: the Artemis runner picks up any artemis_results.json inside it
OUT = ROOT.parent / 'perf-eval-results'
EXE = '.exe' if sys.platform == 'win32' else ''

GEMMA_TEMPLATE = '<|turn>user\n{prompt}<turn|>\n<|turn>model\n'

DEFAULT = {
    'name': None,
    'model': None,
    'threads': 16,
    'batch': 2048,
    'ubatch': 512,
    'cache_k': 'f16',
    'cache_v': 'f16',
    'flash': 'auto',
    'repack': True,
    # quality
    'chat_template': GEMMA_TEMPLATE,  # prompt wrapper used by the probe, '{prompt}' is replaced
    'ppl_text': str(DATA / 'ppl.txt'),
    'ppl_chunks': 8,
    'ppl_tolerance': 0.05,  # allowed relative perplexity increase vs reference
    'task_tolerance': 2,    # allowed drop in correct answers, overall and per subset
    'smoke_min': 7,         # smoke prompts out of 8 that must match exactly
    # batched-bench
    'ctx': 8192,
    'npp': [512],
    'ntg': [64],
    'npl': [1, 4],
}
# llama-perplexity names -> ggml_type ids for the probe
CACHE_TYPES = {'f32': 0, 'f16': 1, 'q4_0': 2, 'q4_1': 3, 'q5_0': 6, 'q5_1': 7, 'q8_0': 8, 'iq4_nl': 20, 'bf16': 30}
SMOKE_EXPECT = ['42', 'Bristol', [1, 2, 5, 9], '69.8', 'blue', '12', 'weather', {'status': 'ok', 'count': 3}]
EOG_TEXT = ['<turn|>', '<|im_end|>', '<end_of_turn>', '<eos>']


def config(path):
    c = dict(DEFAULT)
    v = json.loads(Path(path).read_text())
    assert isinstance(v, dict) and not (set(v) - set(c)), f'unknown config keys: {set(v) - set(c)}'
    c.update(v)
    assert isinstance(c['name'], str) and re.fullmatch(r'[A-Za-z0-9_.-]+', c['name']), 'config needs a "name"'
    assert c['model'] and Path(c['model']).is_file(), f"model not found: {c['model']}"
    assert Path(c['ppl_text']).is_file(), f"perplexity text not found: {c['ppl_text']}"
    assert '{prompt}' in c['chat_template'], 'chat_template needs a {prompt} placeholder'
    assert c['ubatch'] <= c['batch'], 'ubatch must be <= batch'
    assert c['cache_k'] in CACHE_TYPES and c['cache_v'] in CACHE_TYPES, 'bad KV cache type'
    assert c['flash'] in ('on', 'off', 'auto')
    assert isinstance(c['repack'], bool)
    for k in ['npp', 'ntg', 'npl']:
        assert isinstance(c[k], list) and c[k] and all(isinstance(x, int) and x > 0 for x in c[k]), k
    need = max(c['npl']) * (max(c['npp']) + max(c['ntg']))
    assert c['ctx'] >= need, f"ctx {c['ctx']} too small for batched-bench, need >= {need}"
    return c


def identity(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=2) + '\n')


def run(args, log, env=None, timeout=6 * 3600):
    args = list(map(str, args))
    t0 = time.monotonic()
    p = subprocess.run(args, capture_output=True, env=env, timeout=timeout)
    out = p.stdout.decode(errors='replace') + '\n' + p.stderr.decode(errors='replace')
    Path(log).write_text(' '.join(args) + '\n\n' + out, encoding='utf-8')
    if p.returncode:
        raise RuntimeError(f'{Path(args[0]).name} exit {p.returncode}, see {log}\n' + out[-1500:])
    return p.stdout.decode(errors='replace'), out, time.monotonic() - t0


def common_args(c):
    args = ['-m', c['model'], '-t', c['threads'], '-b', c['batch'], '-ub', c['ubatch'],
            '-fa', c['flash'], '-ctk', c['cache_k'], '-ctv', c['cache_v']]
    if not c['repack']:
        args += ['--no-repack']
    return args


def probe(c, prompts, log, tokens):
    env = dict(os.environ)
    env.update({'EDGE_CTX': '2048', 'EDGE_BATCH': str(min(c['batch'], 2048)), 'EDGE_UBATCH': str(c['ubatch']),
                'EDGE_REPACK': str(int(c['repack'])), 'EDGE_FA': str(int(c['flash'] == 'on')),
                'EDGE_K': str(CACHE_TYPES[c['cache_k']]), 'EDGE_V': str(CACHE_TYPES[c['cache_v']]),
                'EDGE_TOKENS': str(tokens), 'EDGE_TEMPLATE': c['chat_template']})
    stdout, _, _ = run([BIN / f'edge-probe{EXE}', c['model'], prompts, c['threads'], 'quality'], log, env)
    return [json.loads(l) for l in stdout.splitlines() if l.startswith('{')]


def answer(row):
    s = row['text'].strip()
    for eog in EOG_TEXT:
        s = s.removesuffix(eog).strip()
    return s


def smoke(rows):
    checks = []
    for row, e in zip(rows, SMOKE_EXPECT):
        s = answer(row)
        if s.startswith('```'):
            s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s).strip()
        try:
            a = json.loads(s) if isinstance(e, (dict, list)) else s
        except ValueError:
            a = None
        checks.append(a == e)
    return checks


def tasks(rows, split):
    data = json.loads((DATA / f'{split}.json').read_text())
    assert len(rows) == len(data) == 64, (len(rows), len(data))
    checks = [answer(r) == d['answerKey'] for r, d in zip(rows, data)]
    return {'correct': sum(checks), 'subsets': [sum(checks[:32]), sum(checks[32:])], 'checks': checks}


def perplexity(c, D):
    args = [BIN / f'llama-perplexity{EXE}', *common_args(c), '-f', c['ppl_text'], '-c', 512, '--chunks', c['ppl_chunks']]
    _, out, secs = run(args, D / 'ppl.log')
    m = re.findall(r'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)', out)
    assert m, f"no perplexity estimate in {D / 'ppl.log'}"
    ppl, err = map(float, m[-1])
    return {'ppl': ppl, 'ppl_err': err, 'ppl_seconds': secs}


def quality(c, D):
    sm = probe(c, DATA / 'smoke.txt', D / 'smoke.log', 64)
    rows = probe(c, DATA / 'screen.txt', D / 'screen.log', 8)
    q = {'smoke': sum(smoke(sm)), 'smoke_checks': smoke(sm), **tasks(rows, 'screen'), **perplexity(c, D)}
    q['smoke_answers'] = [answer(r) for r in sm]
    q['answers'] = [answer(r) for r in rows]
    return q


def gate(q, ref, c):
    tol = c['task_tolerance']
    assert q['smoke'] >= c['smoke_min'], f"smoke {q['smoke']}/8 below {c['smoke_min']}/8"
    assert q['correct'] >= ref['correct'] - tol, f"tasks {q['correct']}/64 dropped more than {tol} from reference {ref['correct']}/64"
    assert all(a >= b - tol for a, b in zip(q['subsets'], ref['subsets'])), f"subset {q['subsets']} dropped more than {tol} from reference {ref['subsets']}"
    assert math.isfinite(q['ppl']), 'perplexity is not finite'
    limit = ref['ppl'] * (1 + c['ppl_tolerance'])
    assert q['ppl'] <= limit, f"perplexity {q['ppl']:.4f} > {limit:.4f} (reference {ref['ppl']:.4f} + {c['ppl_tolerance']:.0%})"


def summary(q):
    return {'smoke': f"{q['smoke']}/8", 'tasks': f"{q['correct']}/64", 'subsets': q['subsets'], 'ppl': q['ppl']}


def main(mode, path):
    start = time.monotonic()
    c = config(path)
    E = OUT / c['name']
    D = E / identity(c)
    D.mkdir(parents=True, exist_ok=True)
    dump(D / 'config.json', c)
    reference = E / 'reference.json'

    if mode == 'reference':
        q = quality(c, D)
        dump(reference, {'config': c, **q})
        print('REFERENCE ' + json.dumps(summary(q)))
    elif mode == 'benchmark':
        Path('artemis_results.json').unlink(missing_ok=True)
        assert reference.exists(), f'no reference for {c["name"]} yet, run: python scripts/perf-eval.py reference {path}'
        ref = json.loads(reference.read_text())
        assert Path(ref['config']['model']).name == Path(c['model']).name, \
            f"reference was made with {Path(ref['config']['model']).name}, rerun: python scripts/perf-eval.py reference {path}"
        q = quality(c, D)
        print('QUALITY ' + json.dumps(summary(q)) + '  reference ' + json.dumps(summary(ref)), flush=True)
        gate(q, ref, c)

        rows = throughput(c, D)
        single = [r for r in rows if r['pl'] == 1] or rows
        widest = [r for r in rows if r['pl'] == max(c['npl'])]
        m = {
            'prompt_tokens_per_second': max(r['speed_pp'] for r in single),
            'generation_tokens_per_second': max(r['speed_tg'] for r in single),
            'batched_generation_tokens_per_second': max(r['speed_tg'] for r in widest),
            'batched_total_tokens_per_second': max(r['speed'] for r in widest),
            'quality_correct': q['correct'],
            'smoke_passed': q['smoke'],
            'perplexity': q['ppl'],
            'model_size_mib': Path(c['model']).stat().st_size / 1048576,
        }
        assert all(math.isfinite(v) and v > 0 for v in m.values()), m
        dump(D / f"sample-{time.strftime('%Y%m%dT%H%M%S')}.json", {'config': c, 'metrics': m, 'quality': q, 'bench': rows})
        # labelled copy per eval; the one in the checkout must stay numeric-only for Artemis
        dump(E / 'artemis_results.json', {'eval': c['name'], 'model': Path(c['model']).name, 'metrics': m})
        dump('artemis_results.tmp', m)
        Path('artemis_results.tmp').replace('artemis_results.json')
        print(f"eval: {c['name']}  model: {Path(c['model']).name}")
        print(json.dumps(m, indent=2))
    else:
        raise ValueError(f'unknown mode {mode!r}, expected reference or benchmark')
    print(f'{mode}: {time.monotonic() - start:.1f}s (logs in {D})', flush=True)


def throughput(c, D):
    args = [BIN / f'llama-batched-bench{EXE}', *common_args(c), '-c', c['ctx'],
            '-npp', ','.join(map(str, c['npp'])), '-ntg', ','.join(map(str, c['ntg'])),
            '-npl', ','.join(map(str, c['npl'])), '--output-format', 'jsonl']
    _, out, _ = run(args, D / 'batched-bench.log')
    rows = [json.loads(l) for l in out.splitlines() if l.startswith('{"n_kv_max"')]
    assert rows, f"no jsonl rows in {D / 'batched-bench.log'}"
    return rows


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: perf-eval.py <reference|benchmark> <config.json>')
    main(sys.argv[1], sys.argv[2])
