#!/usr/bin/env python3
"""Build everything perf-eval.py needs into build/bin: llama-perplexity, llama-batched-bench and
edge-probe (scripts/perf-eval/probe.cpp). CPU-only MinGW/gcc build with the C++ runtime linked
statically so the binaries do not depend on whichever libstdc++ is first on PATH.

  python scripts/perf-eval/build.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build'
JOBS = str(os.cpu_count() or 8)


def run(*args):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


run('cmake', '-B', BUILD, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_C_COMPILER=gcc', '-DCMAKE_CXX_COMPILER=g++', '-DCMAKE_EXE_LINKER_FLAGS=-static')
run('cmake', '--build', BUILD, '-t', 'llama-perplexity', 'llama-batched-bench', 'llama', '-j', JOBS)

exe = '.exe' if sys.platform == 'win32' else ''
libs = [BUILD / 'src' / 'libllama.a'] + [BUILD / 'ggml' / 'src' / f'{n}.a' for n in ('ggml', 'ggml-cpu', 'ggml-base')]
extra = ['-static', '-lpsapi'] if sys.platform == 'win32' else ['-lpthread']
run('g++', '-O2', '-std=c++17', '-fopenmp', '-Iinclude', '-Iggml/include', 'scripts/perf-eval/probe.cpp',
    *libs, *extra, '-o', BUILD / 'bin' / f'edge-probe{exe}')
print('built: ' + ', '.join(f'{n}{exe}' for n in ('llama-perplexity', 'llama-batched-bench', 'edge-probe')))
