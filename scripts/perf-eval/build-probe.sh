#!/bin/sh
# Builds build/bin/edge-probe(.exe) against the static libraries of an existing CPU build:
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-static
#   cmake --build build -t llama-batched-bench llama-perplexity
set -e
cd "$(dirname "$0")/../.."
g++ -O2 -std=c++17 -fopenmp -Iinclude -Iggml/include scripts/perf-eval/probe.cpp \
    build/src/libllama.a build/ggml/src/ggml.a build/ggml/src/ggml-cpu.a build/ggml/src/ggml-base.a \
    -static -lpsapi -o build/bin/edge-probe.exe
