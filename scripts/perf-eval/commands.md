# perf-eval: commands and how to read the results

Measures **quality**, **perplexity** and **throughput** of the current code for one model, on CPU.
Run everything from the repository root in PowerShell.

## Commands

### 1. Build

```powershell
python scripts\perf-eval\build.py
```

Builds `llama-perplexity`, `llama-batched-bench` and `edge-probe` (from `probe.cpp`) into `build\bin\`.
Requires CMake, Ninja and MinGW `gcc`/`g++` on `PATH`. Re-run after every code change. The build is
incremental, so small changes rebuild quickly.

### 2. Benchmark

```powershell
python scripts\perf-eval.py benchmark scripts\perf-eval\gemma-e4b.json
python scripts\perf-eval.py benchmark scripts\perf-eval\qwen.json
```

Each run takes about 8–9 minutes and measures the code that was last built. Run one benchmark at a
time: two at once each get half the CPU and both report wrong speeds. Close other heavy programs too
(including any `llama-server`).

### Artemis

| Step | Command |
|---|---|
| Compile | `python scripts/perf-eval/build.py` |
| Benchmark | `python scripts/perf-eval.py benchmark scripts/perf-eval/gemma-e4b.json` |

## What a run does

1. **Quality** on this copy of the code:
   - smoke test: 8 prompts with exact expected answers (`data\smoke.txt`)
   - accuracy: 64 ARC multiple-choice questions (`data\screen.*`)
   - perplexity: 8 chunks of 512 tokens from `data\ppl.txt`
2. **Broken-model check.** The run stops with an error, and writes no results, if fewer than 7 of the
   8 smoke prompts are right or perplexity is invalid.
3. **Throughput** with `llama-batched-bench`, 3 times: a 256-token prompt, then 32 generated tokens, with 1 and
   with 4 sequences in parallel.
4. **Results:**
   - `artemis_results.json` in the repo root: numbers only, read by Artemis. It is deleted at the start
     of every run, so a failed run never leaves old numbers behind.
   - `run_info.json` in the repo root: what was measured (see below). Kept separate because Artemis
     rejects a results file that contains text.
   - `..\perf-eval-results\<eval>\artemis_results.json`: the run info and the numbers together,
     outside the repo.
   - `..\perf-eval-results\<eval>\<config-hash>\`: logs, every answer given, and the full results.

There are no stored baselines: each run measures the code it is given. Artemis keeps the results of
earlier versions and compares against them.

## How to read the results

### Run info (`run_info.json`)

Also printed on the first line of the benchmark output (`RUN {...}`), which Artemis keeps in the task log.

| Field | Meaning |
|---|---|
| `eval` | config name, e.g. `gemma-4-e4b` |
| `model` | model file name |
| `model_path`, `model_bytes`, `model_sha256` | exact model file measured; the checksum proves two runs used the same file |
| `build_timestamp` | when the measured tools in `build\bin` were last built. If it's older than your code change, you forgot to rebuild |
| `git_commit` | commit of the code copy that was measured (uncommitted edits aren't included) |
| `run_timestamp` | when the benchmark ran |

### Quality

| Field | Meaning | Better | How to read a change |
|---|---|---|---|
| `quality_correct` | questions answered correctly, out of 64 | higher | each question is about 1.6%. A change of 1–2 is small; losing 3 or more means the change hurt the model |
| `smoke_passed` | basic instructions followed exactly, out of 8 | higher | 7–8 is healthy; below 7 the run stops because the model is broken |

### Perplexity

| Field | Meaning | Better | How to read a change |
|---|---|---|---|
| `perplexity` | how well the model predicts real text | lower | under 1% higher: no real effect (kernel changes often move it slightly). 1–5% higher: small loss. Over 5% higher: real quality loss. A huge value (1000+) means broken output |

Only compare perplexity between versions of the same model. Different models use different tokenizers,
so their values can't be compared with each other.

### Throughput

| Field | Meaning | Better |
|---|---|---|
| **`generation_tokens_per_second`** | **headline number:** how fast the model writes its answer for one user | higher |
| `prompt_tokens_per_second` | how fast it reads the input, which sets the wait before the first word | higher |
| `batched_generation_tokens_per_second` | total writing speed with 4 users at once | higher |
| `batched_total_tokens_per_second` | reading and writing combined, 4 users | higher |
| `model_size_mib` | size of the model file | lower |

Speed varies from run to run (see "Speed on this laptop" below). Each run reports the median of
3 throughput runs; still treat a change under ±5% as noise and run again. A change over 5% that
repeats is real.

### Reading one result

1. Is `quality_correct` within 2 of the original, and `perplexity` within 5%? If not, the version
   made the model worse.
2. If quality held, compare `generation_tokens_per_second`: higher is faster.

### Baseline on this machine (unmodified code)

| | Gemma 4 E4B Q4_0 | Qwen3.5-35B-A3B Q4_K_M |
|---|---|---|
| `generation_tokens_per_second` | 6.57 | 4.18 |
| `prompt_tokens_per_second` | 51.9 | 51.4 |
| `batched_generation_tokens_per_second` | 18.8 | 13.1 |
| `batched_total_tokens_per_second` | 44.1 | 51.0 |
| `quality_correct` | 60 | 64 |
| `smoke_passed` | 7 | 8 |
| `perplexity` | 27.40 | 4.81 |

Gemma's quality numbers match the original Linux harness (7/8, 60/64, perplexity 27.40). Its one smoke
miss is expected: it answers `70` instead of `69.8`.

## Configs

Each eval is a JSON file in `scripts\perf-eval\` with a `name` and a `model`. Any other key overrides a
default in `perf-eval.py`, for example:

```json
{
  "name": "gemma-4-e4b",
  "model": "C:\\Users\\aiDAPTIV\\Desktop\\models\\gemma-4-E4B-it-Q4_0.gguf",
  "repack": false,
  "threads": 8,
  "cache_k": "q8_0"
}
```

| Key | Default | What it sets |
|---|---|---|
| `threads` | 4 | CPU threads for generating tokens (one per fast core) |
| `threads_batch` | 16 | CPU threads for reading prompts and perplexity |
| `repeats` | 3 | throughput runs; the median of each speed is reported |
| `batch` / `ubatch` | 2048 / 512 | batch sizes |
| `cache_k` / `cache_v` | `f16` | KV cache type (`f16`, `q8_0`, `q4_0`, …) |
| `flash` | `auto` | flash attention (`on`, `off`, `auto`) |
| `repack` | `true` | CPU weight repacking |
| `chat_template` | Gemma 4 format | how prompts are wrapped; `{prompt}` is replaced |
| `ppl_chunks` | 8 | perplexity chunks |
| `smoke_min` | 7 | smoke answers needed to count as not broken |
| `npp` / `ntg` / `npl` | `[256]` / `[32]` / `[1, 4]` | throughput prompt length, generated tokens, parallel sequences |

Unknown keys are rejected. To add a model, copy a config and change `name` and `model`. Models with
another chat format need their own `chat_template`; see `qwen.json`, which also switches off Qwen's
thinking so it answers directly.

## Speed on this laptop

The Intel Core Ultra X7 368H has three kinds of cores: CPUs 0–3 are 4 fast performance cores, CPUs
4–11 are 8 efficiency cores, and CPUs 12–15 are 4 low-power cores. Two things follow:

- **Thread count.** llama.cpp splits each step evenly across threads, so with 16 threads the fast
  cores wait for the low-power ones. Generation measured about twice as fast with 4 threads as with
  16, hence `threads: 4`. Reading prompts still gains from more cores, hence `threads_batch: 16`.
- **Clock speed changes during a run.** The fast cores start at about 4.8 GHz and drop to about
  2.7 GHz after a few seconds (power limit), so identical runs can differ by up to 2×. The median of 3
  runs reduces this. For steadier numbers, keep the laptop plugged in, set Windows power mode to
  **Best performance**, and run nothing else at the same time, including other benchmarks.

## Known issues

- **Gemma 4 E4B crashes with weight repacking on** (exit code `3221225477` / `0xC0000005`), so
  `gemma-e4b.json` sets `"repack": false`.
- **Gemma 4 26B A4B (Unsloth Q4_K_M) gives meaningless perplexity** (about 20,000) with repacking on or
  off; the cause is not yet known. Don't use it for quality measurements.
