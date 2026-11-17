# bench: the two uses of a SQLite store, measured

This directory holds the measurements behind this branch, the tools that repeat them,
and the write-ups of what they found. Building llama.cpp does not need any of it.

## The two uses

An LLM spends its cost in two places: reading tokens into its context, and computing
over them. A SQLite store keeps work that does not need doing again. It serves two
separate uses, at two different points:

| | Use 1: speed and token reduction | Use 2: out-of-memory mitigation |
|---|---|---|
| **Problem** | The same files are read and the same answers derived again in every session, spending tokens and time. | A long context runs out of GPU memory while it is computed, even though the model and its KV cache fit. |
| **What the store holds** | What the model derived, as text: answers and summaries, keyed by the question and the files they came from. | What the engine computed: the KV cache after a long context (the depth state), keyed by model file, depth, and K and V cache types. |
| **How it helps** | A harness gives the model stored entries instead of file reads: shorter prompts, fewer turns, fewer generated tokens. | `llama-bench --store` copies the stored KV cache into memory reserved at startup. The computation that runs out of memory never runs. |
| **Where it applies** | Any model: llama.cpp, or a hosted one through its API. | llama.cpp, or any engine you run yourself: only there can you restore a KV cache. |
| **Suite** | `reuse` | `state` |
| **Measured (v0.9.0)** | Qwen3.8-27B, 13 repeated C++ tasks, warm store against none: 73% fewer prompt tokens computed, 58% fewer generated, 64% less time, 13/13 correct. Filling the store costs +26% prompt tokens and +93% time; it pays back after one reuse in tokens, two in time. | Gemma 4 31B at a 65,536-token context on a 24 GB GPU: without the store, prefill runs out of memory; with it and CUDA graphs off, the run finishes at 20.09 ± 0.27 t/s. The control (graphs off, no store) still fails. |
| **Write-up** | [`reuse-store-benchmark.md`](reuse-store-benchmark.md) | [`llama-bench-oom-reuse-sqlite.md`](llama-bench-oom-reuse-sqlite.md), from [`llama-bench-oom-gemma-4-31b.md`](llama-bench-oom-gemma-4-31b.md) |

They do not substitute for each other. Use 1 does not change the memory a context
reserves at startup, so it cannot stop use 2's failure. Use 2 does not shorten prompts;
it removes the computation of tokens already processed once.

## The lldb trace: why the store removes the error

The trace is how the out-of-memory error was located, and how the store's effect was
seen directly rather than inferred. It is the evidence to replicate.

Without it, the result would have been missed. Earlier measurements tracked only the
scratch pool. They could not see the memory that kernel code and CUDA graphs take, so
they never found the second error (CUDA graphs, below). With the store on, the run still
failed, and the store would have looked like it did not work. The trace showed the first
error gone and a second, separate one in its place, and so led to the run that finishes.
It also found why llama-server did not fail where llama-bench did.

**What it records.** [`llama-bench-oom-trace.lldb`](llama-bench-oom-trace.lldb) is a
plain lldb script (no Python). It stops at each phase of a llama-bench run (warmup, the
restore from the store, the depth computation, measured generation), at every growth of
CUDA's scratch pool (`cuMemCreate` in `ggml-cuda.cu`), and at every CUDA graph
instantiation. At each stop it asks CUDA for the free GPU memory (`cudaMemGetInfo`,
evaluated in the stopped process) and prints it, then continues. The result is a
step-by-step table of free memory through the run.

**What it found** (Gemma 4 31B, 65,536-token context, 24 GB GPU, v0.9.0):

1. **Startup leaves 36.7 MiB free.** llama.cpp reserves the model (16,818 MiB), the KV
   cache for the whole context (6,320 MiB) and compute buffers (619.53 MiB) at startup.
   Memory needed after that, by the scratch pool, kernel code and CUDA graphs, is not
   counted by the startup check.
2. **The warmup takes it to 4.7 MiB.**
3. **Error 1, prefill.** The first batch of the depth asks the scratch pool for 5.12 MiB
   more (the matrix multiply's fixup buffer) with 2.7 MiB free: out of memory. It fails
   on the first batch. How full the context is plays no part.
4. **The store removes error 1.** With the store, the depth is restored instead of
   computed: free memory stays at 4.7 MiB across the 6.19 GB copy. The copy writes into
   the KV cache already reserved and runs no kernel, so the failing request never
   happens. The trace shows this, rather than leaving it as an argument.
5. **Error 2, CUDA graphs.** Generation then fails building the fourth CUDA graph chunk
   (`cudaGraphInstantiate`) with 2.7 MiB free. `GGML_CUDA_DISABLE_GRAPHS=1` removes it.
6. **The control.** With graphs off and no store, error 1 still occurs, in the same place.
   Only the store removes it.

The same trace explains why llama-server did not show the error at 65,536 tokens: it
reserves 386 MiB less at startup than llama-bench (output for one token instead of up
to 512), and reproduces it at 70,400.

**Replicating it.** The trace needs host debug symbols: build with
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` (GPU code is unchanged). Breakpoints are source line
numbers, so each version needs its own script:

- [`llama-bench-oom-trace.lldb`](llama-bench-oom-trace.lldb): the script that measured
  v0.9.0, kept as recorded.
- [`llama-bench-oom-trace-v1.0.0.lldb`](llama-bench-oom-trace-v1.0.0.lldb): the same
  breakpoints on this branch at v1.0.0-store.1, each found again by its source text.
  Not yet run.

```
cd bench
lldb -b -s llama-bench-oom-trace-v1.0.0.lldb -- ../build-debug/bin/llama-bench \
  -m <model.gguf> -fa on -r 3 -o md -p 0 -n 128 -d 65280 --store <depth.sqlite>
```

Leave out `--store` for the runs without the store. For CUDA graphs off, add
`settings set target.env-vars GGML_CUDA_DISABLE_GRAPHS=1` before `run` in the script.
After an upstream update, find each breakpoint's line again by its source text (the
comments in the script name it) and make a script for that version.

## The tools

All Swift 6. [`setup-swift.sh`](setup-swift.sh) checks for Swift and installs it if it
is missing (as you, through swiftly, signature checked against swift.org's keys).

[`bench.swift`](bench.swift) runs from the repository root, against a build of this
repository (`build/bin` by default, `--build <dir>` for another) or against an install
by the linux repository's setup (`llama-on`). Models are pinned by Hugging Face revision
and SHA-256. Reports, each with the llama.cpp version it ran, go to
`~/.local/share/llama-bench/`.

| Command | What it measures |
|---|---|
| `bench/bench.swift list` | The declared models and the suites |
| `bench/bench.swift fetch <model>` | Downloads a model, SHA-256 checked |
| `bench/bench.swift run <model> baseline` | llama-bench pp512 and tg128 at empty context |
| `bench/bench.swift run <model> depth` | The same at 0, 8k, 32k and 64k tokens of context |
| `bench/bench.swift run <model> offload` | Expert offload × threads, for models larger than VRAM |
| `bench/bench.swift run <model> reuse [--store <sqlite>]` | **Use 1**: real tasks through tools, through llama-server; no store, then cold, then warm |
| `bench/bench.swift run <model> state --store <sqlite>` | **Use 2**: run 1 (must run out of memory, or it halts), the fill, runs 2-4 |
| `bench/bench.swift perf [<url>[=<sizes>] ...]` | Running llama-server instances' speed through their endpoints: single requests at 64, 2k and 16k prompt tokens (`url=64,2048` for a server that needs smaller ones: an Intel iGPU, where a 16k prefill resets the GPU), two slots at once, two servers at once (from the server's own timings) |

The reuse task sets (`reuse-tasks-c++*.json`) ask about the C++ enforcement profile in
https://github.com/chainofreasoning/sound-design: clone it beside this repository, or
pass `--source <dir>`. The order of Test 1's runs is in
[`reuse-store-benchmark.md`](reuse-store-benchmark.md).

## The records

These are the v0.9.0 measurements as they were made, unchanged:
[`reuse-store-benchmark.md`](reuse-store-benchmark.md),
[`llama-bench-oom-gemma-4-31b.md`](llama-bench-oom-gemma-4-31b.md),
[`llama-bench-oom-reuse-sqlite.md`](llama-bench-oom-reuse-sqlite.md), with the patch
([`llama-bench-sqlite-store.patch`](llama-bench-sqlite-store.patch), now this branch's
first commit), the run script ([`llama-bench-oom-reuse-sqlite.sh`](llama-bench-oom-reuse-sqlite.sh),
now the `state` suite) and the trace script. They name the paths and the bench script
(`01-bench-llama-cpp.swift`) of the machine they ran on; `bench.swift` is that script.
The v1.0.0 measurements are next.

This repository is the source of truth for all of it; other copies follow it.
