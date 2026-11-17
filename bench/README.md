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
pass `--source <dir>`. How to run each test, in order: "Running the tests", below.

## Running the tests

The tests behind the records below, step by step, from a fresh clone. Run them in this
order. Every command runs from the root of this repository unless it says otherwise.
Each test prints where it wrote its report: `~/.local/share/llama-bench/`, one file per
model and test, named by time, model and test, with the llama.cpp version in it.

### 0. Set up, once

**Get both repositories, side by side.** The `reuse` tests ask questions about the C++
enforcement profile in sound-design and look for it beside this repository:

```
git clone --branch sqlite-store https://github.com/chainofreasoning/llama.cpp
git clone https://github.com/chainofreasoning/sound-design
cd llama.cpp
```

**Install what the tests use.** Swift 6, through the script here; the rest from your
distribution (Ubuntu 26.04 package names shown):

```
bench/setup-swift.sh                          # Swift 6, as you, if it is missing
sudo apt install curl git sqlite3 lldb        # sqlite3: the reuse store; lldb: test 6
```

**Build with CUDA**, as in `SQLITE-STORE.md`, "Build", section 1. Set
`CMAKE_CUDA_ARCHITECTURES` from `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
(12.0 → `120a-real`, 8.9 → `89-real`). Then check that its llama-bench has the store:

```
build/bin/llama-bench --help 2>&1 | grep -- --store
```

The tests use `build/bin` unless the linux repository's setup is installed
(`/opt/llama.cpp/current/bin/llama-on` exists); then they use that install, and
`--build build/bin` makes them use this build instead.

**Fetch the models.** Each is pinned by Hugging Face revision and SHA-256 and goes to
`~/.local/share/llama-models/`. The three the tests need:

```
bench/bench.swift fetch gpt-oss-20b           # 12.1 GB: test 1
bench/bench.swift fetch qwen3.8-27b           # 17.5 GB: tests 3, 4
bench/bench.swift fetch gemma-4-31b           # 18.9 GB: tests 4, 5, 6
```

`bench/bench.swift list` shows every declared model, its size and whether it is present.
Test 2 measures every model that is present; `bench/bench.swift fetch all` gets them all
(about 200 GB).

**Give the GPU to the tests.** The tests (all but test 1) start their own llama-server or
llama-bench and refuse while another process holds the NVIDIA GPU. Stop any llama-server
you run yourself before them.

### 1. `perf`: a running server's speed

Start a server in one terminal (gpt-oss-20b, two slots of 64k tokens, as measured here):

```
build/bin/llama-server -m ~/.local/share/llama-models/gpt-oss-20b-MXFP4.gguf \
  --n-gpu-layers all --flash-attn on --ctx-size 131072 --parallel 2 --port 8080
```

In another, once it answers on `http://127.0.0.1:8080/health`:

```
bench/bench.swift perf http://127.0.0.1:8080
```

It sends single requests of about 64, 2k and 16k prompt tokens, then two at once (the
two slots). Give it several URLs to measure several servers and two at once across them.
A server on an Intel iGPU must stop at 2k: `http://127.0.0.1:8081=64,2048`. A longer
prefill outlasts the i915 driver's preempt timeout and the GPU is reset. Stop the server
(Ctrl-C) before test 2.

### 2. `baseline` and `depth`: llama-bench speed

```
bench/bench.swift run all baseline            # pp512 and tg128 at empty context
bench/bench.swift run all depth               # the same at 0, 8k, 32k and 64k tokens of context
```

`all` means every model present; a model name runs one. Models larger than the GPU are
run with expert weights in system RAM. On another backend: `--build <dir>` with another
build of this repository, for example `--build build-vulkan/bin`.

### 3. `reuse`: use 1, the first run

Thirteen questions about the C++ profile, answered by Qwen3.8-27B through tools, three
times: no store, a new store (cold: it fills), the same store again (warm):

```
bench/bench.swift run qwen3.8-27b reuse
bench/bench.swift run qwen3.8-27b reuse --store ~/.local/share/llm-reuse/c++.sqlite
bench/bench.swift run qwen3.8-27b reuse --store ~/.local/share/llm-reuse/c++.sqlite
```

The store file must not exist before the second command. Each report gives, per task,
the answer and whether it is correct, the peak context, the prompt tokens sent and
computed, the time, and what came from the store.

### 4. `reuse`: Test 1, use 1 against a full context

Six runs, in this order. Each depends on the one before:

1. **Calibration** decides which tasks count: a task qualifies when it is answered
   correctly and its peak context exceeds 32,768 tokens in every pass (the report's
   "Peak > 32,768" column). A task that does not qualify is replaced in
   `bench/reuse-tasks-c++-c.json` and run 1 repeated.
2. **The store** is built for the final tasks, then copied twice (`sqlite3 .backup`), so
   runs 5 and 6 each start from the same store.
3. and 4. put the failure and the baseline on record before the test.
5. and 6. are the test, on Gemma 4 31B and on Qwen3.8-27B.

```
# 1. Calibration
bench/bench.swift run qwen3.8-27b reuse --tasks bench/reuse-tasks-c++-c.json --role measured --ctx 65536 --repeat 3

# 2. Build the store, then copy it for runs 5 and 6
bench/bench.swift run qwen3.8-27b reuse --store ~/.local/share/llm-reuse/c++-oom.sqlite
bench/bench.swift run qwen3.8-27b reuse --tasks bench/reuse-tasks-c++-b.json --store ~/.local/share/llm-reuse/c++-oom.sqlite
bench/bench.swift run qwen3.8-27b reuse --tasks bench/reuse-tasks-c++-c.json --role build --store ~/.local/share/llm-reuse/c++-oom.sqlite
sqlite3 ~/.local/share/llm-reuse/c++-oom.sqlite ".backup $HOME/.local/share/llm-reuse/c++-oom-gemma.sqlite"
sqlite3 ~/.local/share/llm-reuse/c++-oom.sqlite ".backup $HOME/.local/share/llm-reuse/c++-oom-qwen.sqlite"

# 3. The original failure: Gemma at a 65,536-token context, no store
bench/bench.swift run gemma-4-31b reuse --tasks bench/reuse-tasks-c++-c.json --role measured --ctx 65536

# 4. Baseline at the default context (32,768), no store
bench/bench.swift run gemma-4-31b reuse --tasks bench/reuse-tasks-c++-c.json --role measured --repeat 3

# 5. The test: Gemma with the store
bench/bench.swift run gemma-4-31b reuse --tasks bench/reuse-tasks-c++-c.json --role measured --repeat 3 --store ~/.local/share/llm-reuse/c++-oom-gemma.sqlite

# 6. Qwen with the store
bench/bench.swift run qwen3.8-27b reuse --tasks bench/reuse-tasks-c++-c.json --role measured --repeat 3 --store ~/.local/share/llm-reuse/c++-oom-qwen.sqlite
```

[`reuse-store-benchmark.md`](reuse-store-benchmark.md) has the full rationale and the
v0.9.0 results.

### 5. `state`: use 2, the KV-cache store against an out-of-memory error

```
bench/bench.swift run gemma-4-31b state --store ~/.local/share/llm-reuse/depth.sqlite
```

Five llama-bench runs of Gemma 4 31B at a 65,280-token depth (a 65,536-token context):

| Run | Store | CUDA graphs | Expected on a 24 GB GPU |
|---|---|---|---|
| run 1 | no | on | out of memory, computing the depth |
| fill | saves it (micro-batch 256, which fits) | on | finishes |
| run 2 | restores it | on | out of memory, building a CUDA graph |
| run 3 | no | off | out of memory, computing the depth (the control) |
| run 4 | restores it | off | finishes |

The store file must not exist before; it grows to about 6.2 GB. The error it tests
depends on the GPU's memory: if run 1 does not run out of memory, the test stops there
and says so. On a card with a different amount of memory, `--depth <tokens>` sets another
depth. The report gives each run's result, its command, and the store's own lines;
each run's full output is in a `.log` beside it.

### 6. The trace: where the memory goes

Build llama-bench again with debug symbols for host code (GPU code is unchanged), using
your `CMAKE_CUDA_ARCHITECTURES`:

```
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-debug --parallel --target llama-bench
```

Then run test 5's four configurations under lldb, from `bench/`, where the script is,
using test 5's store:

```
cd bench
lldb -b -s llama-bench-oom-trace-v1.0.0.lldb -- ../build-debug/bin/llama-bench \
  -m ~/.local/share/llama-models/gemma-4-31B_q4_0-it.gguf -fa on -r 3 -o md -p 0 -n 128 -d 65280 \
  --store ~/.local/share/llm-reuse/depth.sqlite
```

| Run | Store | CUDA graphs | Change |
|---|---|---|---|
| 1 | no | on | leave out the `--store` line |
| 2 | yes | on | none |
| 3 | no | off | leave out `--store`, and add `settings set target.env-vars GGML_CUDA_DISABLE_GRAPHS=1` before `run` in the script |
| 4 | yes | off | add the same setting |

At every breakpoint lldb prints where it is and the free GPU memory in MiB. Compare the
sequence with the step tables in
[`llama-bench-oom-reuse-sqlite.md`](llama-bench-oom-reuse-sqlite.md), "Results".

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
