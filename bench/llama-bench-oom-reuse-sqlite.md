# Reuse through SQLite against llama-bench's partway out-of-memory error

Reuse through SQLite removes the out-of-memory error that stops llama-bench partway through a run of Gemma 4 31B at a 65,536-token context. The reuse required a change to llama-bench's source: a `--store` option that saves the computed depth to SQLite and restores it in later runs instead of computing it again. A second out-of-memory error then appears during generation, from building CUDA graphs. Turning CUDA graphs off removes that one. With both changes, the run that failed finishes at 20.09 ± 0.27 tokens per second. The control run, with graphs off and no store, still fails at the first error. That rules out the graph switch as the fix for the first error.

All runs took place on 13 November 2026, between 17:21Z and 17:31Z.

This write-up covers the second of the store's two uses: out-of-memory mitigation. The first use, speed and token reduction, stores text the model derived rather than the KV cache it computed. [Two uses of the SQLite store](reuse-store-benchmark.md#two-uses-of-the-sqlite-store), in `reuse-store-benchmark.md`, compares the two.

## Scope and audience

This write-up is for readers who run local language models on one GPU and need to know what fails when a long context fills the card, and what removes the failure. It assumes familiarity with llama.cpp and its `llama-bench` tool.

Earlier work reached these results in stages. That record is in [`llama-bench-oom-gemma-4-31b.md`](llama-bench-oom-gemma-4-31b.md): the first reproduction of the error, the debug build, and the settings (`-ub 256`, `-ctk q8_0 -ctv q8_0`, `-ngl 60`) that avoid it by reserving less memory. This write-up stands on its own and supersedes those stages as the result.

## Terms

The terms in this list keep one meaning throughout this write-up.

- **Context:** the number of tokens the model can hold at once. llama.cpp reserves GPU memory for the full context when it starts. In llama-bench, the context equals the depth plus the generated tokens, rounded up to a multiple of 256: here 65,280 + 128 → 65,536 tokens.
- **Depth:** the tokens llama-bench feeds the model before it measures, to simulate a long conversation. Here, 65,280 tokens.
- **KV cache (key–value cache):** the GPU memory that holds what the model computed for each token in the context. Storing it lets the model skip recomputing those tokens.
- **Depth state:** the KV cache after the depth is processed, as saved by `llama_state_seq_get_data`. For this model and depth it is 6,187,395,544 bytes.
- **Scratch pool:** the CUDA memory pool (`ggml_cuda_pool_vmm`) that hands out temporary GPU memory to kernels while they run. It grows on demand through `cuMemCreate` and is not part of the memory reserved at startup.
- **CUDA graph:** a recording of a step's GPU work that CUDA replays for each generated token. llama.cpp records one for the generation step, split into chunks of 512 operations (`GGML_CUDA_GRAPH_CHUNK_SIZE`, default 512).
- **Store:** the SQLite file that holds the depth state between runs, at `~/.local/share/llm-reuse/llama-bench-depth.sqlite` (6.2 GB).
- **Error 1:** the out-of-memory error while llama-bench computes the depth: the scratch pool cannot grow (`cuMemCreate`, `ggml-cuda.cu:631`).
- **Error 2:** the out-of-memory error while llama-bench builds a CUDA graph for generation (`cudaGraphInstantiate`, `ggml-cuda.cu:3090`).
- **MiB:** mebibyte, 2²⁰ bytes. **t/s:** tokens per second.

## Why removing error 1 needs a change to llama-bench's source

Error 1 happens while llama-bench computes the depth. The run as specified (this model, a 65,536-token context, micro-batch 512, a 16-bit KV cache, all layers on the GPU) leaves too little free memory for the first batch of that computation. Removing error 1 without changing those settings takes one thing: putting the depth into the KV cache without computing it. Unmodified llama-bench cannot do that, for three reasons:

1. **The depth is always computed in a new process.** llama-bench keeps a depth state (`ctx_state cstate`, `llama-bench.cpp:2162` and 2317 in the unpatched source), but only in RAM, for the life of one process. It fills that state by computing the depth (lines 2459–2475). The first test of every run must compute the depth, and that is where error 1 occurs.
2. **No option loads a saved depth.** `llama-bench --help` lists none: its cache options (`-ctk`, `-ctv`, `-nkvo`) set the KV cache's type and location, not its contents.
3. **No other switch reaches error 1.** Run 3 turns CUDA graphs off and still fails at error 1, because the failing request comes from the matrix multiply that computes the depth, before any CUDA graph exists.

The patch, described in [The change to llama-bench](#the-change-to-llama-bench), adds the missing step: it saves the depth state to SQLite once, and later runs restore it instead of computing it.

Error 1 also disappears with other settings, which change the run instead of keeping it. [`llama-bench-oom-gemma-4-31b.md`](llama-bench-oom-gemma-4-31b.md) measured three of them. Each reserves less memory at startup and costs something in return:

| Setting | What it changes | Measured cost |
|---|---|---|
| `-ub 256` | half the micro-batch | none measured in generation (19.71–19.91 t/s) |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache instead of 16-bit | generation about 25% slower (14.66–15.46 t/s); lower precision of stored attention |
| `-ngl 60` | output layer on the CPU | generation 13.85–13.88 t/s, with a spread of ±3.7 t/s |

One further setting, `-nkvo 1` (KV cache in system RAM), was not tested. These settings answer a different question from the one this write-up tests: they make the run fit, and none of them reuses work. Under the run as specified, restoring the depth from the store is the only tested way to remove error 1.

## The change to llama-bench

[`llama-bench-sqlite-store.patch`](llama-bench-sqlite-store.patch) changes two files in llama.cpp v0.9.0: `tools/llama-bench/llama-bench.cpp` (101 added lines) and `tools/llama-bench/CMakeLists.txt` (one changed line). It adds one option, `--store <sqlite>`. Without the option, llama-bench runs exactly as before: run 1 fails the same way as the unpatched build.

### What the patch adds

The patch makes six changes, listed in source order:

1. **Header.** `#include <sqlite3.h>` and `#include <memory>`.
2. **Option.** A `store` field in `cmd_params`, the `--store <sqlite>` argument, and its line in `--help`.
3. **Store type.** A `depth_store` struct that opens the SQLite file and creates one table:

   ```sql
   CREATE TABLE IF NOT EXISTS depth_state (key TEXT NOT NULL, chunk INTEGER NOT NULL, data BLOB NOT NULL,
                                           PRIMARY KEY (key, chunk))
   ```

   `load(key, buf)` reads every chunk of a key in order and joins them. `save(key, buf)` deletes the key's old chunks and writes the new state in 256 MiB chunks inside one transaction. Chunks are needed because SQLite limits a single value to 1 GB, and the depth state here is 6.19 GB.
4. **Store opened once.** When `--store` is given, `llama_bench()` opens the store next to the existing `cstate`, before the first test.
5. **Restore before computing.** Before the depth computation, llama-bench builds a key and looks it up:

   ```cpp
   const std::string store_key = t.model_filename + "|depth=" + std::to_string(t.n_depth) +
       "|type_k=" + ggml_type_name(t.type_k) + "|type_v=" + ggml_type_name(t.type_v);

   if (!is_cached && store) {
       std::vector<uint8_t> buf;
       if (store->load(store_key, buf) && llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0) != 0) {
           // "restored ... not recomputed"
           cstate.depth = t.n_depth;
           cstate.buf   = std::move(buf);
           is_cached    = true;
       }
   }
   ```

   Setting `is_cached` makes llama-bench skip `test_prompt(ctx, t.n_depth, ...)`, the call that fails in error 1. The key holds what determines the state's contents and layout: the model file, the depth, and the K and V cache types.
6. **Save after computing.** When llama-bench does compute the depth, it already copies the state into `cstate.buf`. The patch then writes that buffer to the store with `store->save(store_key, cstate.buf)`.

`CMakeLists.txt` adds `sqlite3` to the libraries that `llama-bench-impl` links.

### Why a restore cannot hit error 1

`llama_state_seq_set_data` reads the saved state on the host (`llama_io_read_host`, `llama-context.cpp:3198`). It then writes each tensor slice into the KV cache with `ggml_backend_tensor_set_n` (`llama-context.cpp:3223`), which copies host memory to GPU memory. The copy runs no kernel and takes no scratch memory, and the KV cache it writes into was reserved at startup. The trace of runs 2 and 4 confirms it: free GPU memory is 4.7 MiB before and after the restore, with no scratch pool growth in between.

### What the patch does not change

- **The measurement:** prompt and generation timing, repetitions, output formats, and the depth tokens are as before.
- **The existing in-process cache:** `cstate` still serves later repetitions within one run, now filled from the store when the store holds the depth.
- **Any other llama.cpp code:** the patch touches no file outside `tools/llama-bench/`.

## Results

### The four runs

The four runs share one command, `llama-bench -m gemma-4-31B_q4_0-it.gguf -fa on -r 3 -o md -p 0 -n 128 -d 65280`, at a 65,536-token context. They differ only in the store and the CUDA graph switch. The table gives each run's outcome.

| Run | Store | CUDA graphs | Outcome | Where it stopped |
|---:|---|---|---|---|
| 1 | no | on | out-of-memory error, exit 134 | computing the depth: `ggml-cuda.cu:631` (`cuMemCreate`) |
| 2 | yes | on | out-of-memory error, exit 134 | generation: `ggml-cuda.cu:3090` (`cudaGraphInstantiate`) |
| 3 | no | off | out-of-memory error, exit 134 | computing the depth: `ggml-cuda.cu:631` (`cuMemCreate`) |
| 4 | yes | off | finished, exit 0 | — tg128 @ d65280 = **20.09 ± 0.27 t/s** |

Run times (UTC): run 1 at 17:21:19Z; run 2 at 17:23:42Z, and again under lldb, ending at 17:27:16Z; run 4 under lldb, ending at 17:29:43Z; run 3 under lldb, ending at 17:30:39Z.

### Error 1: the scratch pool while computing the depth

Run 1 reproduces llama-bench's partway error. The warmup (one generated token) succeeds. The run then fails on the first batch of the depth, before it has processed any of the 65,280 tokens:

```
CUDA error: out of memory
  current device: 0, in function alloc at ggml/src/ggml-cuda/ggml-cuda.cu:631
  cuMemCreate(&handle, reserve_size, &prop, 0)
```

The trace of run 3, which fails the same way, shows the free GPU memory at each step. The table lists those steps in order.

| Step (run 3) | Source | Free GPU memory |
|---|---|---:|
| Startup complete, before the warmup | `llama-bench.cpp:2513` | 36.7 MiB |
| Warmup: the scratch pool grows 0 → 2 MiB | `ggml-cuda.cu:631`, in the warmup | 34.7 MiB |
| Depth, first batch: the scratch pool grows 2 → 4 MiB (request 3.11 MiB, the batch converted to 8-bit) | `ggml-cuda.cu:631`, in the depth computation (`llama-bench.cpp:2560`) | 4.7 MiB |
| Depth, first batch: the scratch pool asks for 6 MiB more (request 5.12 MiB, the matrix multiply's fixup buffer) | `ggml-cuda.cu:631`, in the depth computation | 2.7 MiB → out-of-memory error |

The warmup used 30 MiB beyond its 2 MiB of scratch pool, which took free memory from 34.7 to 4.7 MiB. The trace measures that drop but not what fills it. Kernel code that CUDA loads on first use is the likely cause, and remains unconfirmed.

Reuse removes error 1. In runs 2 and 4, the patched llama-bench reads the depth state from the store and copies it into the KV cache instead of computing it:

```
llama-bench: store: depth 65280 restored from ~/.local/share/llm-reuse/llama-bench-depth.sqlite (6187395544 bytes), not recomputed
```

Free GPU memory stays at 4.7 MiB across the restore. The restore runs no kernels, so the 3.11 MiB and 5.12 MiB scratch requests never happen.

### Error 2: building a CUDA graph during generation

Run 2 gets past the depth and then fails in generation:

```
CUDA error: out of memory
  current device: 0, in function ggml_cuda_graph_launch_chunk at ggml/src/ggml-cuda/ggml-cuda.cu:3090
  cudaGraphInstantiate(&graph->instances[idx], cuda_graph, nullptr, nullptr, 0)
```

The trace of run 2 shows where. The table lists its steps after the restore.

| Step (run 2) | Source | Free GPU memory |
|---|---|---:|
| Depth state restored from the store | `llama-bench.cpp:2545` | 4.7 MiB |
| Measured generation: the scratch pool grows 2 → 4 MiB (request 2.58 MiB, flash attention) | `ggml-cuda.cu:631`, in measured generation (`llama-bench.cpp:2605`) | 4.7 MiB |
| CUDA graph chunks 0, 1, and 2 instantiated | `ggml-cuda.cu:3090` | 2.7 MiB, unchanged |
| CUDA graph chunk 3: `cudaGraphInstantiate` | `ggml-cuda.cu:3090` | 2.7 MiB → out-of-memory error |

Generation is the first phase to use CUDA graphs. The warmup token runs without one. Chunks 0 to 2 took no measurable memory at 0.1 MiB resolution. Chunk 3 needed more than the 2.7 MiB left. The trace does not show what chunk 3 allocates.

Turning CUDA graphs off removes error 2. Run 4 sets `GGML_CUDA_DISABLE_GRAPHS=1`: the depth comes from the store, generation runs without graphs at 2.7 MiB free, and all three repetitions of 128 tokens finish.

### The control

Run 3 turns graphs off without the store, and fails at error 1 in the same place and the same way as run 1. Error 1 happens before generation and before any CUDA graph exists, so the graph switch cannot affect it. Only the store removes it.

### Speed

Run 4 generated at 20.09 ± 0.27 t/s. The store fill (described in the Method section) generated at 19.84 ± 0.06 t/s with graphs on. That fill used `-ub 256` and computed its depth, so the two figures do not isolate the cost of turning graphs off. The difference of 0.25 t/s is inside run 4's spread.

## Why both errors exist

Both errors come from what the startup reservation leaves free. At a 65,536-token context, llama-bench reserves 16,818 MiB for the model, 6,320 MiB for the KV cache (5,120 MiB that scales with the context plus 1,200 MiB of fixed sliding-window cache), and 619.53 MiB for compute buffers. That leaves 36.7 MiB free.

Memory after startup comes from three sources: the scratch pool, kernel code, and CUDA graphs. The startup check counts none of them. Free memory falls from 36.7 MiB to 2.7 MiB before either failing request. Neither error depends on how full the context is: both fail on the first request of their kind.

## Method

### Software and hardware

- **Host:** ThinkPad P16 Gen 3, Core Ultra 9 275HX, 128 GB of DDR5 memory, Ubuntu 26.04, kernel 7.0.0-43-generic.
- **GPU:** NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, 24,463 MiB (CUDA reports 24,075 MiB usable), driver 615.71.09.
- **CUDA:** 13.4 (`nvcc` V13.4.92).
- **llama.cpp:** v0.9.0, commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`.
- **Model:** `gemma-4-31B_q4_0-it.gguf` (google/gemma-4-31B-it-qat-q4_0-gguf at revision 59dde24573e7), all 61 layers on the GPU.
- **SQLite:** 3.46.1 (`libsqlite3-dev` 3.46.1-9ubuntu0.3).
- **lldb:** 21.1.8.

### Building

The patched build lives in its own tree. The installed llama.cpp in `/opt/llama.cpp/current` is untouched. Replace `<repo>` with the directory that holds this write-up.

```
cp -a ~/.cache/llama.cpp/v0.9.0/src ~/.cache/llama.cpp/v0.9.0/src-store
cd ~/.cache/llama.cpp/v0.9.0/src-store && patch -p1 < <repo>/llama-bench-sqlite-store.patch
cmake -S ~/.cache/llama.cpp/v0.9.0/src-store -B ~/.cache/llama.cpp/v0.9.0/build-store -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build ~/.cache/llama.cpp/v0.9.0/build-store --target llama-bench
```

The options match the installed build (`00-setup-llama-cpp.swift`), except for two. `RelWithDebInfo` adds debug symbols to host code for lldb. It leaves out CUDA's `-G` flag, so the GPU code is the same as the release build's. `LLAMA_BUILD_UI=OFF` skips the web interface, which llama-bench does not use. Applied with `patch -p1` to a clean v0.9.0 tree, the patch reproduces the tested source byte for byte.

### Running

[`llama-bench-oom-reuse-sqlite.sh`](llama-bench-oom-reuse-sqlite.sh) runs the sequence: run 1, the store fill, run 2, run 3, and run 4. It halts if run 1 does not reproduce the out-of-memory error.

The store fill is run 1's command plus `-ub 256 --store <file>`. At micro-batch 512 the depth cannot be computed at this context: that is error 1. Micro-batch 256 reserves less memory, so the depth fits. The fill took from 17:21:23Z to 17:23:42Z and generated at 19.84 ± 0.06 t/s.

### Tracing

[`llama-bench-oom-trace.lldb`](llama-bench-oom-trace.lldb) produced the step tables. It uses plain lldb commands only. Run it from the directory that holds it:

```
lldb-21 -b -s llama-bench-oom-trace.lldb -- ~/.cache/llama.cpp/v0.9.0/build-store/bin/llama-bench \
  -m ~/.local/share/llama-models/gemma-4-31B_q4_0-it.gguf -fa on -r 3 -o md -p 0 -n 128 -d 65280 \
  --store ~/.local/share/llm-reuse/llama-bench-depth.sqlite
```

For runs 3 and 4, add `settings set target.env-vars GGML_CUDA_DISABLE_GRAPHS=1` before `run` in the lldb script; for runs 1 and 3, leave out `--store`.

The script sets breakpoints at:

- **the phases:** warmup at `llama-bench.cpp:2513`, the restore at line 2545, the depth computation at line 2560, and measured generation at line 2605;
- **scratch pool growth:** every `cuMemCreate` at `ggml-cuda.cu:631`, printing the request (`size`), the growth (`reserve_size`), and the pool's size and use before it;
- **CUDA graphs:** every instantiation at lines 3090, 3064, and 3092, printing the chunk index (`idx`).

Each breakpoint prints its location (`frame info`), then evaluates `cudaMemGetInfo` inside the stopped process with `expr` and prints the free memory CUDA reports, in MiB. It then continues on its own (`-G true`). The phase of a scratch pool or graph event is the phase breakpoint printed before it. Earlier measurements tracked the scratch pool only. They could not see the memory that kernel code and CUDA graphs take, so they missed error 2.

The first traces used lldb callbacks written in Python. They were replaced by this plain lldb script on 2026-11-13, and run 2 was traced again with it. It recorded the same values: free memory of 36.69, 34.69, 4.69, 4.69, and 2.69 MiB; the 2,707,968-byte scratch request; graph chunks 0 to 3; and the out-of-memory error at chunk 3.

## Limitations

- **One run per configuration.** Each run repeats its measurement three times (`-r 3`), which gives the ± spread. Each configuration ran once, apart from run 2, which ran twice with the same result. Free memory after startup read 34.7 MiB and 36.7 MiB in different runs at the same context. The trace reads it at different points: after and before the first 2 MiB of scratch pool.
- **The store fill computed the depth at micro-batch 256, not 512.** Both runs draw the depth tokens from the same unseeded `std::rand()` sequence (`llama-bench.cpp:2181` and 2183), so the tokens match. Batching changes the order of floating-point sums, so the stored KV values can differ in the last bits from a micro-batch-512 computation. The stored state's contents were not compared.
- **llama-bench measures speed, not answers.** Run 4 shows that the run finishes and how fast it generates. It does not show that the restored state gives the same outputs as a computed one.
- **The cause of chunk 3's memory need is unidentified.** The trace locates the failing call but not what that call allocates.
- **One model, one GPU, one context.** The results hold for Gemma 4 31B (q4_0) at a 65,536-token context on a 24 GB card. Other models and sizes have different startup reservations and different headroom.
