# llama-bench out of memory: Gemma 4 31B at 64k

Run 2026-11-13 on the ThinkPad P16 Gen 3 (RTX PRO 5000 Blackwell Laptop GPU,
24 GB; CUDA reports 24,075 MiB usable). llama.cpp v0.9.0 (commit cd959bae).
Model `gemma-4-31B_q4_0-it.gguf`, flash attention on, everything on the GPU.

The original depth sweep (`-fa on -r 3 -d 0,8192,32768,65536`) completed
d0, 8k and 32k and failed at 64k. This file records what that failure is,
where it happens, and which settings remove it.

## Summary

| Question | Answer |
|---|---|
| Does llama-bench run out of memory at 64k? | Yes, 100% reproducible. |
| Where? | At startup (depth 65,536), or on the first batch after startup (depth 65,024). Never deep into the depth fill. |
| Why? | The startup reservation (model + KV cache + compute buffers) grows with the context size. At 64k it leaves 0 to 35 MiB free. The first matrix multiply needs scratch memory the startup check does not count. |
| Can the reuse store change this? | No. The memory is set by the context size at startup, not by what the work puts in the context. |
| What removes it? | Any of `-ub 256`, `-ctk q8_0 -ctv q8_0`, `-ngl 60`. `-ub 256` is the best: fastest generation, same precision, all on the GPU. |

## 1. The two errors, reproduced

### Depth 65,536 (context 66,048): fails at startup

```
llama-bench -m gemma-4-31B_q4_0-it.gguf -fa on -r 3 -o md -d 65536
```

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 620.03 MiB on device 0: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
llama_bench: error: failed to create context with model '.../gemma-4-31B_q4_0-it.gguf'
```

Exit 1. The 620 MiB of compute buffers do not fit after the model and the
KV cache.

### Depth 65,024 (context 65,536): fails on the first batch

```
llama-bench -m gemma-4-31B_q4_0-it.gguf -fa on -r 3 -o md -d 65024
```

```
CUDA error: out of memory
  current device: 0, in function alloc at ggml/src/ggml-cuda/ggml-cuda.cu:631
#7  ggml_cuda_pool_vmm::alloc(unsigned long, unsigned long*)
#8  launch_mul_mat_q<(ggml_type)2, 128, ...>(...)
```

Exit 134 (abort). The context is created, then the run aborts.

## 2. Where it fails: debug build under lldb

A copy of llama-bench was built from the same v0.9.0 source with the
install's options, changing only `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (debug
symbols for host code; no `-G`, so the GPU code is the same). It lives in
`~/.cache/llama.cpp/v0.9.0/build-debug`; `/opt/llama.cpp/current` is
untouched. It fails exactly like the installed build.

lldb sets a breakpoint at `ggml-cuda.cu:631`, the `cuMemCreate` call where
the CUDA memory pool asks the GPU for more memory. At each hit, a Python
callback logs:

- the request, the pool's size and use, and how much it grows;
- the free GPU memory as CUDA reports it (`cudaMemGetInfo`);
- the phase (llama-bench source line) and progress through it;
- the caller.

On SIGABRT it captures the full stack.

### Depth 65,024 under lldb

- The crash is at `llama-bench.cpp:2422`: the warmup run,
  `test_prompt(n_prompt=512)`. The depth run (line 2464) is never reached.
- The pool grew once, by 4 MiB. The next request asked for 6 MiB more and
  failed.
- GPU memory was at 24,073 of the 24,075 MiB CUDA can use.
- The run took about 4 seconds.

The earlier statement that this run "fails partway through the depth" was
wrong. It fails on the first batch of the warmup.

### The scratch request

Scratch memory is temporary GPU memory a kernel needs on top of its inputs
and outputs. It comes from the CUDA memory pool, not from the startup
reservation. The quantized matrix multiply (`mul_mat_q`, 4-bit weights × the
512-token batch) makes two scratch requests:

| Request | Source | Size | What it holds |
|---|---|---|---|
| Converted input | `mmq.cu:261` | 3.1 MiB | The batch's activations converted to 8-bit (`q8_1`) for integer math |
| Fixup buffer | `mmq.cuh:1517` | 5.1 MiB (1,343,488 floats) | Partial sums when the "stream-k" kernel splits one tile across several compute units; a second kernel adds them |

- Neither size depends on the context length. They depend on the batch
  (512), the layer's shape and the GPU's compute-unit count.
- The startup check (`sched_reserve`) does not count them, so a context can
  pass the check and still crash on its first multiply.
- After a multiply, the memory goes back to the pool, which keeps it. The
  pool grows once to its largest need; it does not keep growing as the
  context fills.

## 3. The measurement and the depth calculation

### Measurement at depth 32,768 (passes)

Startup memory, from llama-bench `-v`:

| Context | KV cache (scales with context) | KV cache (sliding window, fixed) | Compute buffers |
|---:|---:|---:|---:|
| 33,024 | 2,580 MiB | 1,200 MiB | 587.78 MiB |
| 33,280 | 2,600 MiB | 1,200 MiB | 588.03 MiB |

That is 20 MiB of KV cache plus 0.25 MiB of compute buffers per 256 tokens
of context. The model takes 16,818 MiB on the GPU. Free after startup at
33,280 is 2,584.7 MiB.

Pool growth (each line is one growth of the CUDA memory pool):

| Test | Phase (source line) | Request | Pool after | Free before | Caller |
|---|---|---:|---:|---:|---|
| pp512 | warmup prompt (2422), 0/512 | 3.11 MiB | 4 MiB | 2,584.7 MiB | `ggml_cuda_mul_mat_q` |
| pp512 | warmup prompt (2422), 0/512 | 5.12 MiB | 10 MiB | 2,552.7 MiB | `launch_mul_mat_q` |
| pp512 | warmup prompt (2422), 0/512 | 10.41 MiB | 12 MiB | 2,546.7 MiB | `launch_fattn<256,32,2>` |
| pp512 | warmup prompt (2422), 0/512 | 5.12 MiB | 18 MiB | 2,544.7 MiB | `launch_mul_mat_q` |
| tg128 | warmup generation (2434) | 0.01 MiB | 2 MiB | 2,574.7 MiB | `ggml_cuda_mul_mat_vec_q` |
| tg128 | measured generation (2504) | 2.58 MiB | 4 MiB | 2,572.7 MiB | `launch_fattn<512,1,8>` |

- **The pp512 test:** all growth happens on the first warmup batch. Nothing
  grows during the 32,768-token depth fill or the measured prompt.
- **The tg128 test:** this test reuses the depth computed by the pp512 test.
  llama-bench saves it to host memory and restores it (lines 2448–2475), so
  no depth is filled again.
- **GPU code:** free memory between the first two readings dropped by 32 MiB
  while the pool grew by only 4 MiB. About 28 MiB is GPU code loaded the
  first time the matrix-multiply kernels run, and it stays loaded.

### Prediction

Free after startup at context C:

    free(C) = 2,584.7 − (C − 33,280) / 256 × 20.25 MiB

The prediction for C = 65,536 was checked against the confirmation run's
log:

| | Predicted | Measured |
|---|---:|---:|
| KV cache, scales with context | 5,120 MiB | 5,120 MiB |
| Compute buffers | 619.53 MiB | 619.53 MiB |
| Free after startup | 33.2 MiB | 34.7 MiB |

- **Default tests (`-p 512 -n 128`):** the pp512 test runs first, with the
  larger context, and needs about 46 MiB on its first warmup batch. So any
  depth that fails, fails there.
- **Generation only (`-p 0 -n 128 -d 65280`, context 65,536):** the warmup is
  a single generated token. The prediction was that the warmup passes and
  the run fails when the depth fill starts.

### Confirmation: depth 65,280, generation only (fails partway)

```
llama-bench -m gemma-4-31B_q4_0-it.gguf -fa on -r 3 -o md -v -p 0 -n 128 -d 65280
```

| | Predicted | Measured |
|---|---|---|
| Free after startup | 33.2 MiB | 34.7 MiB |
| Warmup (generation) | passes | passed |
| Failure | start of the depth phase | depth phase, `llama-bench.cpp:2464`, first batch (0/65,280 tokens), the same 5.1 MiB `mul_mat_q` request at `ggml-cuda.cu:631` |

One estimate was off. The warmup was expected to use about 4 MiB; it used
30 MiB (2 MiB of pool and about 28 MiB of GPU code loaded on first use).
Free memory went from 34.7 to 4.7 MiB, which still covered it.

"Partway" means the run fails after the warmup succeeds, at the first batch
of the depth. No llama-bench run fails deep into the depth fill: nothing in
the pool grows while the depth fills.

## 4. The designs: do they remove the error?

Installed llama-bench, original sweep settings (`-fa on -r 3`), plus each
design's flag:

| Design | Depth | Result | Prompt (pp512), t/s | Generation (tg128), t/s |
|---|---:|---|---:|---:|
| `-ub 256` (micro-batch 256, default 512) | 65,536 | ran, exit 0 | 414.68 ± 6.76 | **19.71 ± 0.01** |
| `-ub 256` | 65,024 | ran, exit 0 | 423.98 ± 4.10 | **19.91 ± 0.02** |
| `-ctk q8_0 -ctv q8_0` (8-bit KV cache) | 65,536 | ran, exit 0 | 417.97 ± 6.98 | 15.46 ± 0.04 |
| `-ctk q8_0 -ctv q8_0` | 65,024 | ran, exit 0 | 420.83 ± 4.03 | 14.66 ± 1.30 |
| `-ngl 60` (output layer, 61st of 61, on the CPU) | 65,536 | ran, exit 0 | 418.47 ± 4.54 | 13.88 ± 3.66 |
| `-ngl 60` | 65,024 | ran, exit 0 | 424.05 ± 1.12 | 13.85 ± 3.74 |

- **All three designs remove both errors.**
- **`-ub 256` is the best.** It generates fastest and most steadily, with no
  change in precision and nothing moved off the GPU.
- **`-ctk q8_0 -ctv q8_0`** costs about 25% of generation speed and stores
  attention at lower precision.
- **`-ngl 60`** is the slowest and least steady, because the output layer
  runs on the CPU for every token.
- **Prompt speed** at 64k is about 415–424 t/s with all three. There is no
  64k baseline to compare against, because the default settings crash.

## Reproducing

- **Debug build:** `cmake -S ~/.cache/llama.cpp/v0.9.0/src -B
  ~/.cache/llama.cpp/v0.9.0/build-debug -G Ninja` with the install's options
  from `00-setup-llama-cpp.swift`, changing `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
  and adding `-DLLAMA_BUILD_UI=OFF`. Then `cmake --build ... --target
  llama-bench`.
- **lldb:** `lldb-21 -b -s measure.lldb -- <debug llama-bench> <args>`, where
  `measure.lldb` imports the callback, sets `breakpoint set --file
  ggml-cuda.cu --line 631`, attaches the callback, stops on SIGABRT, runs,
  and prints `bt 25`.
