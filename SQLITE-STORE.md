# llama.cpp with a SQLite reuse store

This branch (`sqlite-store`) is llama.cpp at an upstream release tag, plus a reuse
store: computed KV cache states kept in SQLite, so a later run restores them instead
of computing them again. Restoring copies the state into memory reserved at startup
and runs no kernel, so it also avoids out-of-memory errors that prefill can hit at
long contexts.

| Release of this branch | Upstream base |
|---|---|
| `sqlite-store` (current) | `v1.0.0` (`b1b770d`) |

## What it adds to upstream

- **`llama-bench --store <sqlite>`:** saves the depth state (`-d`) to the SQLite file
  after computing it, and restores it in later runs instead of computing it. Keyed by
  model file, depth, and K and V cache types. Without `--store`, llama-bench runs as
  upstream.
- **`vendor/sqlite3`:** SQLite 3.54.1 (amalgamation), compiled into the binaries that
  use it. No system SQLite is needed. `vendor/sqlite3/README.md` records its source
  and checksum.

Planned on this branch: the same store for `llama-server`.

## Get it

```
git clone --branch sqlite-store https://github.com/chainofreasoning/llama.cpp
cd llama.cpp
```

## Build

Requirements: CMake, a C/C++ compiler, and for the GPU build the CUDA toolkit
(`nvcc`). Nothing beyond what upstream needs.

### One set of backends per build

llama.cpp starts every backend compiled in. Starting CUDA wakes the NVIDIA GPU even
when every CUDA device is hidden (`CUDA_VISIBLE_DEVICES=-1`). So the CUDA and Vulkan
backends go in separate builds:

1. **CPU + CUDA (required):** the NVIDIA GPU.
2. **CPU + Vulkan (optional):** another GPU, such as an Intel iGPU through Mesa's
   Vulkan driver, without waking the NVIDIA GPU. Skip it if you only use the NVIDIA GPU.

### 1. CPU + CUDA

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability
(`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`): for example `120a-real`
for Blackwell (12.0), `89-real` for Ada (8.9).

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --parallel --target llama-server llama-cli llama-bench
```

### 2. CPU + Vulkan (optional)

Needs the Vulkan headers, `glslc` and the SPIR-V headers (Ubuntu: `libvulkan-dev
glslc spirv-headers`).

```
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON \
  -DGGML_CUDA=OFF -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-vulkan --parallel --target llama-server llama-cli llama-bench
```

### Options

- `LLAMA_BUILD_UI=ON` builds the web UI from the release's source (`tools/ui`, needs
  Node and npm) and embeds it in llama-server. `LLAMA_USE_PREBUILT_UI=OFF` keeps the
  build from downloading a prebuilt UI.
- `-DCMAKE_BUILD_TYPE=RelWithDebInfo` adds debug symbols to host code, for lldb; GPU
  code is unchanged.

## Containers

Two Containerfiles in `container/`, built with Podman, both on Ubuntu 26.04. Each
builds this checkout of the source: the image is the commit you build it from.

| Containerfile | Base images (pinned by digest) | Backends |
|---|---|---|
| `container/cuda.Containerfile` | NVIDIA CUDA 13.4.3 for Ubuntu 26.04, `nvcr.io/nvidia/cuda` (devel to build, runtime to run) | CPU + CUDA |
| `container/vulkan.Containerfile` (optional) | Canonical's Ubuntu 26.04, `public.ecr.aws/ubuntu/ubuntu` | CPU + Vulkan (Mesa) |

Each has two targets:

- `server` (the default): `llama-server` only, listening on port 8080.
- `full`: `llama-server`, `llama-cli` and `llama-bench`, with `--store`.

Models and stores are never built into an image. Mount them at `/models` and `/store`.
The images have no web UI (it needs npm to build) and no Python.

### Build

From the repository root:

```
podman build -f container/cuda.Containerfile   --target server -t llama-store-cuda:server .
podman build -f container/cuda.Containerfile   --target full   -t llama-store-cuda:full .
podman build -f container/vulkan.Containerfile --target server -t llama-store-vulkan:server .
```

Build arguments (`--build-arg NAME=value`):

- `CUDA_ARCHITECTURES` (CUDA only): `default` builds ggml's list for CUDA 13.4, Turing to
  Blackwell. One architecture is much faster to build: `120a-real` for Blackwell (12.0),
  `89-real` for Ada (8.9).
- `GGML_NATIVE`: `OFF` (default) runs on any x86-64 CPU; `ON` tunes CPU code for the
  machine that builds the image.

The build context keeps `.git` (see `.containerignore`), so `--version` reports the commit.

### Run

The CUDA image needs the host's NVIDIA driver and the NVIDIA Container Toolkit, with its
CDI specification in place (`nvidia-ctk cdi list` shows `nvidia.com/gpu=all`):

```
podman run --rm --device nvidia.com/gpu=all -p 127.0.0.1:8080:8080 \
  -v ~/models:/models:ro -v ~/llm-store:/store \
  llama-store-cuda:server -m /models/model.gguf -ngl all

podman run --rm --device nvidia.com/gpu=all -v ~/models:/models:ro -v ~/llm-store:/store \
  llama-store-cuda:full -m /models/model.gguf -fa on -p 0 -n 128 -d 65280 --store /store/depth.sqlite
```

The Vulkan image needs the GPU's render device:

```
podman run --rm --device /dev/dri -p 127.0.0.1:8080:8080 -v ~/models:/models:ro \
  llama-store-vulkan:server -m /models/model.gguf -ngl all
```

## Use the store

```
# first run: computes the depth and saves it
build/bin/llama-bench -m model.gguf -fa on -p 0 -n 128 -d 65280 -ub 256 --store depth.sqlite
# later runs: restore it instead of computing it
build/bin/llama-bench -m model.gguf -fa on -p 0 -n 128 -d 65280 --store depth.sqlite
```

The first run must fit: if the depth cannot be computed with the settings you want
(prefill runs out of GPU memory), compute it once with settings that need less memory,
such as a smaller micro-batch (`-ub 256`). The store key leaves the micro-batch out, so
later runs restore it under the settings that failed. A state is large (6.19 GB for
65,280 tokens of Gemma 4 31B with an f16 KV cache) and valid only for the same model
file, tokens, and cache types.

If generation then runs out of memory while building CUDA graphs, turn them off:
`GGML_CUDA_DISABLE_GRAPHS=1`.

## Measured

On llama.cpp v0.9.0, Gemma 4 31B at a 65,536-token context on a 24 GB GPU: without the
store, prefill runs out of GPU memory; with it (and CUDA graphs off), the run finishes
at 20.09 ± 0.27 t/s. Write-up, traces and method:
https://github.com/chainofreasoning/sound-design/blob/main/enforcement/languages/bechmarking/llama-bench-oom-reuse-sqlite.md
Being repeated on v1.0.0.

## Updating to a new upstream release

```
git remote add upstream https://github.com/ggml-org/llama.cpp.git   # once
git fetch upstream --tags
git rebase --onto <new-tag> <old-tag> sqlite-store
```

Then rebuild, rerun the measurements, update the table at the top, and push.
