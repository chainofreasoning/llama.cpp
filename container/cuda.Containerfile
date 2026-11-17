# llama.cpp with the SQLite reuse store: CPU + CUDA, on Ubuntu 26.04.
# Build and run with Podman from the repository root: see SQLITE-STORE.md, "Containers".
#
# Base images: NVIDIA's CUDA images for Ubuntu 26.04 (nvcr.io), pinned by digest.

ARG CUDA_BUILD_IMAGE=nvcr.io/nvidia/cuda:13.4.3-devel-ubuntu26.04@sha256:751ce45fa6934b382600b011a040a11d0ae0deb58040b43368098d859f6e156b
ARG CUDA_RUN_IMAGE=nvcr.io/nvidia/cuda:13.4.3-runtime-ubuntu26.04@sha256:83b873636b7ad03b418984f38f6b8a09e6f3c3a105d5cdab8088e998e8ed93f1

## Build
FROM ${CUDA_BUILD_IMAGE} AS build

# GPU architectures: "default" builds ggml's list for this CUDA version (Turing to
# Blackwell); one value, such as 120a-real or 89-real, builds only that and is faster.
ARG CUDA_ARCHITECTURES=default
# ON tunes CPU code for the build host; OFF (default) runs on any x86-64 CPU.
ARG GGML_NATIVE=OFF

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN if [ "$CUDA_ARCHITECTURES" != "default" ]; then ARCH="-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES"; fi \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=$GGML_NATIVE \
       -DGGML_CUDA=ON -DGGML_VULKAN=OFF $ARCH \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
       -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --parallel $(nproc) --target llama-server llama-cli llama-bench

## Runtime base
FROM ${CUDA_RUN_IMAGE} AS base

LABEL org.opencontainers.image.title="llama.cpp with a SQLite reuse store (CUDA)" \
      org.opencontainers.image.source="https://github.com/chainofreasoning/llama.cpp"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# models and stores are mounted here, never built into the image
VOLUME ["/models", "/store"]
WORKDIR /app

### Full: llama-server, llama-cli and llama-bench (with --store)
FROM base AS full
COPY --from=build /src/build/bin/llama-server /src/build/bin/llama-cli /src/build/bin/llama-bench /app/
ENV PATH=/app:$PATH
ENTRYPOINT ["/app/llama-bench"]

### Server: llama-server only (the default target)
FROM base AS server
COPY --from=build /src/build/bin/llama-server /app/
ENV LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080
EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
