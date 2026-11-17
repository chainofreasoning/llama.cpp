# llama.cpp with the SQLite reuse store: CPU + Vulkan, on Ubuntu 26.04 (optional build:
# a GPU through Vulkan, such as an Intel iGPU through Mesa).
# Build and run with Podman from the repository root: see SQLITE-STORE.md, "Containers".
#
# Base image: Canonical's Ubuntu 26.04 (public.ecr.aws/ubuntu), pinned by digest.

ARG UBUNTU_IMAGE=public.ecr.aws/ubuntu/ubuntu:26.04@sha256:1cec10ad04e9e13d1107ae55c475ea3e8305871bd644fe39d7a926e91eaa3642

## Build
FROM ${UBUNTU_IMAGE} AS build

# ON tunes CPU code for the build host; OFF (default) runs on any x86-64 CPU.
ARG GGML_NATIVE=OFF

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
       libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=$GGML_NATIVE \
       -DGGML_CUDA=OFF -DGGML_VULKAN=ON \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
       -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --parallel $(nproc) --target llama-server llama-cli llama-bench

## Runtime base
FROM ${UBUNTU_IMAGE} AS base

LABEL org.opencontainers.image.title="llama.cpp with a SQLite reuse store (Vulkan)" \
      org.opencontainers.image.source="https://github.com/chainofreasoning/llama.cpp"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libvulkan1 mesa-vulkan-drivers \
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
