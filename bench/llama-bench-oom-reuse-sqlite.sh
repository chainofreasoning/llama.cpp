#!/bin/zsh
# Reuse through SQLite against llama-bench's partway out-of-memory error.
# Gemma 4 31B, llama-bench -fa on -r 3 -p 0 -n 128 -d 65280 (context 65,536 tokens).
# Needs the patched llama-bench (llama-bench-sqlite-store.patch on llama.cpp v0.9.0).
# Each run prints its result rows, its out-of-memory lines and its store lines.
set -u
B=${LLAMA_BENCH_STORE:-$HOME/.cache/llama.cpp/v0.9.0/build-store/bin/llama-bench}
M=$HOME/.local/share/llama-models/gemma-4-31B_q4_0-it.gguf
S=$HOME/.local/share/llm-reuse/llama-bench-depth.sqlite
A=(-m $M -fa on -r 3 -o md -p 0 -n 128 -d 65280)

show() { echo "$1" | grep -E "^\| (model|gemma)|out of memory|current device|cuMemCreate|cudaGraphInstantiate|store:"; }

echo "== run 1: no store, CUDA graphs on $(date -u +%FT%TZ)"
out=$($B $A 2>&1); rc=$?; show "$out"; echo "   exit $rc"
if ! echo "$out" | grep -q "out of memory"; then echo "== HALT: run 1 did not reproduce the out-of-memory error"; exit 1; fi

echo "== store fill: run 1 plus -ub 256 --store $(date -u +%FT%TZ)"
out=$($B $A -ub 256 --store $S 2>&1); rc=$?; show "$out"; echo "   exit $rc"

echo "== run 2: store, CUDA graphs on $(date -u +%FT%TZ)"
out=$($B $A --store $S 2>&1); rc=$?; show "$out"; echo "   exit $rc"

echo "== run 3: no store, CUDA graphs off $(date -u +%FT%TZ)"
out=$(GGML_CUDA_DISABLE_GRAPHS=1 $B $A 2>&1); rc=$?; show "$out"; echo "   exit $rc"

echo "== run 4: store, CUDA graphs off $(date -u +%FT%TZ)"
out=$(GGML_CUDA_DISABLE_GRAPHS=1 $B $A --store $S 2>&1); rc=$?; show "$out"; echo "   exit $rc"

echo "== done $(date -u +%FT%TZ)"
