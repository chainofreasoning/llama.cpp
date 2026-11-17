#pragma once

// The SQLite reuse store for llama-server: a second tier under the prompt cache.
//
// A state the prompt cache keeps in RAM is also written here, and stays when RAM
// evicts it or the server exits. When nothing in RAM matches a new prompt better,
// the cache looks here and restores the stored state instead of computing it.
// Restoring copies into the KV cache reserved at startup and runs no kernel, so a
// context that runs out of memory while it is computed can still be restored.
//
// A state is valid only for the same model file (its SHA-256), K and V cache
// types, flash attention setting and llama.cpp build: together, the key. The
// token sequence is stored with each state, so a prompt matches on a shared
// start. The micro-batch is not in the key: a state computed with settings that
// fit restores under settings that do not. Prompts with images are not stored.

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct sqlite3;
struct server_prompt;
struct server_prompt_data;
struct server_prompt_cache_state;
struct server_tokens;

struct server_state_store {
    // opens or creates the SQLite file; limit_mib 0: no size limit
    server_state_store(const std::string & path, int32_t limit_mib);
    ~server_state_store();

    bool ok() const { return db != nullptr; }

    // the key: the model file's SHA-256 (computed once per file, then kept in the
    // store by path, size and modification time), K and V types, flash attention, build
    void set_key(const std::string & model_path, ggml_type type_k, ggml_type type_v,
                 llama_flash_attn_type flash_attn, const std::string & build);

    // writes a prompt cache state; false when it is not stored (media, empty, error)
    bool save(const server_prompt_cache_state & state);

    struct candidate {
        int64_t id      = -1;
        int     n_tokens = 0;
        int     lcp      = 0;
        float   f_keep   = 0.0f;
        float   f_sim    = 0.0f;
    };

    // the stored state that keeps the most of the new prompt, by the prompt
    // cache's own rule; better than f_keep_base and f_sim_base, or id -1
    candidate find(const server_tokens & tokens_new, float f_keep_base, float f_sim_base);

    // reads a stored state: its tokens, checkpoints and KV data
    bool load(int64_t id, bool has_mtmd, server_prompt & prompt, server_prompt_data & data);

    std::string path;

private:
    sqlite3 *   db = nullptr;
    uint64_t    limit_bytes = 0;
    std::string key;

    bool exec(const char * sql);
    void evict(int64_t keep_id);
};
