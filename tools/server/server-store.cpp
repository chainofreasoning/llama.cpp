#include "server-store.h"

#include "server-common.h"
#include "hash/hash.h"
#include "server-task.h"

#include "common.h"

#include <sqlite3.h>

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

// A single SQLite value is limited to 1 GB; a state can be several GB.
static constexpr size_t STORE_CHUNK = size_t(256) << 20;

static const char * STORE_SCHEMA =
    "PRAGMA journal_mode = WAL;"
    "CREATE TABLE IF NOT EXISTS model_hash ("
    "  path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS state ("
    "  id INTEGER PRIMARY KEY, key TEXT NOT NULL, n_tokens INTEGER NOT NULL, tokens_sha256 TEXT NOT NULL,"
    "  tokens BLOB NOT NULL, bytes INTEGER NOT NULL, created INTEGER NOT NULL, last_used INTEGER NOT NULL,"
    "  UNIQUE (key, tokens_sha256));"
    "CREATE TABLE IF NOT EXISTS state_checkpoint ("
    "  state_id INTEGER NOT NULL, idx INTEGER NOT NULL, n_tokens INTEGER NOT NULL,"
    "  pos_min INTEGER NOT NULL, pos_max INTEGER NOT NULL, PRIMARY KEY (state_id, idx));"
    // part: main, drft, or ckpt_tgt / ckpt_dft / ckpt_spec with the checkpoint's idx
    "CREATE TABLE IF NOT EXISTS state_chunk ("
    "  state_id INTEGER NOT NULL, part TEXT NOT NULL, idx INTEGER NOT NULL, chunk INTEGER NOT NULL,"
    "  data BLOB NOT NULL, PRIMARY KEY (state_id, part, idx, chunk));";

static int64_t now_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string sha256_hex(const unsigned char * digest) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        out += hex[digest[i] >> 4];
        out += hex[digest[i] & 0xf];
    }
    return out;
}

static std::string sha256_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "";
    }
    sha256_t ctx;
    sha256_init(&ctx);
    std::vector<char> buf(size_t(64) << 20);
    while (in) {
        in.read(buf.data(), buf.size());
        const std::streamsize n = in.gcount();
        if (n > 0) {
            sha256_update(&ctx, (const unsigned char *) buf.data(), (size_t) n);
        }
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    return sha256_hex(digest);
}

// a prepared statement, finalized when it goes out of scope
struct store_stmt {
    sqlite3_stmt * st = nullptr;
    store_stmt(sqlite3 * db, const char * sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
            SRV_ERR("store: %s: %s\n", sql, sqlite3_errmsg(db));
            st = nullptr;
        }
    }
    ~store_stmt() { sqlite3_finalize(st); }
    operator sqlite3_stmt * () const { return st; }
};

server_state_store::server_state_store(const std::string & path, int32_t limit_mib) : path(path) {
    limit_bytes = limit_mib > 0 ? uint64_t(limit_mib) << 20 : 0;
    // SQLite creates the file, not its directory
    std::error_code ec;
    const auto dir = std::filesystem::path(path).parent_path();
    if (!dir.empty()) {
        std::filesystem::create_directories(dir, ec);
    }
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
        SRV_ERR("store: cannot open %s: %s\n", path.c_str(), sqlite3_errmsg(db));
        sqlite3_close(db);
        db = nullptr;
        return;
    }
    if (!exec(STORE_SCHEMA)) {
        sqlite3_close(db);
        db = nullptr;
    }
}

server_state_store::~server_state_store() {
    if (db) {
        sqlite3_close(db);
    }
}

bool server_state_store::exec(const char * sql) {
    char * err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        SRV_ERR("store: %s: %s\n", sql, err ? err : "?");
        sqlite3_free(err);
        return false;
    }
    return true;
}

void server_state_store::set_key(const std::string & model_path, ggml_type type_k, ggml_type type_v,
                                 llama_flash_attn_type flash_attn, const std::string & build) {
    if (!db) {
        return;
    }
    std::error_code ec;
    const auto size = (int64_t) std::filesystem::file_size(model_path, ec);
    const auto mtime = std::filesystem::last_write_time(model_path, ec);
    const int64_t mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count();

    std::string sha;
    {
        store_stmt st(db, "SELECT sha256 FROM model_hash WHERE path = ? AND size = ? AND mtime_ns = ?");
        sqlite3_bind_text(st, 1, model_path.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(st, 2, size);
        sqlite3_bind_int64(st, 3, mtime_ns);
        if (sqlite3_step(st) == SQLITE_ROW) {
            sha = (const char *) sqlite3_column_text(st, 0);
        }
    }
    if (sha.empty()) {
        SRV_INF("store: computing the SHA-256 of %s (once per model file)\n", model_path.c_str());
        const auto t0 = std::chrono::steady_clock::now();
        sha = sha256_file(model_path);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (sha.empty()) {
            SRV_ERR("store: cannot read %s; the store is off\n", model_path.c_str());
            sqlite3_close(db);
            db = nullptr;
            return;
        }
        SRV_INF("store: model SHA-256 %s (%.1f s)\n", sha.c_str(), s);
        store_stmt st(db, "INSERT OR REPLACE INTO model_hash (path, size, mtime_ns, sha256) VALUES (?, ?, ?, ?)");
        sqlite3_bind_text(st, 1, model_path.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(st, 2, size);
        sqlite3_bind_int64(st, 3, mtime_ns);
        sqlite3_bind_text(st, 4, sha.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(st);
    }
    key = "model=" + sha + "|type_k=" + ggml_type_name(type_k) + "|type_v=" + ggml_type_name(type_v) +
          "|flash_attn=" + llama_flash_attn_type_name(flash_attn) + "|build=" + build;
    SRV_INF("store: %s, key %s\n", path.c_str(), key.c_str());
}

// writes one byte vector as chunks of a part
static bool store_chunks(sqlite3 * db, int64_t id, const char * part, int idx, const uint8_t * data, size_t size) {
    store_stmt st(db, "INSERT INTO state_chunk (state_id, part, idx, chunk, data) VALUES (?, ?, ?, ?, ?)");
    for (size_t off = 0, n = 0; off < size; off += STORE_CHUNK, ++n) {
        sqlite3_bind_int64(st, 1, id);
        sqlite3_bind_text(st, 2, part, -1, SQLITE_STATIC);
        sqlite3_bind_int(st, 3, idx);
        sqlite3_bind_int64(st, 4, (sqlite3_int64) n);
        sqlite3_bind_blob64(st, 5, data + off, std::min(STORE_CHUNK, size - off), SQLITE_STATIC);
        if (sqlite3_step(st) != SQLITE_DONE) {
            SRV_ERR("store: writing %s: %s\n", part, sqlite3_errmsg(db));
            return false;
        }
        sqlite3_reset(st);
    }
    return true;
}

// reads all chunks of a part, in order
template <typename V>
static bool load_chunks(sqlite3 * db, int64_t id, const char * part, int idx, V & out) {
    out.clear();
    store_stmt st(db, "SELECT data FROM state_chunk WHERE state_id = ? AND part = ? AND idx = ? ORDER BY chunk");
    sqlite3_bind_int64(st, 1, id);
    sqlite3_bind_text(st, 2, part, -1, SQLITE_STATIC);
    sqlite3_bind_int(st, 3, idx);
    int rc;
    while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
        const auto * data = (const uint8_t *) sqlite3_column_blob(st, 0);
        const size_t n = (size_t) sqlite3_column_bytes(st, 0);
        const size_t old = out.size();
        out.resize(old + n);
        if (n > 0) {
            memcpy(out.data() + old, data, n);
        }
    }
    return rc == SQLITE_DONE;
}

static void delete_state(sqlite3 * db, int64_t id) {
    for (const char * sql : { "DELETE FROM state_chunk WHERE state_id = ?", "DELETE FROM state_checkpoint WHERE state_id = ?",
                              "DELETE FROM state WHERE id = ?" }) {
        store_stmt st(db, sql);
        sqlite3_bind_int64(st, 1, id);
        sqlite3_step(st);
    }
}

bool server_state_store::save(const server_prompt_cache_state & state) {
    if (!db || key.empty()) {
        return false;
    }
    // by index: get_tokens() asserts on a multimodal server, and positions an
    // image or audio chunk occupies read LLAMA_TOKEN_NULL here
    llama_tokens tokens(state.prompt.tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = state.prompt.tokens[i];
    }
    if (tokens.empty() || state.data.main.empty()) {
        return false;
    }
    // images and audio occupy positions as LLAMA_TOKEN_NULL; such prompts are not stored
    if (std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end()) {
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::string tokens_sha = hash_sha256_hex(tokens.data(), tokens.size() * sizeof(llama_token));
    {
        store_stmt st(db, "SELECT id FROM state WHERE key = ? AND tokens_sha256 = ?");
        sqlite3_bind_text(st, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 2, tokens_sha.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(st) == SQLITE_ROW) {
            return true; // already stored
        }
    }
    const size_t bytes = state.size();

    if (!exec("BEGIN")) {
        return false;
    }
    int64_t id = -1;
    bool ok = true;
    {
        store_stmt st(db, "INSERT INTO state (key, n_tokens, tokens_sha256, tokens, bytes, created, last_used) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?)");
        sqlite3_bind_text(st, 1, key.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(st, 2, (int) tokens.size());
        sqlite3_bind_text(st, 3, tokens_sha.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob64(st, 4, tokens.data(), tokens.size() * sizeof(llama_token), SQLITE_STATIC);
        sqlite3_bind_int64(st, 5, (sqlite3_int64) bytes);
        sqlite3_bind_int64(st, 6, now_s());
        sqlite3_bind_int64(st, 7, now_s());
        ok = sqlite3_step(st) == SQLITE_DONE;
        id = sqlite3_last_insert_rowid(db);
    }
    ok = ok && store_chunks(db, id, "main", 0, state.data.main.data(), state.data.main.size());
    ok = ok && store_chunks(db, id, "drft", 0, state.data.drft.data(), state.data.drft.size());
    int idx = 0;
    for (const auto & ckpt : state.prompt.checkpoints) {
        if (!ok) {
            break;
        }
        store_stmt st(db, "INSERT INTO state_checkpoint (state_id, idx, n_tokens, pos_min, pos_max) VALUES (?, ?, ?, ?, ?)");
        sqlite3_bind_int64(st, 1, id);
        sqlite3_bind_int(st, 2, idx);
        sqlite3_bind_int64(st, 3, ckpt->n_tokens);
        sqlite3_bind_int(st, 4, ckpt->pos_min);
        sqlite3_bind_int(st, 5, ckpt->pos_max);
        ok = sqlite3_step(st) == SQLITE_DONE;
        ok = ok && store_chunks(db, id, "ckpt_tgt", idx, ckpt->data_tgt.data(), ckpt->data_tgt.size());
        ok = ok && store_chunks(db, id, "ckpt_dft", idx, ckpt->data_dft.data(), ckpt->data_dft.size());
        ok = ok && store_chunks(db, id, "ckpt_spec", idx, ckpt->data_spec.data(), ckpt->data_spec.size());
        ++idx;
    }
    if (!ok) {
        exec("ROLLBACK");
        return false;
    }
    exec("COMMIT");
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    SRV_INF("store: saved %zu tokens, %.1f MiB, %zu checkpoints (%.2f s)\n",
            tokens.size(), bytes / (1024.0 * 1024.0), state.prompt.checkpoints.size(), s);
    evict(id);
    return true;
}

// over the size limit: remove the least recently used states, never keep_id
void server_state_store::evict(int64_t keep_id) {
    if (limit_bytes == 0) {
        return;
    }
    for (;;) {
        uint64_t total = 0;
        {
            store_stmt st(db, "SELECT COALESCE(SUM(bytes), 0) FROM state");
            if (sqlite3_step(st) == SQLITE_ROW) {
                total = (uint64_t) sqlite3_column_int64(st, 0);
            }
        }
        if (total <= limit_bytes) {
            return;
        }
        int64_t oldest = -1;
        {
            store_stmt st(db, "SELECT id FROM state WHERE id != ? ORDER BY last_used, id LIMIT 1");
            sqlite3_bind_int64(st, 1, keep_id);
            if (sqlite3_step(st) == SQLITE_ROW) {
                oldest = sqlite3_column_int64(st, 0);
            }
        }
        if (oldest < 0) {
            return;
        }
        SRV_INF("store: over %.0f MiB, removing the least recently used state %lld\n",
                limit_bytes / (1024.0 * 1024.0), (long long) oldest);
        exec("BEGIN");
        delete_state(db, oldest);
        exec("COMMIT");
    }
}

server_state_store::candidate server_state_store::find(const server_tokens & tokens_new, float f_keep_base, float f_sim_base) {
    candidate best;
    if (!db || key.empty() || tokens_new.size() == 0) {
        return best;
    }
    // by index: get_tokens() asserts on a multimodal server
    llama_tokens tnew(tokens_new.size());
    for (size_t i = 0; i < tnew.size(); ++i) {
        tnew[i] = tokens_new[i];
    }
    float f_keep_best = f_keep_base;
    float f_sim_best  = f_sim_base;

    store_stmt st(db, "SELECT id, n_tokens, tokens FROM state WHERE key = ?");
    sqlite3_bind_text(st, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    while (sqlite3_step(st) == SQLITE_ROW) {
        const int64_t id = sqlite3_column_int64(st, 0);
        const int n = sqlite3_column_int(st, 1);
        const auto * tok = (const llama_token *) sqlite3_column_blob(st, 2);
        const size_t m = std::min<size_t>((size_t) n, tnew.size());
        int lcp = 0;
        while ((size_t) lcp < m && tok[lcp] == tnew[lcp] && tnew[lcp] != LLAMA_TOKEN_NULL) {
            ++lcp;
        }
        const float f_keep = float(lcp) / n;
        const float f_sim  = float(lcp) / tnew.size();
        // the prompt cache's rule: don't trash large prompts; better on both measures
        if (f_keep < 0.25f) {
            continue;
        }
        if (f_keep_best < f_keep && f_sim_best < f_sim) {
            f_keep_best = f_keep;
            f_sim_best  = f_sim;
            best = { id, n, lcp, f_keep, f_sim };
        }
    }
    return best;
}

bool server_state_store::load(int64_t id, bool has_mtmd, server_prompt & prompt, server_prompt_data & data) {
    if (!db) {
        return false;
    }
    llama_tokens tokens;
    {
        store_stmt st(db, "SELECT tokens FROM state WHERE id = ?");
        sqlite3_bind_int64(st, 1, id);
        if (sqlite3_step(st) != SQLITE_ROW) {
            return false;
        }
        const auto * tok = (const llama_token *) sqlite3_column_blob(st, 0);
        tokens.assign(tok, tok + sqlite3_column_bytes(st, 0) / sizeof(llama_token));
    }
    if (!load_chunks(db, id, "main", 0, data.main) || !load_chunks(db, id, "drft", 0, data.drft)) {
        return false;
    }
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;
    {
        store_stmt st(db, "SELECT idx, n_tokens, pos_min, pos_max FROM state_checkpoint WHERE state_id = ? ORDER BY idx");
        sqlite3_bind_int64(st, 1, id);
        while (sqlite3_step(st) == SQLITE_ROW) {
            const int idx = sqlite3_column_int(st, 0);
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            ckpt->n_tokens = sqlite3_column_int64(st, 1);
            ckpt->pos_min  = sqlite3_column_int(st, 2);
            ckpt->pos_max  = sqlite3_column_int(st, 3);
            if (!load_chunks(db, id, "ckpt_tgt", idx, ckpt->data_tgt) ||
                !load_chunks(db, id, "ckpt_dft", idx, ckpt->data_dft) ||
                !load_chunks(db, id, "ckpt_spec", idx, ckpt->data_spec)) {
                return false;
            }
            checkpoints.push_back(std::move(ckpt));
        }
    }
    prompt.tokens = server_tokens(tokens, has_mtmd);
    prompt.checkpoints = std::move(checkpoints);
    {
        store_stmt st(db, "UPDATE state SET last_used = ? WHERE id = ?");
        sqlite3_bind_int64(st, 1, now_s());
        sqlite3_bind_int64(st, 2, id);
        sqlite3_step(st);
    }
    return true;
}
