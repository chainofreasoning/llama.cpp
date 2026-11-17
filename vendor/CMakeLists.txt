# mtmd needs these even when common is not built
add_subdirectory(hash)
add_subdirectory(miniaudio)
add_subdirectory(nlohmann)
add_subdirectory(sheredom)
add_subdirectory(stb)

# the reuse store (llama-bench --store)
add_subdirectory(sqlite3)

# only used by common
if (LLAMA_BUILD_COMMON)
    add_subdirectory(cpp-httplib)
endif()
