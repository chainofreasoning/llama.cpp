# SQLite, the amalgamation, for the reuse store (llama-bench --store).
# Version and source: README.md in this directory.
set(TARGET vendor-sqlite3)

add_library(${TARGET} STATIC sqlite3.c sqlite3.h)
add_library(vendor::sqlite3 ALIAS ${TARGET})

# built only when a target links it
set_target_properties(${TARGET} PROPERTIES EXCLUDE_FROM_ALL ON POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${TARGET} PRIVATE
    SQLITE_THREADSAFE=1
    SQLITE_DQS=0
    SQLITE_OMIT_LOAD_EXTENSION
)

# third-party code: no warnings
if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${TARGET} PRIVATE /w)
else()
    target_compile_options(${TARGET} PRIVATE -w)
endif()

find_package(Threads REQUIRED)
target_link_libraries(${TARGET} PUBLIC Threads::Threads)

# consumers include <sqlite3.h>
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
