# backend/CMakeLists.txt
#
# Builds `autograd_server`, the REST API layer around the core engine.
# Pulls in two small, header-only(ish) dependencies via FetchContent so
# nothing needs to be installed system-wide first -- `cmake --build` alone
# is enough (an internet connection is needed the first time, to download
# them; CMake caches them under build/_deps/ after that).

include(FetchContent)

# cpp-httplib: single-header HTTP server/client library.
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)

# nlohmann/json: single-header JSON library.
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

# We only need the headers from both, not their own test suites/examples,
# so keep the fetch lightweight.
set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE INTERNAL "" FORCE)

FetchContent_MakeAvailable(httplib json)

add_executable(autograd_server
    src/server.cpp
    src/api_handlers.cpp
    src/expr_parser.cpp
    src/graph_json.cpp
)

target_include_directories(autograd_server PRIVATE include)

target_link_libraries(autograd_server
    PRIVATE
        autograd_core        # the core Value/MLP engine (defined in the root CMakeLists.txt)
        httplib::httplib
        nlohmann_json::nlohmann_json
)

# cpp-httplib needs threading support for its request-handling thread pool.
find_package(Threads REQUIRED)
target_link_libraries(autograd_server PRIVATE Threads::Threads)
