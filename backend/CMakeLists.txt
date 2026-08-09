 include(FetchContent)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

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
        autograd_core
        httplib::httplib
        nlohmann_json::nlohmann_json
)

find_package(Threads REQUIRED)
target_link_libraries(autograd_server PRIVATE Threads::Threads)

find_package(OpenSSL REQUIRED)
target_compile_definitions(autograd_server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(autograd_server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
