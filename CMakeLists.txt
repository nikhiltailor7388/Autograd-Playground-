cmake_minimum_required(VERSION 3.14) # 3.14+ needed for FetchContent_MakeAvailable
project(autograd_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
enable_testing()

include(FetchContent)

# nlohmann/json: single-header JSON library, used by the core lib itself
# now (model_io.cpp, for weight/bias serialization) as well as by
# backend/. Declared once, here, so both consumers share a single
# populated copy instead of each fetching (and potentially drifting on)
# their own. backend/CMakeLists.txt re-declares the same name/tag, which
# is a harmless no-op once it's already been made available here.
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(json)

# ---- Core engine library ----
# Phases 1-4 (Value + backprop + MLP/XOR) plus Phase 7's advanced
# features (custom exceptions, CSV loading, model serialization),
# compiled ONCE into a static library, instead of every executable below
# re-listing and recompiling the same sources itself. Both the CLI
# executables and the backend/ server (Phase 5) link against this one
# engine rather than risking divergent copies.
add_library(autograd_core STATIC
    src/value.cpp
    src/nn.cpp
    src/csv_loader.cpp
    src/model_io.cpp
)
target_include_directories(autograd_core PUBLIC include)
target_link_libraries(autograd_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(autograd_demo
    src/main.cpp
)
target_link_libraries(autograd_demo PRIVATE autograd_core)

add_executable(grad_check
    tests/grad_check.cpp
)
target_link_libraries(grad_check PRIVATE autograd_core)
add_test(NAME grad_check COMMAND grad_check)

add_executable(advanced_tests
    tests/advanced_tests.cpp
)
target_link_libraries(advanced_tests PRIVATE autograd_core)
add_test(NAME advanced_tests COMMAND advanced_tests)

add_executable(train
    src/train.cpp
)
target_link_libraries(train PRIVATE autograd_core)

# ---- Phase 5: web backend (REST API around the engine) ----
# Opt-out switch in case someone wants to build just the original CLI
# tools without pulling the backend's extra dependency (httplib) over
# the network.
option(AUTOGRAD_BUILD_BACKEND "Build the autograd_server REST API backend" ON)
if(AUTOGRAD_BUILD_BACKEND)
    add_subdirectory(backend)
endif()
