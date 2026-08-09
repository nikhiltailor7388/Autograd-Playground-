 # syntax=docker/dockerfile:1
#
# Multi-stage build:
#   1. "builder" has the full compiler toolchain and CMake, fetches the
#      header-only deps (httplib, nlohmann/json) via FetchContent, and
#      compiles everything.
#   2. The final image copies out only the compiled autograd_server
#      binary and the static frontend assets it serves alongside itself
#      -- no compiler, no CMake cache, no dependency source trees ship
#      in the runtime image.

# ---- Stage 1: build ----
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

# Release build of everything (core lib, CLI tools, tests, backend
# server). AUTOGRAD_BUILD_BACKEND stays at its default ON.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure

# ---- Stage 2: runtime ----
FROM ubuntu:24.04 AS runtime

# libstdc++ is needed at runtime even though we don't need the rest of
# build-essential; it's part of ubuntu:24.04's base already via
# libc6/libgcc, but pull the explicit runtime package to be safe across
# base image variants.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /workspace/build/backend/autograd_server ./autograd_server
COPY --from=builder /workspace/docs ./docs
COPY --from=builder /workspace/data ./data

EXPOSE 8080

# The server takes an optional port argument (defaults to 8080 if omitted).
ENTRYPOINT ["./autograd_server"]
CMD ["8080"]
