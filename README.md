# autograd-cpp

A tiny **automatic differentiation (autograd) engine**, built from scratch in
C++17 using only the standard library — no PyTorch, no TensorFlow, no
external ML/tensor/autograd dependencies of any kind.

It can:
1. Build a computation graph automatically just from writing normal-looking
   arithmetic on `Value` objects.
2. Compute gradients of any expression via reverse-mode automatic
   differentiation (backpropagation).
3. Train a tiny Multi-Layer Perceptron (MLP) on the XOR problem, with the
   loss visibly decreasing to ~0.

This project is scalar-valued: each `Value` holds one `double`, not a
tensor/array. That keeps every operation's math (and its derivative)
readable in a few lines, at the cost of speed — see **Known Limitations**
below.

---

## What is a computation graph?

Every time you combine two `Value`s with `+`, `*`, `tanh`, etc., instead of
just getting back a plain number, you get a new `Value` **node** that
remembers which `Value`s it was built from and which operation combined
them. Chain enough of these together and you get a graph (technically a
DAG — directed acyclic graph) that records the entire history of how a
final result was computed.

For example, `L = (a * b) + c` builds this graph:

```
              L  ( "+" node )
             / \
        (a*b)   c
        /   \
       a     b
```

Each arrow above actually points from child to parent in the code (a `+`
node stores `children = {a*b, c}`), but conceptually, data flows
**forward** from the leaves (`a`, `b`, `c`) up to the root (`L`) — that's
the normal arithmetic you wrote. Gradients flow the opposite direction,
**backward**, from the root down to the leaves — that's backpropagation.

## How does backpropagation work?

Backpropagation is one big application of the chain rule from calculus:
if `L` depends on `m`, and `m` depends on `x`, then

```
dL/dx = (dL/dm) * (dm/dx)
```

Starting from the output node `L` (with `dL/dL = 1`, trivially), we can
push gradients backward one node at a time: each node already knows its
own **local** derivative with respect to its children (e.g. for `m = a*b`,
`dm/da = b`), so once a node knows `dL/dm` (its own gradient, arriving from
further up the graph), it can compute `dL/da = dL/dm * b` and hand that
down to `a`.

The catch: if a `Value` is used in **more than one place** (e.g. `x + x`,
or a weight shared by several neurons), its true gradient is the **sum**
of every contribution flowing back into it, and none of those
contributions is complete until every node that used it has run. So
before walking backward, we compute a **topological order** of the graph
(via post-order DFS) — an order where every node comes after everything
that depends on it. Walking that order in reverse guarantees each node's
gradient is fully accumulated (via `+=`, never `=`) before it's used to
compute gradients for its own children.

```
forward pass (build graph):     a ----\
                                       *---\
                                 b ----/     \
                                              + ---> L
                                 c -----------/

backward pass (after L->backward(), grad seeded 1.0 at L, walked in
reverse topological order):

     a <---- dL/da            \
                                *<---- dL/d(a*b) <--\
     b <---- dL/db            /                      + <---- dL/dL = 1
                                                      /
     c <-------------------------------- dL/dc <----/
```

---

## Project structure

```
autograd-cpp/
├── CMakeLists.txt          # build config: core lib + CLI tools + tests + backend subdir
├── Dockerfile              # multi-stage build -> containerized autograd_server
├── .gitignore
├── README.md
├── include/
│   ├── value.h                  # Value class: the core autograd node + operators
│   ├── nn.h                     # Neuron / Layer / MLP / mse_loss, built on Value
│   ├── autograd_exceptions.h    # custom exception hierarchy (Phase 7)
│   ├── csv_loader.h             # CSV dataset loader (Phase 7)
│   └── model_io.h               # MLP JSON save/load (Phase 7)
├── src/
│   ├── value.cpp            # Value operator implementations, backward()
│   ├── nn.cpp               # Neuron / Layer / MLP / mse_loss implementations
│   ├── csv_loader.cpp       # Phase 7: parses tabular data into a Dataset
│   ├── model_io.cpp         # Phase 7: MLP <-> JSON (weights + architecture)
│   ├── main.cpp             # Phase 1 demo: builds & prints (a*b)+c
│   └── train.cpp            # trains an MLP on XOR, or on a CSV file if given one
├── data/
│   └── xor.csv              # sample dataset in the format csv_loader.h expects
├── tests/
│   ├── grad_check.cpp       # gradient checker: analytical vs numerical
│   └── advanced_tests.cpp   # Phase 7: exceptions, CSV loader, model I/O round-trip
├── backend/                 # REST API server around the engine (see below)
└── frontend/                # browser UI (see below)
```

## Building and running

Requires CMake >= 3.14 and a C++17 compiler.

### Deploying to Render

The repository includes `render.yaml` for a two-service deployment: a Docker
web service for the real C++ backend and a Node web service for the frontend
and PieGeni proxy. In Render, choose **New > Blueprint**, connect this
repository, and apply the blueprint. Set `PIEGENI_API_KEY` in the frontend
service's environment variables; do not commit the key. The frontend proxies
compute and training requests to the C++ service automatically.

```bash
cmake -B build
cmake --build build
```

The first configure needs an internet connection once, to fetch two
small header-only dependencies (`nlohmann/json` for the core lib and
backend, `cpp-httplib` for the backend only) via CMake's `FetchContent`;
they're cached under `build/_deps/` after that. If you only want the
original CLI tools and don't want to fetch `cpp-httplib`, turn the
backend off with `-DAUTOGRAD_BUILD_BACKEND=OFF` (the core lib's own
`nlohmann/json` dependency, needed for model saving/loading, is still
fetched either way).

The expression API supports arithmetic, fixed numeric powers, and the
autograd-aware functions `sin`, `cos`, `tan`, `cot`, `sec`, `cosec`, `tanh`,
`exp`, `log`, and `relu` (trigonometric arguments are in radians). The
frontend plots equations, shows exact steps for common square/linear forms,
and uses numerical differentiation plus Simpson integration for general
plotted expressions.

This produces the following executables in `build/`:

| Executable       | What it does                                                             |
|------------------|---------------------------------------------------------------------------|
| `autograd_demo`  | Builds `L = (a * b) + c` and prints the computation graph structure.    |
| `grad_check`     | Verifies every operation's analytical gradient against a finite-difference numerical gradient; prints a pass/fail table. |
| `advanced_tests` | Verifies Phase 7's custom exceptions, CSV loader, and model save/load round-trip; prints a pass/fail table. |
| `train`          | Trains a small MLP on XOR (default) or on a CSV file (if given a path), prints loss every 100 epochs, then final predictions; optionally saves the trained weights to JSON. |

Run them with, e.g.:

```bash
./build/autograd_demo
./build/grad_check
./build/advanced_tests
./build/train                              # trains on the built-in XOR dataset
./build/train data/xor.csv                 # trains on a CSV file instead
./build/train data/xor.csv model.json      # ...and saves the trained weights
```

Or run the whole test suite through CTest, which is wired up for both:

```bash
ctest --test-dir build --output-on-failure
```

Expected `grad_check` output: `14 / 14 checks passed`.
Expected `advanced_tests` output: `17 / 17 checks passed`.

Expected `train` output: loss starting around `~2.3` and dropping to
effectively `0` (around `1e-20` or smaller) within ~1000-2000 epochs, with
final predictions matching the XOR truth table (`(0,0)->0`, `(0,1)->1`,
`(1,0)->1`, `(1,1)->0`) to several decimal places.

## Using it yourself

```cpp
#include "value.h"

auto a = make_value(2.0, "a");
auto b = make_value(-3.0, "b");
auto c = make_value(10.0, "c");

auto L = a * b + c;
L->backward();          // computes gradients w.r.t. every leaf that fed into L

std::cout << a->grad;   // dL/da
```

```cpp
#include "nn.h"

MLP mlp(2, {4, 4, 1});         // 2 inputs -> hidden(4) -> hidden(4) -> output(1)
auto out = mlp.forward({make_value(0.0), make_value(1.0)});
```

---

## Advanced features

### Custom exceptions

Every error the engine itself raises — as opposed to errors raised by the
standard library or a dependency — derives from `autograd::AutogradError`
(itself a `std::runtime_error`, so any code that already catches
`std::exception`, like the backend's route handler, keeps working
unmodified):

| Exception                          | Thrown when...                                                        |
|-------------------------------------|-------------------------------------------------------------------------|
| `autograd::MathError`               | `log(x)` for `x <= 0`, division by zero, `0` raised to a negative power |
| `autograd::DimensionMismatchError`  | a `Neuron`/`mse_loss` call is given inputs of the wrong size            |
| `autograd::DatasetError`            | a CSV file is missing, empty, ragged, or has a non-numeric cell         |
| `autograd::SerializationError`      | a model JSON file is missing, malformed, or doesn't match the architecture |

This means invalid math never silently produces `nan`/`inf` that then
poisons every downstream gradient — it fails loudly, at the point of the
mistake, with a message describing exactly what was wrong.

```cpp
#include "value.h"
#include "autograd_exceptions.h"

try {
    auto x = make_value(-1.0);
    auto y = log(x);           // log of a non-positive number
} catch (const autograd::MathError& e) {
    std::cerr << "Math error: " << e.what() << "\n";
}
```

### CSV dataset loader

`include/csv_loader.h` parses a simple CSV file — every column except
the last is treated as a numeric input feature, the last column as the
numeric target — into a `Dataset{inputs, targets, feature_names}`, so
`train.cpp` (or your own code) can train on real tabular data instead of
only the hardcoded XOR truth table:

```cpp
#include "csv_loader.h"

autograd::Dataset ds = autograd::load_csv("data/xor.csv"); // has_header=true by default
// ds.inputs[i]  -> feature vector for row i
// ds.targets[i] -> label for row i
```

`data/xor.csv` ships as a working example of the expected format:

```csv
x1,x2,label
0.0,0.0,0
0.0,1.0,1
1.0,0.0,1
1.0,1.0,0
```

The CLI `train` tool already wires this in end-to-end — `train
data/xor.csv` loads and trains on it instead of the hardcoded arrays.

### Model serialization (JSON export/import)

`include/model_io.h` saves a trained `MLP`'s full architecture (input
size, per-layer neuron counts, per-layer activation) and every weight
and bias to a JSON file, and can reconstruct an identical `MLP` from
that file later — so a training run's result can outlive the process
that produced it:

```cpp
#include "model_io.h"

autograd::save_model(mlp, "trained_xor.json");
// ... later, possibly in a different run/process ...
MLP loaded = autograd::load_model("trained_xor.json");
```

`train data/xor.csv model.json` does exactly this at the end of
training. The JSON is plain and human-readable — worth a quick look if
you're curious what a trained MLP's weights actually look like:

```json
{
  "n_inputs": 2,
  "layer_sizes": [4, 4, 1],
  "layers": [
    { "use_activation": true, "neurons": [ { "weights": [0.30, -0.65], "bias": -0.04 }, ... ] },
    ...
  ]
}
```

---

## Known Limitations

Being explicit about what this project does *not* do:

- **Scalar-only.** Every `Value` wraps a single `double`, not a
  vector/matrix/tensor. Real ML frameworks batch computation across
  tensors for performance; this engine processes one number at a time,
  which is simple to read but slow for anything beyond toy problems.
- **No batching.** The training loop in `train.cpp` loops over the 4 XOR
  examples one at a time inside a single epoch's forward pass, rather
  than using vectorized batch operations.
- **No GPU support.** Everything runs on CPU, single-threaded.
- **Only plain SGD.** The optimizer is `param.data -= lr * param.grad`,
  the simplest possible gradient descent update. No momentum, Adam,
  learning-rate schedules, etc.
- **No weight-decay / regularization.** Nothing prevents overfitting on
  larger problems; XOR is small enough that this doesn't matter here.
- **Basic memory/perf profile.** Every operation allocates a new
  `shared_ptr<Value>` node (Python-frameworks-in-eager-mode do this too,
  but production C++ ML code typically avoids per-op heap allocation).
  For a learning project this trade-off is intentional: readability of
  the graph structure over raw speed.
- **No automatic graph freeing between training steps.** Each epoch's
  forward pass builds an entirely new graph of `Value` nodes; old graphs
  are dropped once nothing still references them (`shared_ptr` handles
  this automatically), but the engine doesn't reuse graph memory across
  epochs the way a production framework's allocator might.
- **Single fixed random seed** (`std::mt19937 rng(42)` in `nn.cpp`) for
  weight initialization, chosen for reproducible runs while
  developing/debugging rather than true randomness.
- **CSV loader is intentionally minimal.** No quoted-field support (a
  comma inside a quoted cell will be misread as a column separator), no
  categorical/string feature encoding, no missing-value imputation — it
  expects a clean, fully-numeric, comma-delimited file with the target
  in the last column.
- **Model files trust their own `layer_sizes`/architecture fields but
  not much else.** `load_model` validates shapes (weight counts, neuron
  counts, activation flags) against what the declared architecture
  expects and throws `autograd::SerializationError` on mismatch, but it
  doesn't checksum the file or protect against a maliciously crafted
  JSON payload — treat model files as trusted input, the same as source
  code.

These are deliberate simplifications for a from-scratch learning project,
not oversights — the goal was to make backpropagation itself fully
transparent, not to build a production training system.

---

## Full-stack web interface

On top of the CLI engine above, this project also ships a small full-stack
app: a C++ REST API (`backend/`) wrapping the exact same engine, and a
static HTML/JS frontend (`frontend/`) that lets you build expressions,
watch `backward()` run on a live computation-graph diagram, and train the
XOR network with a real-time loss chart, all in the browser.

```
autograd-cpp/
├── include/, src/, tests/     — the core engine (Phases 1-4, unchanged)
├── backend/                   — Phase 5: REST API around the engine
│   ├── include/                 expr_parser.h, graph_json.h, api_handlers.h
│   ├── src/                     expr_parser.cpp, graph_json.cpp, api_handlers.cpp, server.cpp
│   └── CMakeLists.txt            fetches cpp-httplib + nlohmann/json, builds `autograd_server`
├── frontend/                  — Phase 6: browser UI (no build step)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── CMakeLists.txt             — root build: core library + CLI tools + backend subdirectory
```

### 1. Build and run the backend

Requires CMake 3.14+ and a C++17 compiler. The first build needs an
internet connection once, to fetch `cpp-httplib` and `nlohmann/json` via
CMake's `FetchContent` (they're cached under `build/_deps/` after that).

```bash
cd autograd
cmake -B build
cmake --build build -j
./build/backend/autograd_server        # listens on http://localhost:8080 by default
# or choose a different port:
./build/backend/autograd_server 9000
```

You should see:

```
autograd-cpp backend listening on http://localhost:8080
  GET  /api/health
  POST /api/compute      { "expr": "(a * b) + a", "a": 2.0, "b": 3.0 }
  POST /api/train-xor    { "epochs": 500, "learning_rate": 0.05, "hidden_layers": [4, 4] }
```

If you only want the original CLI tools (`autograd_demo`, `grad_check`,
`train`) and don't want to fetch the backend's dependencies, configure
with the backend turned off:

```bash
cmake -B build -DAUTOGRAD_BUILD_BACKEND=OFF
```

Quick sanity check with `curl`, once the server is running:

```bash
curl -X POST http://localhost:8080/api/compute \
  -H "Content-Type: application/json" \
  -d '{"expr": "(a * b) + a", "a": 2.0, "b": 3.0}'
```

### 2. Open the frontend

The frontend is plain static files — no build step, no npm install.
Just open `frontend/index.html` directly in a browser, **or** serve it
locally (recommended, some browsers restrict `fetch()` from `file://`
pages):

```bash
cd autograd/frontend
python3 -m http.server 5500
# then open http://localhost:5500 in your browser
```

The page assumes the backend is at `http://localhost:8080`. If you ran it
on a different port or host, click **backend settings** in the top-right
of the page and update the API base URL — it's saved in the browser for
next time.

### 3. Using it

- **Expression Sandbox** — set your variables (defaults to `a = 2`,
  `b = 3`), type an expression like `(a * b) + a` or `tanh(a) + b`, and
  click **run backward()**. The right-hand panel draws the resulting
  computation graph, with every node's `data` and `grad` labeled, and
  animates a small pulse tracing the gradient backward from the output to
  each leaf — the same reverse-topological order `Value::backward()` runs
  internally.
- **XOR Training Dashboard** — set hidden-layer widths (e.g. `4, 4`),
  epochs, and learning rate, then click **train on XOR**. This calls
  `POST /api/train-xor`, which runs the whole training loop server-side
  and returns the full loss history in one response (not a live stream —
  see note below), which the page then charts.

### Notes and known limitations of the web layer

- **`^` (power) only supports a constant numeric exponent** in the
  expression parser (e.g. `x^2` works, `x^y` doesn't) — this matches the
  underlying engine's `pow(ValuePtr, double)`, which has no gradient rule
  for a *Value*-valued exponent.
- **`/api/train-xor` is request/response, not a stream.** It runs the
  requested number of epochs synchronously and returns every epoch's loss
  in one JSON array; the frontend then renders that array as a chart. For
  a very large epoch count this means a real (if short) wait before the
  chart appears, rather than watching it update epoch-by-epoch live. True
  live streaming would need Server-Sent Events or a WebSocket endpoint —
  a natural next step, but out of scope here.
- **No auth, no persistence, no rate limiting beyond basic input
  bounds** (epoch count, layer sizes, and hidden-layer widths are capped
  server-side in `api_handlers.cpp`) — this is a local development/demo
  tool, not hardened for exposing on the open internet.

---

## Running with Docker

A `Dockerfile` is included for a one-command, dependency-free way to
build and run the backend server (no local CMake/compiler/git needed —
just Docker):

```bash
docker build -t autograd-cpp .
docker run --rm -p 8080:8080 autograd-cpp
```

This runs a multi-stage build: a `builder` stage compiles everything
(including running the full test suite via `ctest` as part of the
build, so a broken build fails the `docker build` itself) and a slim
`runtime` stage ships only the compiled `autograd_server` binary plus
the static `frontend/` and `data/` assets — no compiler toolchain, no
CMake cache, no dependency source trees end up in the final image.

Once running, open `frontend/index.html` locally (see **Open the
frontend** above) and point it at `http://localhost:8080`, or hit the
API directly:

```bash
curl http://localhost:8080/api/health
```

To use a different port:

```bash
docker run --rm -p 9000:9000 autograd-cpp 9000
```

---

## Screenshots / demo

*(Add screenshots or a short GIF of the web dashboard here before
publishing — e.g. the Expression Sandbox's computation-graph view mid
`backward()`-pulse-animation, and the XOR Training Dashboard's live loss
chart after a run. A simple way to capture one: run the backend and
frontend as described above, use the UI, and drop the image files into
a `docs/` or `screenshots/` folder, then reference them here, e.g.:)*

```markdown
![Expression Sandbox: computation graph view](docs/screenshot-graph.png)
![XOR Training Dashboard: loss chart](docs/screenshot-training.png)
```

---

## License

No license file is included yet — add one (e.g. MIT) before treating
this as open source others can freely reuse.
