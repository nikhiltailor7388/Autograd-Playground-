#pragma once

#include "nn.h"
#include <string>

// Save/load a trained MLP's full architecture and parameters (weights +
// biases) to/from a JSON file, so a training run's result can survive
// past the process that produced it (e.g. train once via the CLI or the
// web backend, then load the same weights later for inference without
// retraining).
namespace autograd {

// Writes `mlp` to `path` as JSON, including enough architecture
// information (input size, per-layer neuron counts, per-layer
// activation flag) to reconstruct an identical MLP via load_model,
// not just the raw numbers.
//
// Throws autograd::SerializationError if the file can't be opened for
// writing.
void save_model(const MLP& mlp, const std::string& path);

// Reads a JSON file previously written by save_model, reconstructs an
// MLP with the same architecture, and restores every weight and bias to
// its saved value.
//
// Throws autograd::SerializationError if: the file can't be opened, the
// JSON is malformed, required fields are missing, or a layer's saved
// weight/bias counts don't match its declared shape.
MLP load_model(const std::string& path);

} // namespace autograd
