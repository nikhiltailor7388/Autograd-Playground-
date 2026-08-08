#pragma once

#include <nlohmann/json.hpp>

// Each handler takes the parsed JSON request body and returns the JSON
// response body. Handlers throw std::exception (or a subclass, e.g.
// ExprParseError / std::invalid_argument) on any bad input; server.cpp is
// responsible for catching that and turning it into an HTTP 400.
//
// Kept separate from server.cpp so the request/response logic can be
// tested or reused without spinning up an actual HTTP server.

// POST /api/compute
// Body:   { "expr": "(a * b) + a", "a": 2.0, "b": 3.0 }
//         (any top-level numeric field other than "expr" is treated as a
//         variable the expression can reference)
// Result: { "expr", "result": {"data","grad"}, "graph": {...} }
nlohmann::json handle_compute(const nlohmann::json& body);

// POST /api/train-xor
// Body (all fields optional):
//   { "epochs": 500, "learning_rate": 0.05, "hidden_layers": [4, 4] }
// Result: { "epochs", "learning_rate", "hidden_layers",
//           "losses": [loss_epoch_0, loss_epoch_1, ...],
//           "final_predictions": [ {"input":[x0,x1], "predicted", "target"}, ... ] }
nlohmann::json handle_train_xor(const nlohmann::json& body);
