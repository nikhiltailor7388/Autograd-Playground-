#pragma once

#include "value.h"
#include <nlohmann/json.hpp>

// Serializes the computation graph rooted at `root` into a JSON object of
// the shape:
//   {
//     "nodes": [ { "id", "label", "op", "data", "grad", "is_leaf" }, ... ],
//     "edges": [ { "source", "target" }, ... ],   // source (child) -> target (parent),
//                                                  // i.e. the direction data flows forward
//     "root_id": "..."
//   }
//
// Call this AFTER root->backward() has run if you want gradients included;
// grad will simply be 0.0 for every node otherwise (that's still valid,
// just means "not yet differentiated").
nlohmann::json build_graph_json(const ValuePtr& root);
