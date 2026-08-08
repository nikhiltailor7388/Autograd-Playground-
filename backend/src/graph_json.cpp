#include "graph_json.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>

using json = nlohmann::json;

namespace {

// Picks a human-readable display name for a node: its label if it has one
// (e.g. a leaf the caller named "a"), otherwise its operator (e.g. "*"),
// otherwise a generic fallback for an unlabeled leaf/constant.
std::string display_label(const Value& v) {
    if (!v.label.empty()) return v.label;
    if (!v.op.empty()) return v.op;
    return "const";
}

} // namespace

json build_graph_json(const ValuePtr& root) {
    // Same post-order-DFS-over-shared_ptr-children walk that Value::backward()
    // uses to build a topological order, reused here purely for traversal
    // (we don't need reverse order for serialization, just "visit every
    // reachable node exactly once").
    std::vector<Value*> order;
    std::unordered_set<Value*> visited;
    std::unordered_map<Value*, std::string> ids;

    std::function<void(const ValuePtr&)> visit = [&](const ValuePtr& v) {
        if (visited.count(v.get())) return;
        visited.insert(v.get());
        for (const auto& child : v->children) {
            visit(child);
        }
        // Assign the id once a node is first discovered in this walk, in
        // the same order nodes are appended to `order`, so ids are stable
        // and easy to cross-reference between "nodes" and "edges" below.
        ids[v.get()] = "n" + std::to_string(order.size());
        order.push_back(v.get());
    };
    visit(root);

    json nodes = json::array();
    for (Value* v : order) {
        nodes.push_back({
            {"id", ids[v]},
            {"label", display_label(*v)},
            {"op", v->op},
            {"data", v->data},
            {"grad", v->grad},
            {"is_leaf", v->children.empty()},
        });
    }

    json edges = json::array();
    for (Value* v : order) {
        for (const auto& child : v->children) {
            // source = child (computed first), target = v (uses child as
            // an input) -- this is the direction data flows forward, which
            // is the intuitive direction to draw an arrow in the UI.
            edges.push_back({
                {"source", ids[child.get()]},
                {"target", ids[v]},
            });
        }
    }

    json result;
    result["nodes"] = nodes;
    result["edges"] = edges;
    result["root_id"] = ids[root.get()];
    return result;
}
