#include "api_handlers.h"
#include "value.h"
#include "nn.h"
#include "expr_parser.h"
#include "graph_json.h"
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

namespace {

// Reasonable ceilings so a single synchronous request can't tie up the
// server for an unbounded amount of time. /api/train-xor computes and
// returns the FULL loss history in one response (no streaming yet -- see
// the note in api_handlers.h), so epochs is capped fairly low.
constexpr int kMaxEpochs = 5000;
constexpr int kMaxHiddenLayers = 6;
constexpr int kMaxLayerWidth = 64;

// Builds the VarMap the parser needs from every top-level numeric field in
// the request body except "expr" (and, if present, merges in a nested
// "vars" object too, so both
//   {"expr": "...", "a": 2.0, "b": 3.0}
// and
//   {"expr": "...", "vars": {"a": 2.0, "b": 3.0}}
// are accepted).
VarMap extract_vars(const json& body) {
    VarMap vars;
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (it.key() == "expr" || it.key() == "vars") continue;
        if (!it.value().is_number()) continue; // silently skip non-numeric extra fields
        vars[it.key()] = make_value(it.value().get<double>(), it.key());
    }
    if (body.contains("vars") && body["vars"].is_object()) {
        for (auto it = body["vars"].begin(); it != body["vars"].end(); ++it) {
            if (!it.value().is_number()) {
                throw std::invalid_argument("vars." + it.key() + " must be a number");
            }
            vars[it.key()] = make_value(it.value().get<double>(), it.key());
        }
    }
    return vars;
}

} // namespace

json handle_compute(const json& body) {
    if (!body.contains("expr") || !body["expr"].is_string()) {
        throw std::invalid_argument("request body must contain a string field \"expr\"");
    }
    std::string expr = body["expr"].get<std::string>();

    VarMap vars = extract_vars(body);

    // Build the graph and run it forward (parsing IS the forward pass --
    // every operator computes ->data immediately as the graph is built)
    // then backward to populate every node's ->grad.
    ValuePtr root = parse_expression(expr, vars);
    root->label = root->label.empty() ? "output" : root->label;
    root->backward();

    json response;
    response["expr"] = expr;
    response["result"] = {
        {"data", root->data},
        {"grad", root->grad}, // dL/dL = 1.0 for the root itself, included for consistency
    };
    response["graph"] = build_graph_json(root);
    return response;
}

json handle_train_xor(const json& body) {
    int epochs = body.value("epochs", 500);
    double learning_rate = body.value("learning_rate", 0.05);
    std::vector<int> hidden_layers = body.value("hidden_layers", std::vector<int>{4, 4});

    if (epochs <= 0 || epochs > kMaxEpochs) {
        throw std::invalid_argument("epochs must be between 1 and " + std::to_string(kMaxEpochs));
    }
    if (learning_rate <= 0.0 || learning_rate > 5.0) {
        throw std::invalid_argument("learning_rate must be between 0 and 5");
    }
    if (hidden_layers.empty() || hidden_layers.size() > static_cast<size_t>(kMaxHiddenLayers)) {
        throw std::invalid_argument("hidden_layers must have between 1 and " + std::to_string(kMaxHiddenLayers) + " entries");
    }
    for (int width : hidden_layers) {
        if (width <= 0 || width > kMaxLayerWidth) {
            throw std::invalid_argument("every hidden layer width must be between 1 and " + std::to_string(kMaxLayerWidth));
        }
    }

    // XOR dataset -- same as train.cpp.
    const std::vector<std::vector<double>> xs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
    };
    const std::vector<double> ys = {0.0, 1.0, 1.0, 0.0};

    // Network: 2 inputs -> the requested hidden layers (tanh) -> 1 linear
    // output neuron, exactly like MLP's documented convention.
    std::vector<int> layer_sizes = hidden_layers;
    layer_sizes.push_back(1);
    MLP mlp(2, layer_sizes);

    std::vector<std::vector<ValuePtr>> xs_values;
    for (const auto& x : xs) {
        std::vector<ValuePtr> row;
        for (double v : x) row.push_back(make_value(v));
        xs_values.push_back(row);
    }

    std::vector<double> losses;
    losses.reserve(static_cast<size_t>(epochs));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<ValuePtr> predictions;
        predictions.reserve(xs_values.size());
        for (const auto& x_values : xs_values) {
            predictions.push_back(mlp.forward(x_values)[0]);
        }

        auto loss = mse_loss(predictions, ys);

        // See train.cpp for why zero_grad() must run before backward()
        // every single epoch.
        mlp.zero_grad();
        loss->backward();

        for (const auto& p : mlp.parameters()) {
            p->data -= learning_rate * p->grad;
        }

        losses.push_back(loss->data);
    }

    json final_predictions = json::array();
    for (size_t i = 0; i < xs_values.size(); ++i) {
        double predicted = mlp.forward(xs_values[i])[0]->data;
        final_predictions.push_back({
            {"input", xs[i]},
            {"predicted", predicted},
            {"target", ys[i]},
        });
    }

    json response;
    response["epochs"] = epochs;
    response["learning_rate"] = learning_rate;
    response["hidden_layers"] = hidden_layers;
    response["losses"] = losses;
    response["final_predictions"] = final_predictions;
    return response;
}
