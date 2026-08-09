 #include "api_handlers.h"
#include "value.h"
#include "nn.h"
#include "expr_parser.h"
#include "graph_json.h"
#include <httplib.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

constexpr int kMaxEpochs = 5000;
constexpr int kMaxHiddenLayers = 6;
constexpr int kMaxLayerWidth = 64;

VarMap extract_vars(const json& body) {
    VarMap vars;
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (it.key() == "expr" || it.key() == "vars") continue;
        if (!it.value().is_number()) continue;
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

    ValuePtr root = parse_expression(expr, vars);
    root->label = root->label.empty() ? "output" : root->label;
    root->backward();

    json response;
    response["expr"] = expr;
    response["result"] = {
        {"data", root->data},
        {"grad", root->grad},
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

    const std::vector<std::vector<double>> xs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
    };
    const std::vector<double> ys = {0.0, 1.0, 1.0, 0.0};

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

json handle_piegeni(const json& body) {
    if (!body.contains("prompt") || !body["prompt"].is_string()) {
        throw std::invalid_argument("request body must contain a string field \"prompt\"");
    }
    std::string prompt = body["prompt"].get<std::string>();
    if (prompt.empty()) {
        throw std::invalid_argument("\"prompt\" must not be empty");
    }

    const char* api_key_env = std::getenv("PIEGENI_API_KEY");
    if (!api_key_env || std::string(api_key_env).empty()) {
        throw std::runtime_error("server is not configured with PIEGENI_API_KEY");
    }
    std::string api_key = api_key_env;

    httplib::Client cli("https://generativelanguage.googleapis.com");
    cli.set_connection_timeout(20, 0);
    cli.set_read_timeout(30, 0);

    json request_body = {
        {"contents", json::array({
            json{{"parts", json::array({ json{{"text", prompt}} })}}
        })}
    };

    std::string path = "/v1beta/models/gemini-1.5-flash:generateContent?key=" + api_key;
    auto res = cli.Post(path.c_str(), request_body.dump(), "application/json");

    if (!res) {
        throw std::runtime_error("could not reach the Gemini API (network error)");
    }
    if (res->status != 200) {
        throw std::runtime_error("Gemini API returned status " + std::to_string(res->status));
    }

    json gemini_response;
    try {
        gemini_response = json::parse(res->body);
    } catch (const json::parse_error&) {
        throw std::runtime_error("Gemini API returned a non-JSON response");
    }

    if (!gemini_response.contains("candidates") || gemini_response["candidates"].empty()) {
        throw std::runtime_error("Gemini API response did not contain any answer");
    }

    std::string answer;
    try {
        answer = gemini_response["candidates"][0]["content"]["parts"][0]["text"].get<std::string>();
    } catch (const json::exception&) {
        throw std::runtime_error("could not parse the answer out of the Gemini API response");
    }

    json response;
    response["answer"] = answer;
    return response;
}
