#include "model_io.h"
#include "autograd_exceptions.h"

#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

namespace autograd {

void save_model(const MLP& mlp, const std::string& path) {
    if (mlp.layers.empty() || mlp.layers.front().neurons.empty()) {
        throw SerializationError("save_model - cannot save an empty MLP");
    }

    json root;
    root["n_inputs"] = static_cast<int>(mlp.layers.front().neurons.front().weights.size());

    json layer_sizes = json::array();
    json layers_json = json::array();

    for (const auto& layer : mlp.layers) {
        layer_sizes.push_back(static_cast<int>(layer.neurons.size()));

        json neurons_json = json::array();
        for (const auto& neuron : layer.neurons) {
            json weights_json = json::array();
            for (const auto& w : neuron.weights) {
                weights_json.push_back(w->data);
            }
            neurons_json.push_back(json{
                {"weights", weights_json},
                {"bias", neuron.bias->data},
            });
        }
        layers_json.push_back(json{
            {"use_activation", layer.neurons.front().use_activation},
            {"neurons", neurons_json},
        });
    }

    root["layer_sizes"] = layer_sizes;
    root["layers"] = layers_json;

    std::ofstream out(path);
    if (!out.is_open()) {
        throw SerializationError("save_model - could not open file for writing: " + path);
    }
    // 2-space indent: readable enough to diff/inspect by hand, still compact.
    out << root.dump(2);
}

MLP load_model(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw SerializationError("load_model - could not open file: " + path);
    }

    json root;
    try {
        in >> root;
    } catch (const json::parse_error& e) {
        throw SerializationError(std::string("load_model - malformed JSON: ") + e.what());
    }

    if (!root.contains("n_inputs") || !root.contains("layer_sizes") || !root.contains("layers")) {
        throw SerializationError(
            "load_model - JSON missing one of the required fields: "
            "n_inputs, layer_sizes, layers");
    }

    int n_inputs = root.at("n_inputs").get<int>();
    std::vector<int> layer_sizes = root.at("layer_sizes").get<std::vector<int>>();
    const json& layers_json = root.at("layers");

    if (layer_sizes.empty() || layers_json.size() != layer_sizes.size()) {
        throw SerializationError(
            "load_model - layer_sizes/layers length mismatch or empty architecture");
    }

    // Reconstruct the MLP's shape first (with fresh random init), then
    // overwrite every weight and bias below with the saved values. This
    // reuses the MLP constructor's existing activation-assignment rule
    // (every layer but the last uses tanh) instead of duplicating it here.
    MLP mlp(n_inputs, layer_sizes);

    for (size_t li = 0; li < mlp.layers.size(); ++li) {
        auto& layer = mlp.layers[li];
        const json& layer_json = layers_json.at(li);

        if (!layer_json.contains("neurons")) {
            throw SerializationError("load_model - layer " + std::to_string(li) +
                                      " is missing its 'neurons' field");
        }
        const json& neurons_json = layer_json.at("neurons");
        if (neurons_json.size() != layer.neurons.size()) {
            throw SerializationError(
                "load_model - layer " + std::to_string(li) + " has " +
                std::to_string(neurons_json.size()) + " saved neurons, expected " +
                std::to_string(layer.neurons.size()) + " from layer_sizes");
        }

        bool expected_activation = layer_json.value("use_activation", layer.neurons.front().use_activation);
        if (expected_activation != layer.neurons.front().use_activation) {
            throw SerializationError(
                "load_model - layer " + std::to_string(li) +
                " saved activation flag does not match this architecture's expected rule "
                "(every layer but the last uses tanh)");
        }

        for (size_t ni = 0; ni < layer.neurons.size(); ++ni) {
            auto& neuron = layer.neurons[ni];
            const json& neuron_json = neurons_json.at(ni);

            if (!neuron_json.contains("weights") || !neuron_json.contains("bias")) {
                throw SerializationError("load_model - layer " + std::to_string(li) +
                                          " neuron " + std::to_string(ni) +
                                          " is missing 'weights' or 'bias'");
            }
            std::vector<double> saved_weights = neuron_json.at("weights").get<std::vector<double>>();
            if (saved_weights.size() != neuron.weights.size()) {
                throw SerializationError(
                    "load_model - layer " + std::to_string(li) + " neuron " + std::to_string(ni) +
                    " has " + std::to_string(saved_weights.size()) + " saved weights, expected " +
                    std::to_string(neuron.weights.size()));
            }
            for (size_t wi = 0; wi < saved_weights.size(); ++wi) {
                neuron.weights[wi]->data = saved_weights[wi];
            }
            neuron.bias->data = neuron_json.at("bias").get<double>();
        }
    }

    return mlp;
}

} // namespace autograd
