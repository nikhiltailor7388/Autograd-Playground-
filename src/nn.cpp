#include "nn.h"
#include "autograd_exceptions.h"
#include <random>
#include <stdexcept>

namespace {
// A single RNG shared by all weight/bias initialization, seeded with a
// FIXED value (not std::random_device) so that runs are reproducible --
// useful while debugging a training run ("did my change actually help,
// or did I just get a luckier random init this time?"). Swap in
// std::random_device{}() here if true non-reproducible randomness is
// ever wanted instead.
std::mt19937 rng(42);
std::uniform_real_distribution<double> uniform(-1.0, 1.0);

double random_weight() { return uniform(rng); }
} // namespace

// ---- Neuron ----

Neuron::Neuron(int n_inputs, bool use_activation_) : use_activation(use_activation_) {
    weights.reserve(static_cast<size_t>(n_inputs));
    for (int i = 0; i < n_inputs; ++i) {
        weights.push_back(make_value(random_weight(), "w"));
    }
    bias = make_value(0.0, "b"); // bias starts at 0; weights carry the randomness needed to break symmetry
}

ValuePtr Neuron::forward(const std::vector<ValuePtr>& inputs) const {
    if (inputs.size() != weights.size()) {
        throw autograd::DimensionMismatchError(
            "Neuron::forward - input size (" + std::to_string(inputs.size()) +
            ") does not match number of weights (" + std::to_string(weights.size()) + ")");
    }
    // sum = bias + w0*x0 + w1*x1 + ...
    ValuePtr sum = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum = sum + weights[i] * inputs[i];
    }
    return use_activation ? tanh(sum) : sum;
}

std::vector<ValuePtr> Neuron::parameters() const {
    std::vector<ValuePtr> params = weights;
    params.push_back(bias);
    return params;
}

// ---- Layer ----

Layer::Layer(int n_inputs, int n_outputs, bool use_activation) {
    neurons.reserve(static_cast<size_t>(n_outputs));
    for (int i = 0; i < n_outputs; ++i) {
        neurons.emplace_back(n_inputs, use_activation);
    }
}

std::vector<ValuePtr> Layer::forward(const std::vector<ValuePtr>& inputs) const {
    std::vector<ValuePtr> outputs;
    outputs.reserve(neurons.size());
    for (const auto& neuron : neurons) {
        outputs.push_back(neuron.forward(inputs));
    }
    return outputs;
}

std::vector<ValuePtr> Layer::parameters() const {
    std::vector<ValuePtr> params;
    for (const auto& neuron : neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

// ---- MLP ----

MLP::MLP(int n_inputs, const std::vector<int>& layer_sizes) {
    int prev_size = n_inputs;
    for (size_t i = 0; i < layer_sizes.size(); ++i) {
        bool is_last_layer = (i == layer_sizes.size() - 1);
        // Every layer except the last is non-linear (tanh); the last
        // layer is linear -- see the rationale in nn.h.
        layers.emplace_back(prev_size, layer_sizes[i], /*use_activation=*/!is_last_layer);
        prev_size = layer_sizes[i];
    }
}

std::vector<ValuePtr> MLP::forward(const std::vector<ValuePtr>& inputs) const {
    std::vector<ValuePtr> current = inputs;
    for (const auto& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

std::vector<ValuePtr> MLP::parameters() const {
    std::vector<ValuePtr> params;
    for (const auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void MLP::zero_grad() const {
    for (const auto& p : parameters()) {
        p->grad = 0.0;
    }
}

// ---- MSE loss ----

ValuePtr mse_loss(const std::vector<ValuePtr>& predictions,
                   const std::vector<double>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) {
        throw autograd::DimensionMismatchError(
            "mse_loss - predictions/targets size mismatch (" +
            std::to_string(predictions.size()) + " vs " + std::to_string(targets.size()) +
            ") or empty input");
    }
    ValuePtr total = make_value(0.0);
    for (size_t i = 0; i < predictions.size(); ++i) {
        auto target = make_value(targets[i]);
        auto diff = predictions[i] - target;
        total = total + pow(diff, 2.0);
    }
    auto n = make_value(static_cast<double>(predictions.size()));
    return total / n;
}
