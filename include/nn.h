#pragma once

#include "value.h"
#include <vector>

// A single neuron: computes  activation( sum(weight_i * input_i) + bias ).
// The weights and bias are themselves Values, so once they're combined
// with the (also-Value) inputs using the operators from Phase 1, the
// resulting output Value carries a full computation graph back through
// every weight and bias -- backward() (Phase 2) can then differentiate
// the whole thing with no neuron-specific gradient code required.
class Neuron {
public:
    std::vector<ValuePtr> weights; // one weight per input
    ValuePtr bias;
    bool use_activation; // if true, squash output through tanh; if false, leave it linear

    // n_inputs: how many numbers this neuron accepts.
    // use_activation: hidden-layer neurons use tanh (true); the network's
    //   final output neuron(s) are usually left linear (false) so the
    //   output isn't artificially squeezed into (-1, 1) before the loss
    //   sees it -- see the assumption noted below in MLP.
    Neuron(int n_inputs, bool use_activation_ = true);

    // Runs this neuron on one input vector, returning a single output Value.
    ValuePtr forward(const std::vector<ValuePtr>& inputs) const;

    // All learnable Values owned by this neuron (weights + bias), so the
    // training loop can update and zero-grad them without knowing the
    // network's internal structure.
    std::vector<ValuePtr> parameters() const;
};

// A layer: a set of Neurons, each independently run on the SAME input
// vector, producing one output value each.
class Layer {
public:
    std::vector<Neuron> neurons;

    // n_inputs: size of the input vector each neuron in this layer accepts.
    // n_outputs: how many neurons in this layer (== size of this layer's output vector).
    Layer(int n_inputs, int n_outputs, bool use_activation = true);

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& inputs) const;

    std::vector<ValuePtr> parameters() const;
};

// A Multi-Layer Perceptron: a chain of Layers, each layer's output vector
// feeding directly into the next layer's input vector.
class MLP {
public:
    std::vector<Layer> layers;

    // n_inputs: size of the network's input vector.
    // layer_sizes: number of neurons in each successive layer, e.g. {4, 4, 1}
    //   means: hidden layer of 4, hidden layer of 4, output layer of 1.
    //   Every layer EXCEPT THE LAST uses a tanh activation; the last layer
    //   is left linear. Assumption made here (stated per the project's
    //   ambiguity-handling rule): using a linear output layer, trained
    //   with plain MSE against raw 0/1 targets, is the simplest standard
    //   setup for a regression-style toy classification problem like XOR,
    //   and avoids the output being stuck squashed into (-1, 1) by tanh
    //   right before the loss.
    MLP(int n_inputs, const std::vector<int>& layer_sizes);

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& inputs) const;

    std::vector<ValuePtr> parameters() const;

    // Sets grad = 0 on every parameter in the network. MUST be called
    // before each backward() during training -- see the detailed
    // explanation next to its use in train.cpp for why skipping this
    // silently corrupts training.
    void zero_grad() const;
};

// Mean Squared Error loss over a batch of (prediction, target) pairs:
//   MSE = (1/N) * sum_i (prediction_i - target_i)^2
// Built entirely out of Value arithmetic, so calling backward() on the
// returned Value differentiates the loss all the way back through
// whatever produced `predictions` (i.e. the whole network).
ValuePtr mse_loss(const std::vector<ValuePtr>& predictions,
                   const std::vector<double>& targets);
