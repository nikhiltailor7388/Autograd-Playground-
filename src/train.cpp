#include "nn.h"
#include "value.h"
#include "csv_loader.h"
#include "model_io.h"
#include "autograd_exceptions.h"
#include <iostream>
#include <vector>

// Usage:
//   train                          -- trains on the built-in XOR dataset
//   train data.csv                 -- trains on a CSV file instead (last
//                                      column = target, rest = features;
//                                      see include/csv_loader.h for format)
//   train data.csv model_out.json  -- also saves the trained weights
int main(int argc, char** argv) {
    std::vector<std::vector<double>> xs;
    std::vector<double> ys;
    std::string save_path;

    if (argc > 1) {
        // Custom dataset path: load real tabular data via the CSV loader
        // instead of the hardcoded XOR truth table below.
        try {
            autograd::Dataset dataset = autograd::load_csv(argv[1]);
            xs = std::move(dataset.inputs);
            ys = std::move(dataset.targets);
            std::cout << "Loaded " << xs.size() << " rows from " << argv[1] << "\n";
        } catch (const autograd::DatasetError& e) {
            std::cerr << "Failed to load dataset: " << e.what() << "\n";
            return 1;
        }
        if (argc > 2) save_path = argv[2];
    } else {
        // XOR dataset: 4 examples, 2 inputs each, 1 target output.
        // XOR is the classic "smallest problem that needs a hidden layer"
        // -- a single linear neuron (no hidden layer) provably cannot
        // separate these four points, so successfully training this is a
        // real test that backprop through multiple layers is working
        // correctly.
        xs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0},
        };
        ys = {0.0, 1.0, 1.0, 0.0};
    }

    const int n_inputs = static_cast<int>(xs.front().size());

    // Network: n_inputs -> hidden layer of 4 (tanh) -> hidden layer of 4
    // (tanh) -> output layer of 1 (linear, per the MLP constructor's rule
    // that the last layer has no activation).
    MLP mlp(n_inputs, {4, 4, 1});

    // Wrap the raw dataset in Value leaves ONCE, outside the training
    // loop. Their .data never changes across epochs (it's fixed input
    // data, not a learnable parameter), so there's no need to recreate
    // them every epoch -- only the network's weights change, and a fresh
    // computation graph is naturally built every forward() call anyway
    // since every operator creates new Value nodes.
    std::vector<std::vector<ValuePtr>> xs_values;
    for (const auto& x : xs) {
        std::vector<ValuePtr> row;
        for (double v : x) row.push_back(make_value(v));
        xs_values.push_back(row);
    }

    const double learning_rate = 0.05;
    const int epochs = 2000;
    const int print_every = 100;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 1) FORWARD PASS: run every example through the network to get predictions.
        std::vector<ValuePtr> predictions;
        for (const auto& x_values : xs_values) {
            auto out = mlp.forward(x_values); // vector of size 1 (one output neuron)
            predictions.push_back(out[0]);
        }

        // 2) COMPUTE LOSS: how far off are the predictions, on average (squared)?
        auto loss = mse_loss(predictions, ys);

        // 3) ZERO GRADIENTS -- BEFORE backward(), not after.
        // This is critical, and here's exactly why: every Value's grad is
        // accumulated with += (Phase 1/2), specifically so that a
        // parameter used multiple times in one forward pass gets the SUM
        // of all its gradient contributions. But that same += means that
        // if we DON'T reset grad to 0 before this epoch's backward() call,
        // this epoch's gradient contributions get added ON TOP OF last
        // epoch's leftover gradient values instead of replacing them.
        // The result: every parameter's effective "gradient" grows every
        // single epoch regardless of the actual current loss landscape,
        // gradient descent steps get larger and larger, and training
        // diverges (loss explodes to nan/inf) within a handful of epochs.
        // zero_grad() must run once per epoch, after using last epoch's
        // gradients (i.e. after the previous update) and before this
        // epoch's backward().
        mlp.zero_grad();

        // 4) BACKWARD PASS: compute dLoss/d(param) for every parameter.
        loss->backward();

        // 5) GRADIENT DESCENT UPDATE: nudge every parameter a small step
        // in the direction that REDUCES the loss (i.e. opposite its
        // gradient, hence the minus sign).
        for (const auto& p : mlp.parameters()) {
            p->data -= learning_rate * p->grad;
        }

        if (epoch % print_every == 0 || epoch == epochs - 1) {
            std::cout << "epoch " << epoch << "  loss = " << loss->data << "\n";
        }
    }

    std::cout << "\nFinal predictions vs targets:\n";
    for (size_t i = 0; i < xs_values.size(); ++i) {
        auto out = mlp.forward(xs_values[i]);
        std::cout << "  input=(";
        for (size_t j = 0; j < xs[i].size(); ++j) {
            std::cout << xs[i][j] << (j + 1 < xs[i].size() ? ", " : "");
        }
        std::cout << ")  predicted=" << out[0]->data << "  target=" << ys[i] << "\n";
    }

    if (!save_path.empty()) {
        try {
            autograd::save_model(mlp, save_path);
            std::cout << "\nSaved trained model to " << save_path << "\n";
        } catch (const autograd::SerializationError& e) {
            std::cerr << "Failed to save model: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}
