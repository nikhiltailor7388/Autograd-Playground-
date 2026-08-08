// Tests for Phase 7's advanced features:
//   1. Custom exceptions are thrown (not std::domain_error/etc, and not
//      a silent nan) for invalid math and mismatched shapes.
//   2. The CSV loader correctly parses a well-formed file and correctly
//      rejects malformed ones.
//   3. A trained-ish MLP's weights survive a save_model -> load_model
//      round trip bit-for-bit (same predictions on the same inputs).
//
// Uses the same lightweight "count passes, print PASS/FAIL rows, return
// nonzero on any failure" pattern as tests/grad_check.cpp so both plug
// into CTest identically (see root CMakeLists.txt's add_test calls).

#include "value.h"
#include "nn.h"
#include "csv_loader.h"
#include "model_io.h"
#include "autograd_exceptions.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int g_checks = 0;
int g_passed = 0;

std::string test_path(const char* name) {
    const char* temp = std::getenv("TEMP");
    if (!temp) temp = std::getenv("TMP");
    if (!temp) temp = ".";
    return std::string(temp) + "/" + name;
}

void expect(bool condition, const std::string& description) {
    ++g_checks;
    std::cout << (condition ? "PASS  " : "FAIL  ") << description << "\n";
    if (condition) ++g_passed;
}

template <typename ExceptionT, typename Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const ExceptionT&) {
        return true;
    } catch (...) {
        return false; // wrong exception type
    }
    return false; // didn't throw at all
}

void test_exceptions() {
    std::cout << "-- custom exceptions --\n";
    expect(throws<autograd::MathError>([] { auto x = make_value(-1.0); log(x); }),
           "log(-1) throws autograd::MathError");
    expect(throws<autograd::MathError>([] { auto a = make_value(1.0); auto b = make_value(0.0); a / b; }),
           "division by zero throws autograd::MathError");
    expect(throws<autograd::DimensionMismatchError>([] {
               Neuron n(3);
               n.forward({make_value(1.0), make_value(2.0)}); // wrong size: 2 inputs for a 3-weight neuron
           }),
           "Neuron::forward with wrong input size throws autograd::DimensionMismatchError");
    expect(throws<autograd::DimensionMismatchError>([] {
               mse_loss({make_value(1.0)}, {1.0, 2.0}); // size mismatch
           }),
           "mse_loss with mismatched sizes throws autograd::DimensionMismatchError");
    // Every custom exception must still be catchable as std::exception,
    // since backend/src/server.cpp relies on exactly that to turn any
    // thrown error into an HTTP 400 instead of crashing.
    expect(throws<std::exception>([] { auto x = make_value(0.0); log(x); }),
           "autograd::MathError is still catchable as std::exception");
}

void test_csv_loader() {
    std::cout << "-- CSV loader --\n";

    const std::string good_path = test_path("autograd_test_good.csv");
    {
        std::ofstream f(good_path);
        f << "x1,x2,label\n";
        f << "0.0,0.0,0\n";
        f << "0.0,1.0,1\n";
        f << "1.0,0.0,1\n";
        f << "1.0,1.0,0\n";
    }
    auto ds = autograd::load_csv(good_path, /*has_header=*/true);
    expect(ds.inputs.size() == 4, "load_csv reads correct number of data rows");
    expect(ds.feature_names.size() == 2 && ds.feature_names[0] == "x1" && ds.feature_names[1] == "x2",
           "load_csv captures feature names from header, excluding target column");
    expect(ds.inputs[1].size() == 2 && ds.inputs[1][0] == 0.0 && ds.inputs[1][1] == 1.0,
           "load_csv parses feature values correctly");
    expect(ds.targets[2] == 1.0, "load_csv parses target values correctly");

    const std::string ragged_path = test_path("autograd_test_ragged.csv");
    {
        std::ofstream f(ragged_path);
        f << "x1,x2,label\n";
        f << "0.0,0.0,0\n";
        f << "1.0,1\n"; // wrong column count
    }
    expect(throws<autograd::DatasetError>([&] { autograd::load_csv(ragged_path, true); }),
           "load_csv throws autograd::DatasetError on a ragged row");

    const std::string bad_number_path = test_path("autograd_test_bad_number.csv");
    {
        std::ofstream f(bad_number_path);
        f << "x1,x2,label\n";
        f << "0.0,notanumber,0\n";
    }
    expect(throws<autograd::DatasetError>([&] { autograd::load_csv(bad_number_path, true); }),
           "load_csv throws autograd::DatasetError on a non-numeric cell");

    expect(throws<autograd::DatasetError>([] { autograd::load_csv(test_path("does_not_exist_12345.csv")); }),
           "load_csv throws autograd::DatasetError on a missing file");
}

void test_model_io() {
    std::cout << "-- model serialization --\n";

    MLP mlp(2, {4, 4, 1});
    // Nudge weights away from their initial random values so a
    // round-trip test can't accidentally pass just because save/load
    // both happen to leave everything at its untouched initial state.
    for (const auto& p : mlp.parameters()) p->data += 0.1234;

    std::vector<ValuePtr> probe = {make_value(0.3), make_value(0.7)};
    double original_output = mlp.forward(probe)[0]->data;

    const std::string model_path = test_path("autograd_test_model.json");
    autograd::save_model(mlp, model_path);

    MLP loaded = autograd::load_model(model_path);
    double loaded_output = loaded.forward(probe)[0]->data;

    expect(std::abs(original_output - loaded_output) < 1e-12,
           "save_model -> load_model round trip reproduces identical predictions");
    expect(loaded.layers.size() == mlp.layers.size(), "loaded MLP has the same number of layers");
    expect(loaded.layers[0].neurons.size() == mlp.layers[0].neurons.size(),
           "loaded MLP's first layer has the same number of neurons");

    expect(throws<autograd::SerializationError>([] { autograd::load_model(test_path("does_not_exist_98765.json")); }),
           "load_model throws autograd::SerializationError on a missing file");

    const std::string malformed_path = test_path("autograd_test_malformed_model.json");
    {
        std::ofstream f(malformed_path);
        f << "{ \"n_inputs\": 2 }"; // missing layer_sizes/layers
    }
    expect(throws<autograd::SerializationError>([&] { autograd::load_model(malformed_path); }),
           "load_model throws autograd::SerializationError on incomplete JSON");
}

} // namespace

int main() {
    test_exceptions();
    test_csv_loader();
    test_model_io();

    std::cout << "\n" << g_passed << " / " << g_checks << " checks passed\n";
    return (g_passed == g_checks) ? 0 : 1;
}
