// Gradient checker: for every operation, compares the ANALYTICAL gradient
// (produced by Value::backward(), i.e. the chain-rule code we hand-wrote)
// against a NUMERICAL gradient computed independently via the finite
// difference formula:
//
//     f'(x) ~= (f(x + h) - f(x - h)) / (2h)
//
// The numerical version never uses our backward() logic at all -- it just
// evaluates the plain-double forward math twice at nearby points. If our
// analytical gradient is wrong (e.g. a sign flip, a forgotten chain-rule
// factor, an accidental "=" instead of "+="), the two numbers will disagree
// well beyond floating-point noise, and this test catches it immediately.

#include "value.h"
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CheckResult {
    std::string name;
    double analytical;
    double numerical;
    bool passed;
};

constexpr double kTolerance = 1e-4;
constexpr double kH = 1e-5; // finite-difference step size

double numerical_derivative(const std::function<double(double)>& f, double x) {
    return (f(x + kH) - f(x - kH)) / (2.0 * kH);
}

// Checks d(out)/d(x) for a single-input operation.
//   raw_f       : the plain-double version of the same math, e.g. [](double x){ return std::tanh(x); }
//   build_graph : builds the Value graph for the same op, e.g. [](ValuePtr x){ return tanh(x); }
CheckResult check_unary(const std::string& name,
                         const std::function<double(double)>& raw_f,
                         const std::function<ValuePtr(const ValuePtr&)>& build_graph,
                         double x0) {
    auto x = make_value(x0);
    auto out = build_graph(x);
    out->backward();

    double analytical = x->grad;
    double numerical = numerical_derivative(raw_f, x0);
    bool passed = std::fabs(analytical - numerical) < kTolerance;
    return {name, analytical, numerical, passed};
}

// Checks d(out)/d(a) AND d(out)/d(b) for a two-input operation, returning
// one CheckResult per input.
//   raw_f       : plain-double version, e.g. [](double a, double b){ return a * b; }
//   build_graph : Value-graph version, e.g. [](ValuePtr a, ValuePtr b){ return a * b; }
std::vector<CheckResult> check_binary(
    const std::string& name,
    const std::function<double(double, double)>& raw_f,
    const std::function<ValuePtr(const ValuePtr&, const ValuePtr&)>& build_graph,
    double a0, double b0) {
    auto a = make_value(a0);
    auto b = make_value(b0);
    auto out = build_graph(a, b);
    out->backward();

    double analytical_a = a->grad;
    double analytical_b = b->grad;

    double numerical_a = numerical_derivative(
        [&](double av) { return raw_f(av, b0); }, a0);
    double numerical_b = numerical_derivative(
        [&](double bv) { return raw_f(a0, bv); }, b0);

    bool passed_a = std::fabs(analytical_a - numerical_a) < kTolerance;
    bool passed_b = std::fabs(analytical_b - numerical_b) < kTolerance;

    return {
        {name + " (d/da)", analytical_a, numerical_a, passed_a},
        {name + " (d/db)", analytical_b, numerical_b, passed_b},
    };
}

void print_row(const CheckResult& r) {
    std::cout << std::left << std::setw(16) << r.name
               << std::right << std::setw(14) << std::fixed << std::setprecision(6) << r.analytical
               << std::setw(14) << r.numerical
               << "   " << (r.passed ? "PASS" : "FAIL")
               << "\n";
}

} // namespace

int main() {
    std::vector<CheckResult> results;

    // +
    auto add_results = check_binary(
        "add",
        [](double a, double b) { return a + b; },
        [](const ValuePtr& a, const ValuePtr& b) { return a + b; },
        2.0, 3.0);
    results.insert(results.end(), add_results.begin(), add_results.end());

    // -  (binary)
    auto sub_results = check_binary(
        "sub",
        [](double a, double b) { return a - b; },
        [](const ValuePtr& a, const ValuePtr& b) { return a - b; },
        5.0, 2.0);
    results.insert(results.end(), sub_results.begin(), sub_results.end());

    // *
    auto mul_results = check_binary(
        "mul",
        [](double a, double b) { return a * b; },
        [](const ValuePtr& a, const ValuePtr& b) { return a * b; },
        2.0, -3.0);
    results.insert(results.end(), mul_results.begin(), mul_results.end());

    // /
    auto div_results = check_binary(
        "div",
        [](double a, double b) { return a / b; },
        [](const ValuePtr& a, const ValuePtr& b) { return a / b; },
        6.0, 2.0);
    results.insert(results.end(), div_results.begin(), div_results.end());

    // unary -
    results.push_back(check_unary(
        "neg",
        [](double x) { return -x; },
        [](const ValuePtr& x) { return -x; },
        4.0));

    // pow (exponent is a constant, not a Value -- only check d/d(base))
    results.push_back(check_unary(
        "pow (^3)",
        [](double x) { return std::pow(x, 3.0); },
        [](const ValuePtr& x) { return pow(x, 3.0); },
        2.0));

    // tanh
    results.push_back(check_unary(
        "tanh",
        [](double x) { return std::tanh(x); },
        [](const ValuePtr& x) { return tanh(x); },
        0.5));

    results.push_back(check_unary(
        "sin",
        [](double x) { return std::sin(x); },
        [](const ValuePtr& x) { return sin(x); },
        0.7));

    results.push_back(check_unary(
        "cos",
        [](double x) { return std::cos(x); },
        [](const ValuePtr& x) { return cos(x); },
        0.7));

    results.push_back(check_unary(
        "tan",
        [](double x) { return std::tan(x); },
        [](const ValuePtr& x) { return tan(x); },
        0.4));

    results.push_back(check_unary(
        "cot",
        [](double x) { return std::cos(x) / std::sin(x); },
        [](const ValuePtr& x) { return cot(x); },
        0.7));

    results.push_back(check_unary(
        "sec",
        [](double x) { return 1.0 / std::cos(x); },
        [](const ValuePtr& x) { return sec(x); },
        0.4));

    results.push_back(check_unary(
        "cosec",
        [](double x) { return 1.0 / std::sin(x); },
        [](const ValuePtr& x) { return cosec(x); },
        0.7));

    // exp
    results.push_back(check_unary(
        "exp",
        [](double x) { return std::exp(x); },
        [](const ValuePtr& x) { return exp(x); },
        1.0));

    // log
    results.push_back(check_unary(
        "log",
        [](double x) { return std::log(x); },
        [](const ValuePtr& x) { return log(x); },
        2.0));

    // relu (evaluated at a point away from the x=0 kink, where the
    // derivative is well-defined)
    results.push_back(check_unary(
        "relu",
        [](double x) { return x > 0.0 ? x : 0.0; },
        [](const ValuePtr& x) { return relu(x); },
        0.7));

    std::cout << std::left << std::setw(16) << "op"
               << std::right << std::setw(14) << "analytical"
               << std::setw(14) << "numerical"
               << "   result\n";
    std::cout << std::string(50, '-') << "\n";

    int num_passed = 0;
    for (const auto& r : results) {
        print_row(r);
        if (r.passed) ++num_passed;
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << num_passed << " / " << results.size() << " checks passed\n";

    return (num_passed == static_cast<int>(results.size())) ? 0 : 1;
}
