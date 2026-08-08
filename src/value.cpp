#include "value.h"
#include "autograd_exceptions.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <algorithm>

Value::Value(double data_, std::string label_, std::string op_)
    : data(data_), grad(0.0), label(std::move(label_)), op(std::move(op_)) {
    // Default: no children, no-op backward. Operators below overwrite
    // backward_fn with the real chain-rule logic when they create a node.
    backward_fn = []() {};
}

ValuePtr make_value(double data, const std::string& label) {
    return std::make_shared<Value>(data, label);
}

// ---- operator+ ----
// forward:  out = a + b
// d(out)/da = 1, d(out)/db = 1
// so each parent just receives the SAME grad that "out" has (scaled by 1).
// We use += (not =) when this is wired up in Phase 2's backward(), because
// if a or b is reused elsewhere in the graph, its true gradient is the SUM
// of every contribution flowing back into it -- overwriting with = would
// silently discard earlier contributions. That summing logic lives in the
// node's own grad update, which is why backward_fn below only ever does
// "+=" onto children's grad, never "=".
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data + b->data, "", "+");
    out->children = {a, b};
    // Capture by raw pointer-safe shared_ptr copies so the closure keeps
    // a and b alive as long as out (and this lambda) are alive.
    out->backward_fn = [a, b, out]() {
        a->grad += 1.0 * out->grad;
        b->grad += 1.0 * out->grad;
    };
    return out;
}

// ---- operator* ----
// forward:  out = a * b
// d(out)/da = b, d(out)/db = a   (product rule)
ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data * b->data, "", "*");
    out->children = {a, b};
    out->backward_fn = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

// ---- unary operator- ----
// forward:  out = -a   (equivalent to a * -1)
// d(out)/da = -1
ValuePtr operator-(const ValuePtr& a) {
    auto out = std::make_shared<Value>(-a->data, "", "neg");
    out->children = {a};
    out->backward_fn = [a, out]() {
        a->grad += -1.0 * out->grad;
    };
    return out;
}

// ---- binary operator- ----
// forward:  out = a - b, implemented as a + (-b) so we reuse the
// already-correct gradient rules for + and unary negation instead of
// duplicating chain-rule logic.
ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) {
    return a + (-b);
}

// ---- operator/ ----
// forward:  out = a / b, implemented as a * b^-1 so we reuse pow()'s
// gradient rule below instead of writing a separate quotient rule.
// Guarded against division by zero for numerical stability.
ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
    if (b->data == 0.0) {
        throw autograd::MathError("Value::operator/ - division by zero");
    }
    return a * pow(b, -1.0);
}

// ---- pow ----
// forward:  out = base^exponent   (exponent is a plain double, not a Value,
//           so we don't need to backprop into it)
// d(out)/d(base) = exponent * base^(exponent - 1)   (power rule)
ValuePtr pow(const ValuePtr& base, double exponent) {
    if (base->data == 0.0 && exponent < 0.0) {
        throw autograd::MathError("Value::pow - 0 raised to a negative power");
    }
    double result = std::pow(base->data, exponent);
    auto out = std::make_shared<Value>(result, "", "pow");
    out->children = {base};
    out->backward_fn = [base, exponent, out]() {
        base->grad += exponent * std::pow(base->data, exponent - 1.0) * out->grad;
    };
    return out;
}

// ---- tanh ----
// forward:  t = tanh(x)
// d(t)/dx = 1 - t^2     (standard tanh derivative identity, reusing the
//           already-computed output t instead of recomputing tanh again)
ValuePtr tanh(const ValuePtr& x) {
    double t = std::tanh(x->data);
    auto out = std::make_shared<Value>(t, "", "tanh");
    out->children = {x};
    out->backward_fn = [x, t, out]() {
        x->grad += (1.0 - t * t) * out->grad;
    };
    return out;
}

ValuePtr sin(const ValuePtr& x) {
    double value = std::sin(x->data);
    auto out = std::make_shared<Value>(value, "", "sin");
    out->children = {x};
    out->backward_fn = [x, out]() {
        x->grad += std::cos(x->data) * out->grad;
    };
    return out;
}

ValuePtr cos(const ValuePtr& x) {
    double value = std::cos(x->data);
    auto out = std::make_shared<Value>(value, "", "cos");
    out->children = {x};
    out->backward_fn = [x, out]() {
        x->grad += -std::sin(x->data) * out->grad;
    };
    return out;
}

ValuePtr tan(const ValuePtr& x) {
    double cosine = std::cos(x->data);
    if (std::abs(cosine) < 1e-12) {
        throw autograd::MathError("Value::tan - cosine is zero at the input");
    }
    double value = std::tan(x->data);
    auto out = std::make_shared<Value>(value, "", "tan");
    out->children = {x};
    out->backward_fn = [x, out]() {
        double cosine = std::cos(x->data);
        x->grad += (1.0 / (cosine * cosine)) * out->grad;
    };
    return out;
}

ValuePtr cot(const ValuePtr& x) {
    double sine = std::sin(x->data);
    if (std::abs(sine) < 1e-12) {
        throw autograd::MathError("Value::cot - sine is zero at the input");
    }
    double value = std::cos(x->data) / sine;
    auto out = std::make_shared<Value>(value, "", "cot");
    out->children = {x};
    out->backward_fn = [x, out]() {
        double sine = std::sin(x->data);
        x->grad += -(1.0 / (sine * sine)) * out->grad;
    };
    return out;
}

ValuePtr sec(const ValuePtr& x) {
    double cosine = std::cos(x->data);
    if (std::abs(cosine) < 1e-12) {
        throw autograd::MathError("Value::sec - cosine is zero at the input");
    }
    double value = 1.0 / cosine;
    auto out = std::make_shared<Value>(value, "", "sec");
    out->children = {x};
    out->backward_fn = [x, out]() {
        double cosine = std::cos(x->data);
        x->grad += (std::sin(x->data) / (cosine * cosine)) * out->grad;
    };
    return out;
}

ValuePtr cosec(const ValuePtr& x) {
    double sine = std::sin(x->data);
    if (std::abs(sine) < 1e-12) {
        throw autograd::MathError("Value::cosec - sine is zero at the input");
    }
    double value = 1.0 / sine;
    auto out = std::make_shared<Value>(value, "", "cosec");
    out->children = {x};
    out->backward_fn = [x, out]() {
        double sine = std::sin(x->data);
        x->grad += -(std::cos(x->data) / (sine * sine)) * out->grad;
    };
    return out;
}

// ---- exp ----
// forward:  e = exp(x)
// d(e)/dx = e           (exp is its own derivative)
// Numerical stability: exp(x) overflows to +inf for x beyond ~709 in
// double precision. We clip the INPUT before calling std::exp so the
// forward value stays finite (a large-but-finite number) instead of
// silently becoming inf, which would poison every downstream gradient
// with nan. This is the standard "clip before exp" stabilization.
ValuePtr exp(const ValuePtr& x) {
    constexpr double kExpClip = 700.0; // std::exp(700) is finite; std::exp(710) overflows
    double clipped = std::min(x->data, kExpClip);
    double e = std::exp(clipped);
    auto out = std::make_shared<Value>(e, "", "exp");
    out->children = {x};
    out->backward_fn = [x, e, out]() {
        x->grad += e * out->grad;
    };
    return out;
}

// ---- log ----
// forward:  l = log(x)   (natural log)
// d(l)/dx = 1 / x
// Guarded: log is undefined for x <= 0, and 1/x would divide by zero at
// x == 0, so both cases throw rather than silently producing nan/inf.
ValuePtr log(const ValuePtr& x) {
    if (x->data <= 0.0) {
        throw autograd::MathError("Value::log - input must be strictly positive (got " +
                                   std::to_string(x->data) + ")");
    }
    double l = std::log(x->data);
    auto out = std::make_shared<Value>(l, "", "log");
    out->children = {x};
    out->backward_fn = [x, out]() {
        x->grad += (1.0 / x->data) * out->grad;
    };
    return out;
}

// ---- relu ----
// forward:  r = max(0, x)
// d(r)/dx = 1 if x > 0, else 0   (technically undefined exactly at x==0;
//           by convention we treat it as 0 there, same as most ML frameworks)
ValuePtr relu(const ValuePtr& x) {
    double r = x->data > 0.0 ? x->data : 0.0;
    auto out = std::make_shared<Value>(r, "", "relu");
    out->children = {x};
    out->backward_fn = [x, out]() {
        x->grad += (x->data > 0.0 ? 1.0 : 0.0) * out->grad;
    };
    return out;
}

// ---- backward ----
// Builds a reverse-topological order of the graph rooted at `this`, then
// walks it back-to-front, calling each node's backward_fn.
//
// Why topological order specifically: a node's grad is only complete once
// every parent that used it has already run its backward_fn (see the
// comment on backward_fn in value.h about summing contributions). Visiting
// nodes in reverse-topo order guarantees that by the time we process a
// node, everything "above" it (closer to the loss) has already finished
// contributing to its grad.
void Value::backward() {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;

    // Post-order DFS: recurse into all children FIRST, then append this
    // node. This is exactly what produces a valid topological order (a
    // node's children always end up earlier in `topo` than the node
    // itself), which we then walk in reverse for backprop.
    std::function<void(const ValuePtr&)> build_topo = [&](const ValuePtr& v) {
        if (visited.count(v.get())) return; // already processed -- avoids
                                             // re-visiting a shared child
                                             // (e.g. x used twice) and
                                             // avoids infinite loops
        visited.insert(v.get());
        for (const auto& child : v->children) {
            build_topo(child);
        }
        topo.push_back(v.get());
    };
    build_topo(shared_from_this());

    // dL/dL = 1 by definition -- this is the seed that everything else's
    // gradient is ultimately scaled from via the chain rule.
    grad = 1.0;

    // Walk in reverse topological order: start at the output (this node,
    // which ends up last in `topo`) and move back toward the leaves.
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->backward_fn();
    }
}

// ---- print ----
// Simple indented-text visualization of the graph rooted at this node,
// e.g. for L = (a * b) + c:
//   + (data=..., grad=...)
//     * (data=..., grad=...)
//       a (data=..., grad=...)
//       b (data=..., grad=...)
//     c (data=..., grad=...)
void Value::print(int indent) const {
    std::string pad(static_cast<size_t>(indent) * 2, ' ');
    std::string name = !label.empty() ? label : (!op.empty() ? op : "?");
    std::cout << pad << name
              << " (data=" << data << ", grad=" << grad << ")\n";
    for (const auto& child : children) {
        child->print(indent + 1);
    }
}
