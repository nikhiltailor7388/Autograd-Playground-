#pragma once

#include <memory>
#include <functional>
#include <string>
#include <vector>

// A Value is a single scalar number that also remembers HOW it was
// computed (which other Values were combined, and with what operation).
// Stringing many Values together via +, -, *, / builds a computation
// graph automatically, just by writing normal-looking arithmetic.
class Value : public std::enable_shared_from_this<Value> {
public:
    double data;                              // the actual scalar number
    double grad;                              // dL/d(this) -- 0 until backward() is called on some downstream node
    std::string label;                        // human-readable name, e.g. "a", for debugging/printing
    std::string op;                           // which operation produced this node, e.g. "+", "*", "" if a leaf

    // The Values that were combined to produce this one (this node's
    // "inputs"). shared_ptr is used because a single Value can be a
    // child of MORE THAN ONE parent (e.g. in x + x, the same node x
    // is a child of the "+" node twice) -- ownership is shared, not
    // exclusive, so a raw pointer or unique_ptr would be wrong here.
    std::vector<std::shared_ptr<Value>> children;

    // How to push this node's grad backward onto its children, using
    // the chain rule. Each operator (+, *, ...) fills this in with the
    // correct local derivative for that operation. It does nothing by
    // default (leaf nodes have no children to propagate to).
    // Every implementation of this ALWAYS uses += onto a child's grad,
    // never =. Reason: if a child Value is reused in multiple places in
    // the expression (e.g. x + x, or a weight shared by several
    // neurons), it will appear as a child of more than one parent node,
    // and backward_fn will run once per parent. Its true gradient is the
    // SUM of every one of those contributions (multivariable chain
    // rule). Overwriting with = would silently keep only the LAST
    // contribution and discard the rest -- this is the single most
    // common bug in hand-written autograd engines, so it's called out
    // explicitly here and in every operator below.
    std::function<void()> backward_fn;

    explicit Value(double data_, std::string label_ = "", std::string op_ = "");

    // Prints the graph rooted at this node as indented text, so you can
    // see the tree structure of an expression like (a * b) + c.
    void print(int indent = 0) const;

    // Runs backpropagation with THIS node as the final output (e.g. the
    // loss). Sets this node's own grad to 1.0 (dL/dL = 1 by definition),
    // computes a reverse-topological order of the whole graph reachable
    // from this node, then walks that order back-to-front calling each
    // node's backward_fn so gradients flow from this node down to every
    // leaf that contributed to it. Must be called on a Value that is
    // itself managed by a shared_ptr (all Values created via make_value
    // or any operator satisfy this).
    void backward();
};

using ValuePtr = std::shared_ptr<Value>;

// Convenience factory so callers don't have to spell out
// std::make_shared<Value> everywhere.
ValuePtr make_value(double data, const std::string& label = "");

// Binary and unary operators. Each one:
//   1. computes the forward result (the actual arithmetic),
//   2. creates a new Value node recording its inputs (a, b) and op label,
//   3. records how gradients should flow back to a and b (used in Phase 2).
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, const ValuePtr& b); // binary subtraction: a - b
ValuePtr operator-(const ValuePtr& a);                    // unary negation: -a
ValuePtr operator/(const ValuePtr& a, const ValuePtr& b);

// Raises a Value to a fixed (constant, non-Value) exponent, e.g. pow(x, 2.0) for x^2.
ValuePtr pow(const ValuePtr& base, double exponent);

// Activation / elementwise math functions, each an autograd-aware node
// just like the operators above: forward computes the value, backward_fn
// records the correct local derivative for the chain rule.
ValuePtr tanh(const ValuePtr& x);
ValuePtr sin(const ValuePtr& x);
ValuePtr cos(const ValuePtr& x);
ValuePtr tan(const ValuePtr& x);
ValuePtr cot(const ValuePtr& x);
ValuePtr sec(const ValuePtr& x);
ValuePtr cosec(const ValuePtr& x);
ValuePtr exp(const ValuePtr& x);
ValuePtr log(const ValuePtr& x);
ValuePtr relu(const ValuePtr& x);
