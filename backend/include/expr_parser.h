#pragma once

#include "value.h"
#include <string>
#include <unordered_map>
#include <stdexcept>

// Maps variable names used in an expression string (e.g. "a", "b") to the
// actual Value leaf nodes the caller already created for them. The parser
// looks names up here rather than inventing new leaves, so the ValuePtrs
// the caller holds (and can later read ->grad off of after backward())
// are the SAME nodes woven into the parsed graph.
using VarMap = std::unordered_map<std::string, ValuePtr>;

// Thrown for any syntax error, unknown variable, or unsupported construct
// (e.g. a non-constant exponent) encountered while parsing.
class ExprParseError : public std::runtime_error {
public:
    explicit ExprParseError(const std::string& msg) : std::runtime_error(msg) {}
};

// Parses a scalar arithmetic expression string into a Value computation
// graph and returns its root (output) node.
//
// Supported grammar (informal):
//   expr    := term (('+' | '-') term)*
//   term    := unary (('*' | '/') unary)*
//   unary   := '-' unary | power
//   power   := primary ('^' NUMBER)?      -- exponent must be a numeric
//                                             literal, since Value::pow()
//                                             only supports a constant
//                                             (non-Value) exponent
//   primary := NUMBER
//            | IDENT                      -- looked up in `vars`
//            | FUNC '(' expr ')'          -- FUNC: sin, cos, tan, cot, sec,
//                                             cosec, tanh, exp, log, relu
//            | '(' expr ')'
//
// Examples: "(a * b) + a", "tanh(a) + b / 2", "-(a - b)^2"
ValuePtr parse_expression(const std::string& expr, const VarMap& vars);
