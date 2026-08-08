#pragma once

#include <stdexcept>
#include <string>

// Custom exception hierarchy for the autograd engine.
//
// Every exception derives from std::runtime_error (transitively from
// std::exception), so existing code that catches std::exception --
// notably backend/src/server.cpp's handle_json_route, which turns any
// caught exception into an HTTP 400 with {"error": e.what()} -- keeps
// working unmodified. The hierarchy exists so callers who WANT to can
// catch more specifically (e.g. "retry on DatasetError but not on
// MathError"), while callers who don't care can still just catch
// autograd::AutogradError or std::exception.
namespace autograd {

// Base class for every exception thrown by this project's own code
// (as opposed to exceptions thrown by the standard library or a
// third-party dependency).
class AutogradError : public std::runtime_error {
public:
    explicit AutogradError(const std::string& msg) : std::runtime_error(msg) {}
};

// A mathematically invalid operation on a Value: log of a non-positive
// number, division by zero, 0 raised to a negative power, etc. Thrown
// instead of letting the operation silently produce nan/inf, which
// would otherwise poison every downstream gradient without any clear
// error message pointing at the cause.
class MathError : public AutogradError {
public:
    explicit MathError(const std::string& msg) : AutogradError(msg) {}
};

// A shape/size mismatch: e.g. a Neuron given an input vector whose
// length doesn't match its weight count, or mse_loss given predictions
// and targets of different lengths.
class DimensionMismatchError : public AutogradError {
public:
    explicit DimensionMismatchError(const std::string& msg) : AutogradError(msg) {}
};

// Anything wrong with an on-disk dataset: missing file, empty file,
// a row with the wrong number of columns, a cell that isn't a valid
// number, etc.
class DatasetError : public AutogradError {
public:
    explicit DatasetError(const std::string& msg) : AutogradError(msg) {}
};

// Anything wrong with saving/loading a model: missing file, malformed
// or incomplete JSON, a weight count that doesn't match the declared
// architecture, etc.
class SerializationError : public AutogradError {
public:
    explicit SerializationError(const std::string& msg) : AutogradError(msg) {}
};

} // namespace autograd
