#pragma once

#include <string>
#include <vector>

// A lightweight CSV dataset loader so the MLP can train on arbitrary
// tabular data, not just the hardcoded XOR truth table.
//
// Format assumptions (kept deliberately simple -- see README's "Known
// Limitations"): comma-delimited, one row per line, every column except
// the last is a numeric input feature, the last column is the numeric
// target/label. No support for quoted fields containing embedded commas.
namespace autograd {

struct Dataset {
    std::vector<std::vector<double>> inputs;  // one row per example, one entry per feature
    std::vector<double> targets;              // one entry per example (last CSV column)
    std::vector<std::string> feature_names;   // from the header row, if has_header was true
                                               // (excludes the target column's name); empty otherwise
};

// Reads and parses a CSV file.
//   path       : filesystem path to the .csv file.
//   has_header : if true, the first line is treated as column names
//                (stored in Dataset::feature_names, last name dropped
//                since it names the target column) rather than data.
//
// Throws autograd::DatasetError if: the file can't be opened, the file
// has no data rows, a row has a different number of columns than the
// first data row, or a cell can't be parsed as a number.
Dataset load_csv(const std::string& path, bool has_header = true);

} // namespace autograd
