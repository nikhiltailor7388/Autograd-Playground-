#include "csv_loader.h"
#include "autograd_exceptions.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace autograd {

namespace {

// Trims leading/trailing whitespace (including '\r' left over from
// Windows-style CRLF line endings, which a plain getline() on a text
// stream doesn't strip).
std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        cells.push_back(trim(cell));
    }
    return cells;
}

double parse_double_or_throw(const std::string& cell, size_t row_num, size_t col_num) {
    try {
        size_t consumed = 0;
        double value = std::stod(cell, &consumed);
        if (consumed != cell.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw DatasetError("load_csv - row " + std::to_string(row_num) + ", column " +
                            std::to_string(col_num) + ": '" + cell +
                            "' is not a valid number");
    }
}

} // namespace

Dataset load_csv(const std::string& path, bool has_header) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw DatasetError("load_csv - could not open file: " + path);
    }

    Dataset dataset;
    std::string line;
    size_t row_num = 0;
    size_t expected_columns = 0;
    bool have_expected_columns = false;

    while (std::getline(file, line)) {
        ++row_num;
        // Skip blank lines (common trailing newline at EOF, or stray
        // blank lines in a hand-edited CSV) rather than erroring on them.
        if (trim(line).empty()) continue;

        if (has_header && row_num == 1) {
            auto header_cells = split_line(line);
            if (!header_cells.empty()) {
                // All but the last header cell name the input features;
                // the last one names the target column.
                dataset.feature_names.assign(header_cells.begin(), header_cells.end() - 1);
            }
            continue;
        }

        auto cells = split_line(line);
        if (cells.size() < 2) {
            throw DatasetError("load_csv - row " + std::to_string(row_num) +
                                " has fewer than 2 columns (need at least 1 feature + 1 target)");
        }

        if (!have_expected_columns) {
            expected_columns = cells.size();
            have_expected_columns = true;
        } else if (cells.size() != expected_columns) {
            throw DatasetError("load_csv - row " + std::to_string(row_num) + " has " +
                                std::to_string(cells.size()) + " columns, expected " +
                                std::to_string(expected_columns) +
                                " (based on the first data row)");
        }

        std::vector<double> row_inputs;
        row_inputs.reserve(cells.size() - 1);
        for (size_t col = 0; col + 1 < cells.size(); ++col) {
            row_inputs.push_back(parse_double_or_throw(cells[col], row_num, col + 1));
        }
        double target = parse_double_or_throw(cells.back(), row_num, cells.size());

        dataset.inputs.push_back(std::move(row_inputs));
        dataset.targets.push_back(target);
    }

    if (dataset.inputs.empty()) {
        throw DatasetError("load_csv - file contained no data rows: " + path);
    }

    return dataset;
}

} // namespace autograd
