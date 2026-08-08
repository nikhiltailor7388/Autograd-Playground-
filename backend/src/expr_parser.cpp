#include "expr_parser.h"
#include <cctype>
#include <cmath>

namespace {

// ---- Tokenizer ----
// A tiny hand-rolled tokenizer: no separate token list is built up front,
// the parser just asks for "the next token" as it goes (single token of
// lookahead is all this grammar needs).
enum class TokType { Number, Ident, Plus, Minus, Star, Slash, Caret,
                      LParen, RParen, End };

struct Token {
    TokType type;
    std::string text;   // raw text for Ident, or the literal for Number
    double number = 0.0; // parsed value, only meaningful when type == Number
};

class Lexer {
public:
    explicit Lexer(const std::string& src) : src_(src), pos_(0) {}

    Token next() {
        skip_whitespace();
        if (pos_ >= src_.size()) return {TokType::End, ""};

        char c = src_[pos_];

        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            return read_number();
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            return read_ident();
        }

        ++pos_;
        switch (c) {
            case '+': return {TokType::Plus, "+"};
            case '-': return {TokType::Minus, "-"};
            case '*': return {TokType::Star, "*"};
            case '/': return {TokType::Slash, "/"};
            case '^': return {TokType::Caret, "^"};
            case '(': return {TokType::LParen, "("};
            case ')': return {TokType::RParen, ")"};
            default:
                throw ExprParseError(std::string("unexpected character '") + c + "' in expression");
        }
    }

private:
    const std::string& src_;
    size_t pos_;

    void skip_whitespace() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) ++pos_;
    }

    Token read_number() {
        size_t start = pos_;
        bool seen_dot = false;
        while (pos_ < src_.size() &&
               (std::isdigit(static_cast<unsigned char>(src_[pos_])) ||
                (src_[pos_] == '.' && !seen_dot))) {
            if (src_[pos_] == '.') seen_dot = true;
            ++pos_;
        }
        std::string text = src_.substr(start, pos_ - start);
        try {
            return {TokType::Number, text, std::stod(text)};
        } catch (const std::exception&) {
            throw ExprParseError("invalid number literal: '" + text + "'");
        }
    }

    Token read_ident() {
        size_t start = pos_;
        while (pos_ < src_.size() &&
               (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
            ++pos_;
        }
        return {TokType::Ident, src_.substr(start, pos_ - start)};
    }
};

// ---- Recursive-descent parser ----
class Parser {
public:
    Parser(const std::string& src, const VarMap& vars)
        : lexer_(src), vars_(vars) {
        advance();
    }

    ValuePtr parse_and_finish() {
        auto result = parse_expr();
        if (cur_.type != TokType::End) {
            throw ExprParseError("unexpected trailing input near '" + cur_.text + "'");
        }
        return result;
    }

private:
    Lexer lexer_;
    const VarMap& vars_;
    Token cur_;

    void advance() { cur_ = lexer_.next(); }

    bool check(TokType t) const { return cur_.type == t; }

    void expect(TokType t, const char* what) {
        if (!check(t)) {
            throw ExprParseError(std::string("expected '") + what + "' but found '" + cur_.text + "'");
        }
        advance();
    }

    // expr := term (('+' | '-') term)*
    ValuePtr parse_expr() {
        ValuePtr node = parse_term();
        while (check(TokType::Plus) || check(TokType::Minus)) {
            bool is_plus = check(TokType::Plus);
            advance();
            ValuePtr rhs = parse_term();
            node = is_plus ? (node + rhs) : (node - rhs);
        }
        return node;
    }

    // term := unary (('*' | '/') unary)*
    ValuePtr parse_term() {
        ValuePtr node = parse_unary();
        while (check(TokType::Star) || check(TokType::Slash)) {
            bool is_mul = check(TokType::Star);
            advance();
            ValuePtr rhs = parse_unary();
            node = is_mul ? (node * rhs) : (node / rhs);
        }
        return node;
    }

    // unary := '-' unary | power
    ValuePtr parse_unary() {
        if (check(TokType::Minus)) {
            advance();
            return -parse_unary();
        }
        return parse_power();
    }

    // power := primary ('^' NUMBER)?
    // The exponent must be a bare numeric literal (optionally negated),
    // because the underlying engine's pow() takes a constant double
    // exponent, not a Value -- there's no gradient wired for a variable
    // exponent. e.g. "a^2" and "a^-1" are fine; "a^b" is rejected.
    ValuePtr parse_power() {
        ValuePtr base = parse_primary();
        if (check(TokType::Caret)) {
            advance();
            double sign = 1.0;
            if (check(TokType::Minus)) { sign = -1.0; advance(); }
            if (!check(TokType::Number)) {
                throw ExprParseError("exponent after '^' must be a numeric literal (variable exponents aren't supported)");
            }
            double exponent = sign * cur_.number;
            advance();
            return pow(base, exponent);
        }
        return base;
    }

    // primary := NUMBER | IDENT | FUNC '(' expr ')' | '(' expr ')'
    ValuePtr parse_primary() {
        if (check(TokType::Number)) {
            double v = cur_.number;
            advance();
            return make_value(v);
        }
        if (check(TokType::LParen)) {
            advance();
            ValuePtr inner = parse_expr();
            expect(TokType::RParen, ")");
            return inner;
        }
        if (check(TokType::Ident)) {
            std::string name = cur_.text;
            advance();
            // Function call: name immediately followed by '('.
            if (check(TokType::LParen) && is_known_function(name)) {
                advance();
                ValuePtr arg = parse_expr();
                expect(TokType::RParen, ")");
                return apply_function(name, arg);
            }
            // Otherwise it's a variable reference.
            auto it = vars_.find(name);
            if (it == vars_.end()) {
                throw ExprParseError("unknown variable '" + name + "' (not provided in request)");
            }
            return it->second;
        }
        throw ExprParseError("unexpected token '" + cur_.text + "' while parsing expression");
    }

    static bool is_known_function(const std::string& name) {
        return name == "sin" || name == "cos" || name == "tan" || name == "cot" ||
               name == "sec" || name == "cosec" || name == "tanh" || name == "exp" ||
               name == "log" || name == "relu";
    }

    static ValuePtr apply_function(const std::string& name, const ValuePtr& arg) {
        if (name == "sin") return sin(arg);
        if (name == "cos") return cos(arg);
        if (name == "tan") return tan(arg);
        if (name == "cot") return cot(arg);
        if (name == "sec") return sec(arg);
        if (name == "cosec") return cosec(arg);
        if (name == "tanh") return tanh(arg);
        if (name == "exp") return exp(arg);
        if (name == "log") return log(arg);
        if (name == "relu") return relu(arg);
        throw ExprParseError("internal error: unhandled function '" + name + "'");
    }
};

} // namespace

ValuePtr parse_expression(const std::string& expr, const VarMap& vars) {
    if (expr.empty()) {
        throw ExprParseError("expression string is empty");
    }
    Parser parser(expr, vars);
    return parser.parse_and_finish();
}
