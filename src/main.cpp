#include "value.h"
#include <iostream>

int main() {
    // Build a small expression: L = (a * b) + c
    auto a = make_value(2.0, "a");
    auto b = make_value(-3.0, "b");
    auto c = make_value(10.0, "c");

    auto ab = a * b;   // ab->op == "*"
    auto L  = ab + c;  // L->op  == "+"
    L->label = "L";

    std::cout << "Expression: L = (a * b) + c\n";
    std::cout << "a=" << a->data << " b=" << b->data << " c=" << c->data << "\n";
    std::cout << "Result: L.data = " << L->data << "\n\n";

    std::cout << "Computation graph (root at top, children indented below):\n";
    L->print();

    return 0;
}
