 #include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include "api_handlers.h"

using json = nlohmann::json;

namespace {

void handle_json_route(const httplib::Request& req, httplib::Response& res,
                        const std::function<json(const json&)>& handler) {
    try {
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        json result = handler(body);
        res.set_content(result.dump(), "application/json");
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content(json{{"error", std::string("invalid JSON body: ") + e.what()}}.dump(),
                         "application/json");
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

} // namespace

int main(int argc, char** argv) {
    int port = 8080;
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (const std::exception&) {
            std::cerr << "Invalid port argument '" << argv[1] << "', using default 8080\n";
        }
    }

    httplib::Server svr;

    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"},
    });
    svr.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    svr.Post("/api/compute", [](const httplib::Request& req, httplib::Response& res) {
        handle_json_route(req, res, handle_compute);
    });

    svr.Post("/api/train-xor", [](const httplib::Request& req, httplib::Response& res) {
        handle_json_route(req, res, handle_train_xor);
    });

    svr.Post("/api/piegeni", [](const httplib::Request& req, httplib::Response& res) {
        handle_json_route(req, res, handle_piegeni);
    });

    std::cout << "InfiniGrad backend listening on http://localhost:" << port << "\n";
    std::cout << "  GET  /api/health\n";
    std::cout << "  POST /api/compute      { \"expr\": \"(a * b) + a\", \"a\": 2.0, \"b\": 3.0 }\n";
    std::cout << "  POST /api/train-xor    { \"epochs\": 500, \"learning_rate\": 0.05, \"hidden_layers\": [4, 4] }\n";
    std::cout << "  POST /api/piegeni      { \"prompt\": \"Find the derivative of x^2 sin(x)\" }\n";

    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "Failed to start server on port " << port << " (already in use?)\n";
        return 1;
    }
    return 0;
}
