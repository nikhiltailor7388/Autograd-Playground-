// autograd_server -- the Phase 5 API layer.
//
// Exposes the C++ autograd engine (Phases 1-4) over a tiny local REST API
// so a browser frontend (Phase 6) can drive it. Uses cpp-httplib (a
// single-header HTTP library, fetched by CMake) so there's no heavyweight
// networking dependency to install by hand.
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include "api_handlers.h"

using json = nlohmann::json;

namespace {

// Wraps a handler function so every route gets identical error handling:
// on success, write its JSON return value as the response body; on any
// thrown exception (bad JSON, unknown variable, out-of-range parameter,
// etc.), respond 400 with {"error": "..."} instead of crashing the server
// or leaking an unhandled-exception 500.
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

    // CORS: the frontend (Phase 6) is a static HTML/JS page opened
    // directly in the browser (or served from a different port than
    // this API), so every response needs permissive CORS headers, and
    // preflight OPTIONS requests need to be answered directly.
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

    std::cout << "InfiniGrad backend listening on http://localhost:" << port << "\n";
    std::cout << "  GET  /api/health\n";
    std::cout << "  POST /api/compute      { \"expr\": \"(a * b) + a\", \"a\": 2.0, \"b\": 3.0 }\n";
    std::cout << "  POST /api/train-xor    { \"epochs\": 500, \"learning_rate\": 0.05, \"hidden_layers\": [4, 4] }\n";

    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "Failed to start server on port " << port << " (already in use?)\n";
        return 1;
    }
    return 0;
}
