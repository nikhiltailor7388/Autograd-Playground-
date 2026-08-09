 #include "httplib.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    httplib::Server svr;

    // 1. Static frontend files serve karne ke liye (docs folder)
    svr.set_mount_point("/docs", "./docs");
    svr.set_mount_point("/", "./docs");

    // 2. Health check API endpoint
    svr.Get("/api/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // 3. Environment Variable se Port padhein (Railway PORT assign karta hai)
    int port = 8080;
    if (const char* env_p = std::getenv("PORT")) {
        port = std::atoi(env_p);
    } else if (argc > 1) {
        port = std::atoi(argv[1]);
    }

    std::cout << "Server starting on port " << port << "...\n";
    
    // Server ko listen mode me daalein
    svr.listen("0.0.0.0", port);

    return 0;
}
