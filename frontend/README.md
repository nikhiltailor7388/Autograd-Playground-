# InfiniGrad — Local dev server

This folder includes a tiny Node.js dev server that serves the frontend and provides a mock `/api` so you can run the UI without building the C++ backend.

Quick start:

1. Install Node dependencies:

```bash
cd autograd/frontend
npm install
```

2. Start the dev server:

```bash
npm start
# then open http://localhost:8080 in your browser
```

Notes:
- The `/api` endpoints are mocked: `POST /api/compute` evaluates simple expressions and returns a tiny computation graph; `POST /api/train-xor` returns a simulated loss curve and predictions.
- `PieGeni` uses a server-side Gemini proxy. Set `PIEGENI_API_KEY` in the environment before starting `server.js`; never put the real key in browser code or Git.
- For one-click use, copy `.env.example` to `.env`, put the key in `.env`, and double-click `..\start-fullstack.bat`. The launcher loads `.env` automatically.

For the one-click Windows launcher, run `..\start-fullstack.bat` from File Explorer. It starts the frontend, uses the compiled C++ backend when available, and opens the browser.
- I cannot start the server from this assistant session — run the commands above in a terminal on your machine to launch it.
