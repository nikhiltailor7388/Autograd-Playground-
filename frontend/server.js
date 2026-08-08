const express = require('express');
const path = require('path');
const cors = require('cors');
const https = require('https');
const http = require('http');

const app = express();
app.use(cors());
app.use(express.json());

function askPieGeni(prompt) {
  const apiKey = process.env.PIEGENI_API_KEY;
  const model = process.env.PIEGENI_MODEL || 'gemini-flash-latest';
  if (!apiKey) return Promise.reject(new Error('PieGeni is not configured yet. Set PIEGENI_API_KEY on the server.'));

  const payload = JSON.stringify({
    systemInstruction: { parts: [{ text: 'You are PieGeni, a patient mathematics teacher. Solve the learner problem accurately, show compact numbered steps, define symbols, check the answer, and suggest a graph or chart when useful. Use plain text and readable mathematical notation.' }] },
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { temperature: 0.15, maxOutputTokens: 4096 },
  });

  return new Promise((resolve, reject) => {
    const request = https.request({
      hostname: 'generativelanguage.googleapis.com',
      path: `/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    }, (response) => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        let parsed;
        try { parsed = JSON.parse(body); } catch { reject(new Error(`PieGeni returned invalid JSON (status ${response.statusCode})`)); return; }
        if (response.statusCode < 200 || response.statusCode >= 300) { reject(new Error(parsed.error?.message || `PieGeni request failed (status ${response.statusCode})`)); return; }
        const text = parsed.candidates?.[0]?.content?.parts?.map((part) => part.text || '').join('') || '';
        resolve({ answer: text, model });
      });
    });
    request.on('error', (error) => reject(new Error(`PieGeni network error: ${error.message}`)));
    request.write(payload);
    request.end();
  });
}

function proxyBackend(path, body) {
  const configuredBackendUrl = process.env.BACKEND_URL || 'http://localhost:8080';
  const backendUrl = new URL(/^https?:\/\//i.test(configuredBackendUrl) ? configuredBackendUrl : `https://${configuredBackendUrl}`);
  const transport = backendUrl.protocol === 'https:' ? https : http;
  const payload = JSON.stringify(body || {});
  return new Promise((resolve, reject) => {
    const request = transport.request({
      hostname: backendUrl.hostname,
      port: backendUrl.port || (backendUrl.protocol === 'https:' ? 443 : 80),
      path: `${backendUrl.pathname.replace(/\/$/, '')}${path}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    }, (response) => {
      let result = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => { result += chunk; });
      response.on('end', () => {
        try { resolve({ status: response.statusCode, body: JSON.parse(result) }); }
        catch { reject(new Error(`backend returned invalid JSON (status ${response.statusCode})`)); }
      });
    });
    request.on('error', (error) => reject(new Error(`backend proxy error: ${error.message}`)));
    request.write(payload);
    request.end();
  });
}

// Serve static frontend files from this folder
app.use(express.static(path.join(__dirname)));

// Simple health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', piegeni: Boolean(process.env.PIEGENI_API_KEY) });
});

app.post('/api/piegeni', async (req, res) => {
  const prompt = String(req.body?.prompt || '').trim();
  if (!prompt) return res.status(400).json({ error: 'enter a mathematics question first' });
  if (prompt.length > 12000) return res.status(400).json({ error: 'question is too long (maximum 12,000 characters)' });
  try { res.json(await askPieGeni(prompt)); }
  catch (error) { res.status(502).json({ error: error.message }); }
});

for (const route of ['/api/compute', '/api/train-xor']) {
  app.post(route, async (req, res) => {
    try {
      const result = await proxyBackend(route, req.body);
      res.status(result.status).json(result.body);
    } catch (error) {
      res.status(502).json({ error: error.message });
    }
  });
}

// Basic safe-ish evaluator for the small expression language used by the
// frontend. It's intentionally small and not a substitute for the C++ engine.
function toJsExpr(src) {
  return src.replace(/\^/g, '**')
    .replace(/\bsin\s*\(/g, 'Math.sin(')
    .replace(/\bcos\s*\(/g, 'Math.cos(')
    .replace(/\btan\s*\(/g, 'Math.tan(')
    .replace(/\bcot\s*\(([^()]*)\)/g, '(Math.cos($1) / Math.sin($1))')
    .replace(/\bsec\s*\(([^()]*)\)/g, '(1 / Math.cos($1))')
    .replace(/\bcosec\s*\(([^()]*)\)/g, '(1 / Math.sin($1))')
    .replace(/\btanh\s*\(/g, 'Math.tanh(')
    .replace(/\bexp\s*\(/g, 'Math.exp(')
    .replace(/\blog\s*\(/g, 'Math.log(')
    .replace(/\brelu\s*\(/g, 'Math.max(0,');
}

function validIdent(name) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);
}

app.post('/api/compute', (req, res) => {
  const { expr = '' } = req.body || {};
  const vars = {};
  for (const k of Object.keys(req.body || {})) {
    if (k === 'expr') continue;
    if (!validIdent(k)) continue;
    const v = Number(req.body[k]);
    if (Number.isFinite(v)) vars[k] = v;
  }

  const jsExpr = toJsExpr(expr || '0');
  try {
    const argNames = Object.keys(vars);
    const argVals = argNames.map((n) => vars[n]);
    // Build a function with arg names to avoid globals
    const fn = new Function(...argNames, `return (${jsExpr});`);
    const result = fn(...argVals);

    // Build a trivial graph: one node per variable feeding a root node
    const nodes = [];
    const edges = [];
    for (const [i, name] of argNames.entries()) {
      nodes.push({ id: `v_${name}`, label: name, op: null, data: vars[name], grad: 0, is_leaf: true });
      edges.push({ source: `v_${name}`, target: 'root' });
    }
    nodes.push({ id: 'root', label: expr || 'result', op: null, data: (typeof result === 'number' && Number.isFinite(result)) ? result : 0, grad: 1, is_leaf: false });

    res.json({ result: { data: (typeof result === 'number' ? result : 0), grad: 1 }, graph: { nodes, edges } });
  } catch (err) {
    res.status(400).json({ error: `could not evaluate expression: ${err.message}` });
  }
});

app.post('/api/train-xor', (req, res) => {
  const epochs = Number.isFinite(Number(req.body?.epochs)) ? Number(req.body.epochs) : 200;
  const lr = Number.isFinite(Number(req.body?.learning_rate)) ? Number(req.body.learning_rate) : 0.05;

  // Simulate a decaying loss curve
  const losses = [];
  let loss = 1.0;
  for (let i = 0; i < epochs; i++) {
    loss = loss * (1 - Math.min(0.02 + lr * 0.01, 0.2)) + Math.random() * 0.005;
    if (i % Math.max(1, Math.floor(epochs / 200)) === 0) losses.push(Number(loss.toFixed(6)));
  }

  const final_preds = [
    { input: [0, 0], target: 0, predicted: Math.random() * 0.2 },
    { input: [0, 1], target: 1, predicted: 0.8 + Math.random() * 0.2 },
    { input: [1, 0], target: 1, predicted: 0.8 + Math.random() * 0.2 },
    { input: [1, 1], target: 0, predicted: Math.random() * 0.2 },
  ];

  res.json({ epochs, learning_rate: lr, losses, final_predictions: final_preds });
});

const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Autograd frontend dev server listening on http://localhost:${port}`);
});
