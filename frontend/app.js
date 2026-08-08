// =============================================================================
// Autograd Playground — frontend logic
//
// Talks to the autograd_server REST API (Phase 5):
//   GET  /api/health
//   POST /api/compute      { expr, <var>: <number>, ... }        -> { result, graph }
//   POST /api/train-xor    { epochs, learning_rate, hidden_layers } -> { losses, final_predictions }
//
// No frameworks: plain DOM + a small hand-rolled SVG graph layout (so the
// visual language matches the rest of the "chalkboard" design) + Chart.js
// for the loss curve.
// =============================================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ----------------------------------------------------------------------------
// Backend connection
// ----------------------------------------------------------------------------

const STORAGE_KEY_BACKEND_URL = 'autograd_backend_url';

function getBackendUrl() {
  return (localStorage.getItem(STORAGE_KEY_BACKEND_URL) || 'http://localhost:8080').replace(/\/+$/, '');
}

function setBackendUrl(url) {
  localStorage.setItem(STORAGE_KEY_BACKEND_URL, url.replace(/\/+$/, ''));
}

async function apiPost(path, body) {
  const url = `${getBackendUrl()}${path}`;
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(`could not reach ${url} — is the backend running? (${err.message})`);
  }
  let json;
  try {
    json = await res.json();
  } catch (err) {
    throw new Error(`backend returned a non-JSON response (status ${res.status})`);
  }
  if (!res.ok) {
    throw new Error(json.error || `backend returned status ${res.status}`);
  }
  return json;
}

async function checkHealth() {
  const pill = $('#conn-status');
  const text = $('#conn-status-text');
  try {
    const res = await fetch(`${getBackendUrl()}/api/health`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    pill.className = 'conn-pill conn-pill--ok';
    text.textContent = `connected · ${getBackendUrl()}`;
  } catch (err) {
    pill.className = 'conn-pill conn-pill--error';
    text.textContent = 'backend unreachable';
  }
}

function initSettingsPanel() {
  const input = $('#backend-url');
  input.value = getBackendUrl();

  $('#settings-toggle').addEventListener('click', () => {
    const panel = $('#settings-panel');
    const btn = $('#settings-toggle');
    const isHidden = panel.hasAttribute('hidden');
    if (isHidden) {
      panel.removeAttribute('hidden');
      btn.setAttribute('aria-expanded', 'true');
    } else {
      panel.setAttribute('hidden', '');
      btn.setAttribute('aria-expanded', 'false');
    }
  });

  $('#settings-save').addEventListener('click', () => {
    setBackendUrl(input.value.trim() || 'http://localhost:8080');
    checkHealth();
  });
}

// ----------------------------------------------------------------------------
// Sandbox: variable rows
// ----------------------------------------------------------------------------

let varRows = [
  { name: 'a', value: 2 },
  { name: 'b', value: 3 },
];

function renderVarRows() {
  const container = $('#vars-list');
  container.innerHTML = '';
  varRows.forEach((row, i) => {
    const div = document.createElement('div');
    div.className = 'var-row';
    div.innerHTML = `
      <input type="text" class="mono-input var-name" value="${escapeHtml(row.name)}" aria-label="variable name" spellcheck="false" autocomplete="off">
      <input type="number" step="any" class="mono-input var-value" value="${row.value}" aria-label="variable value">
      <button type="button" class="var-remove-btn" aria-label="remove variable ${escapeHtml(row.name)}">&times;</button>
    `;
    const [nameInput, valueInput] = div.querySelectorAll('input');
    nameInput.addEventListener('input', () => { varRows[i].name = nameInput.value.trim(); });
    valueInput.addEventListener('input', () => { varRows[i].value = parseFloat(valueInput.value); });
    div.querySelector('.var-remove-btn').addEventListener('click', () => {
      varRows.splice(i, 1);
      renderVarRows();
    });
    container.appendChild(div);
  });
}

$('#add-var-btn').addEventListener('click', () => {
  const usedNames = new Set(varRows.map((r) => r.name));
  let candidate = 'x';
  let n = 0;
  while (usedNames.has(candidate)) { candidate = `x${n++}`; }
  varRows.push({ name: candidate, value: 1 });
  renderVarRows();
});

// ----------------------------------------------------------------------------
// Sandbox: presets
// ----------------------------------------------------------------------------

const PRESETS = [
  { label: '(a·b)+a', expr: '(a * b) + a', vars: { a: 2, b: 3 } },
  { label: 'tanh(a)+b', expr: 'tanh(a) + b', vars: { a: 0.8, b: 0.5 } },
  { label: '(a-b)²', expr: '(a - b) ^ 2', vars: { a: 5, b: 2 } },
  { label: 'a/b − 1', expr: 'a / b - 1', vars: { a: 6, b: 4 } },
  { label: 'neuron: tanh(w·x+b)', expr: 'tanh(w * x + b)', vars: { w: 0.6, x: 1.2, b: -0.3 } },
];

function renderPresets() {
  const row = $('#preset-row');
  row.innerHTML = '';
  PRESETS.forEach((preset) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'preset-chip';
    chip.textContent = preset.label;
    chip.addEventListener('click', () => {
      $('#expr-input').value = preset.expr;
      varRows = Object.entries(preset.vars).map(([name, value]) => ({ name, value }));
      renderVarRows();
    });
    row.appendChild(chip);
  });
}

// ----------------------------------------------------------------------------
// Sandbox: run backward()
// ----------------------------------------------------------------------------

function showSandboxError(message) {
  const banner = $('#sandbox-error');
  if (!message) {
    banner.setAttribute('hidden', '');
    banner.textContent = '';
    return;
  }
  banner.textContent = message;
  banner.removeAttribute('hidden');
}

function showCalculusError(message) {
  const banner = $('#calculus-error');
  if (!message) {
    banner.setAttribute('hidden', '');
    banner.textContent = '';
    return;
  }
  banner.textContent = message;
  banner.removeAttribute('hidden');
}

function solveCalculusStatement(source) {
  const normalizedSource = source.trim()
    .replace(/\\frac\s*\{d\}\s*\{d([A-Za-z_]\w*)\}/gi, 'd/d$1')
    .replace(/\{d\}\{d([A-Za-z_]\w*)\}/gi, 'd/d$1')
    .replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, '($1)/($2)')
    .replace(/\\(ln|log|sin|cos|tan|cot|sec|cosec)\b/gi, '$1')
    .replace(/\\,/g, '').replace(/\\!/g, '');
  const text = normalizedSource.replace(/[{}]/g, '');
  const compact = text.replace(/\s+/g, '');
  const derivativeMatch = compact.match(/^d\/d([A-Za-z_]\w*)\((.*)\)$/i);
  const derivative = Boolean(derivativeMatch) || /^(?:derivative(?:of)?)/i.test(compact);
  const variable = derivativeMatch?.[1]
    || (compact.match(/d\/d([A-Za-z_]\w*)/i) || [])[1]
    || (compact.match(/d([A-Za-z_]\w*)$/i) || [])[1]
    || 'x';
  const content = $('#calculus-solution-content');

  if (derivative || /derivative/i.test(text)) {
    const expression = derivativeMatch ? derivativeMatch[2] : compact.replace(/^.*?(?:=|of)/i, '').replace(/^\((.*)\)$/, '$1');
    const derivativeRules = [
      { pattern: /^(?:x|\(?x\)?)[\^](n)$/i, html: `<p class="teacher-step"><strong>1. State the power rule:</strong> <code>d/d${variable}(u^n) = n &middot; u^(n-1)</code>.</p><p class="teacher-step"><strong>2. Identify the base:</strong> here <code>u = ${variable}</code>.</p><p class="teacher-step"><strong>3. Substitute into the rule:</strong> <code>d/d${variable}(${variable}^n) = n &middot; ${variable}^(n-1)</code>.</p><p class="teacher-answer"><strong>Final answer:</strong> <code>n &middot; ${variable}^(n-1)</code>.</p>` },
      { pattern: /^-cos\(x\)$/i, html: `<p class="teacher-step"><strong>1. Use the cosine rule:</strong> <code>d/d${variable}(cos(${variable})) = -sin(${variable})</code>.</p><p class="teacher-step"><strong>2. Keep the leading negative sign:</strong> <code>-(-sin(${variable}))</code>.</p><p class="teacher-answer"><strong>Final answer:</strong> <code>sin(${variable})</code>.</p>` },
      { pattern: /^sin\(x\)$/i, html: `<p class="teacher-step"><strong>1. Recall the derivative:</strong> <code>d/d${variable}(sin(${variable})) = cos(${variable})</code>.</p><p class="teacher-step"><strong>2. Differentiation is direct because the inside is ${variable}.</strong></p><p class="teacher-answer"><strong>Final answer:</strong> <code>cos(${variable})</code>.</p>` },
      { pattern: /^cos\(x\)$/i, html: `<p class="teacher-step"><strong>1. Recall the cosine rule:</strong> <code>d/d${variable}(cos(${variable})) = -sin(${variable})</code>.</p><p class="teacher-answer"><strong>Final answer:</strong> <code>-sin(${variable})</code>.</p>` },
      { pattern: /^tan\(x\)$/i, html: `<p class="teacher-step"><strong>1. Recall the tangent rule:</strong> <code>d/d${variable}(tan(${variable})) = sec(${variable})^2</code>.</p><p class="teacher-answer"><strong>Final answer:</strong> <code>sec(${variable})^2</code>.</p>` },
      { pattern: /^log\(x\)$|^ln\(x\)$/i, html: `<p class="teacher-step"><strong>1. Recall the logarithm rule:</strong> <code>d/d${variable}(ln(${variable})) = 1/${variable}</code>.</p><p class="teacher-step"><strong>2. Domain condition:</strong> the logarithm requires <code>${variable} &gt; 0</code>.</p><p class="teacher-answer"><strong>Final answer:</strong> <code>1/${variable}</code>.</p>` },
    ];
    const ruleExpression = expression.replace(new RegExp(`\\b${variable}\\b`, 'g'), 'x');
    const rule = derivativeRules.find((item) => item.pattern.test(ruleExpression));
    if (rule) return rule.html.replace(/\bx\b/g, variable);
    return `<p><strong>Step 1:</strong> identify the outer operation.</p><p><strong>Step 2:</strong> apply the product, quotient, power, or chain rule.</p><p><strong>Answer:</strong> symbolic differentiation for <code>${escapeHtml(expression)}</code> is not covered by the current rule library yet.</p>`;
  }

  if (/^(?:\\int|∫|int)/i.test(compact)) {
    const expression = compact.replace(/^(?:\\int|∫|int)/i, '').replace(/d[a-z_]\w*$/i, '').replace(/^\((.*)\)$/, '$1');
    const integralRules = [
      { pattern: /^sin\(x\)$/i, html: `<p class="teacher-step"><strong>1. Recall the cosine derivative:</strong> <code>d/d${variable}(cos(${variable})) = -sin(${variable})</code>.</p><p class="teacher-step"><strong>2. Multiply by -1:</strong> <code>d/d${variable}(-cos(${variable})) = sin(${variable})</code>.</p><p class="teacher-step"><strong>3. Reverse differentiation:</strong> the antiderivative of <code>sin(${variable})</code> is <code>-cos(${variable})</code>.</p><p class="teacher-step"><strong>4. Add the constant of integration.</strong></p><p class="teacher-answer"><strong>Final answer:</strong> <code>&int; sin(${variable}) d${variable} = -cos(${variable}) + C</code>.</p>` },
      { pattern: /^cos\(x\)$/i, html: `<p class="teacher-step"><strong>1. Recall:</strong> <code>d/d${variable}(sin(${variable})) = cos(${variable})</code>.</p><p class="teacher-step"><strong>2. Reverse differentiation:</strong> the antiderivative of <code>cos(${variable})</code> is <code>sin(${variable})</code>.</p><p class="teacher-step"><strong>3. Add the constant of integration.</strong></p><p class="teacher-answer"><strong>Final answer:</strong> <code>&int; cos(${variable}) d${variable} = sin(${variable}) + C</code>.</p>` },
      { pattern: /^(?:ln\(x\)|log\(x\))$/i, html: `<p class="teacher-step"><strong>1. Choose integration by parts:</strong> <code>u = ln(${variable})</code>, <code>dv = d${variable}</code>.</p><p class="teacher-step"><strong>2. Differentiate and integrate:</strong> <code>du = (1/${variable})d${variable}</code>, <code>v = ${variable}</code>.</p><p class="teacher-step"><strong>3. Apply</strong> <code>&int;u dv = uv - &int;v du</code>: <code>&int; ln(${variable}) d${variable} = ${variable}ln(${variable}) - &int;1 d${variable}</code>.</p><p class="teacher-step"><strong>4. Simplify and add C.</strong></p><p class="teacher-answer"><strong>Final answer:</strong> <code>${variable}ln(${variable}) - ${variable} + C</code>, for <code>${variable} &gt; 0</code>.</p>` },
      { pattern: /^(?:1\/x|\\frac1x)$/i, html: `<p class="teacher-step"><strong>1. Use the logarithm derivative:</strong> <code>d/d${variable}(ln|${variable}|) = 1/${variable}</code>.</p><p class="teacher-step"><strong>2. Reverse differentiation.</strong></p><p class="teacher-step"><strong>3. Add C:</strong> the domain excludes <code>${variable} = 0</code>.</p><p class="teacher-answer"><strong>Final answer:</strong> <code>&int; 1/${variable} d${variable} = ln|${variable}| + C</code>.</p>` },
      { pattern: /^x\^n$/i, html: `<p class="teacher-step"><strong>1. Use the reverse power rule:</strong> increase the exponent by one.</p><p class="teacher-step"><strong>2. Divide by the new exponent:</strong> <code>&int; ${variable}^n d${variable} = ${variable}^(n+1)/(n+1)</code>.</p><p class="teacher-step"><strong>3. Add C.</strong></p><p class="teacher-answer"><strong>Final answer:</strong> <code>${variable}^(n+1)/(n+1) + C</code>, when <code>n &ne; -1</code>.</p>` },
    ];
    const ruleExpression = expression.replace(new RegExp(`\\b${variable}\\b`, 'g'), 'x');
    const rule = integralRules.find((item) => item.pattern.test(ruleExpression));
    if (rule) return rule.html.replace(/\bx\b/g, variable);
    return `<p><strong>Step 1:</strong> identify the integrand.</p><p><strong>Step 2:</strong> choose a reverse derivative rule or integration method.</p><p><strong>Answer:</strong> symbolic integration for <code>${escapeHtml(expression)}</code> is not covered by the current rule library yet.</p>`;
  }
  throw new Error('Use a derivative such as d/dx(sin(x)) or an integral such as ∫ sin(x) dx.');
}

function runCalculusSolver() {
  showCalculusError(null);
  const solution = $('#calculus-solution');
  try {
    const input = $('#calculus-input').value;
    if (!input.trim()) throw new Error('enter a calculus expression first');
    const normalized = input.trim()
      .replace(/\\frac\s*\{d\}\s*\{d([A-Za-z_]\w*)\}/gi, 'd/d$1')
      .replace(/\{d\}\{d([A-Za-z_]\w*)\}/gi, 'd/d$1')
      .replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, '($1)/($2)')
      .replace(/\\(ln|log)\b/gi, 'log')
      .replace(/\\,/g, '').replace(/\\!/g, '')
      .replace(/[{}]/g, '')
      .replace(/\s+/g, '');
    let graphExpression = null;
    const derivativeMatch = normalized.match(/^d\/d[A-Za-z_]\w*\((.*)\)$/i);
    const integralMatch = normalized.match(/^(?:\\int|∫|int)(.*?)(?:d[A-Za-z_]\w*)$/i);
    if (derivativeMatch) graphExpression = derivativeMatch[1];
    else if (integralMatch) graphExpression = integralMatch[1];
    const functionNames = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'cosec', 'tanh', 'exp', 'log', 'ln', 'relu']);
    const graphVariable = graphExpression?.match(/\b[A-Za-z_]\w*\b/g)?.find((name) => !functionNames.has(name));
    if (graphExpression && graphVariable) renderEquationPlot(`y = ${graphExpression}`);
    else renderEquationPlot('');
    $('#calculus-solution-content').innerHTML = solveCalculusStatement(input);
    solution.removeAttribute('hidden');
  } catch (error) {
    solution.setAttribute('hidden', '');
    showCalculusError(error.message);
  }
}

$('#calculus-run-btn').addEventListener('click', runCalculusSolver);

function showTeacherError(message) {
  const banner = $('#teacher-error');
  if (!message) {
    banner.setAttribute('hidden', '');
    banner.textContent = '';
    return;
  }
  banner.textContent = message;
  banner.removeAttribute('hidden');
}

function solveTeacherProblem(source) {
  const input = source.trim();
  const normalized = input.replace(/\s+/g, '').replace(/\*\*/g, '^');
  if (/^(?:d\/d|\\frac\{d\}|\{d\}|∫|\\int|int)/i.test(normalized)) {
    return solveCalculusStatement(input);
  }

  const equation = input.match(/^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$/);
  if (equation) {
    renderEquationPlot(input);
    return `
      <p class="teacher-step"><strong>Step 1: Identify the equation.</strong> The dependent variable is <code>${escapeHtml(equation[1])}</code>, written in terms of the right-hand expression.</p>
      <p class="teacher-step"><strong>Step 2: Substitute and evaluate.</strong> The graph samples many values of the independent variable and plots the resulting ordered pairs.</p>
      <p class="teacher-step"><strong>Step 3: Read the shape.</strong> The plotted curve shows how <code>${escapeHtml(equation[1])}</code> changes as the input changes.</p>
      <p class="teacher-answer"><strong>Equation:</strong> <code>${escapeHtml(input)}</code></p>
    `;
  }

  const equality = normalized.match(/^([+-]?\d*\.?\d*)\*?([A-Za-z_]\w*)\^?([02]?)([+-]\d*\.?\d+)?=([+-]?\d*\.?\d*)$/);
  if (equality) {
    const coefficientText = equality[1];
    const variable = equality[2];
    const exponent = equality[3] || '1';
    const constant = Number(equality[4] || 0);
    const right = Number(equality[5]);
    const coefficient = coefficientText === '' || coefficientText === '+' ? 1 : coefficientText === '-' ? -1 : Number(coefficientText);
    if (!Number.isFinite(coefficient) || !Number.isFinite(right)) throw new Error('could not read the numeric equation');
    const adjusted = right - constant;
    if (exponent === '2') {
      if (adjusted / coefficient < 0) {
        return `<p class="teacher-step"><strong>Step 1:</strong> isolate <code>${variable}^2</code>: <code>${variable}^2 = ${formatNumber(adjusted / coefficient)}</code>.</p><p class="teacher-answer"><strong>Answer:</strong> no real solution because a square cannot be negative.</p>`;
      }
      const root = Math.sqrt(adjusted / coefficient);
      return `<p class="teacher-step"><strong>Step 1: Isolate the square.</strong> <code>${variable}^2 = ${formatNumber(adjusted / coefficient)}</code>.</p><p class="teacher-step"><strong>Step 2: Take both square roots.</strong> <code>${variable} = &plusmn;&radic;${formatNumber(adjusted / coefficient)}</code>.</p><p class="teacher-answer"><strong>Answer:</strong> <code>${variable} = ${formatNumber(root)}</code> or <code>${variable} = ${formatNumber(-root)}</code>.</p>`;
    }
    const answer = adjusted / coefficient;
    return `<p class="teacher-step"><strong>Step 1: Move the constant.</strong> <code>${coefficient}${variable} = ${formatNumber(adjusted)}</code>.</p><p class="teacher-step"><strong>Step 2: Divide by the coefficient.</strong> <code>${variable} = ${formatNumber(adjusted)} / ${coefficient}</code>.</p><p class="teacher-answer"><strong>Answer:</strong> <code>${variable} = ${formatNumber(answer)}</code>.</p>`;
  }

  renderEquationPlot(input);
  return `<p class="teacher-step"><strong>Step 1:</strong> identify the operations and variables.</p><p class="teacher-step"><strong>Step 2:</strong> use the computation graph to evaluate the expression.</p><p class="teacher-answer"><strong>Result:</strong> this form is supported numerically, but a complete symbolic rearrangement needs a computer-algebra rule that is not yet available.</p>`;
}

function runTeacherSolver() {
  showTeacherError(null);
  try {
    const input = $('#teacher-input').value;
    if (!input.trim()) throw new Error('enter a problem first');
    $('#teacher-solution-content').innerHTML = solveTeacherProblem(input);
    $('#teacher-solution').removeAttribute('hidden');
  } catch (error) {
    $('#teacher-solution').setAttribute('hidden', '');
    showTeacherError(error.message);
  }
}

$('#teacher-run-btn').addEventListener('click', runTeacherSolver);

async function runPieGeni() {
  const button = $('#piegeni-run-btn');
  const answer = $('#piegeni-answer');
  const error = $('#piegeni-error');
  error.setAttribute('hidden', '');
  button.disabled = true;
  button.textContent = 'thinking…';
  try {
    const prompt = $('#piegeni-input').value.trim();
    if (!prompt) throw new Error('enter a mathematics question first');
    const data = await apiPost('/api/piegeni', { prompt });
    $('#piegeni-answer-content').textContent = data.answer;
    answer.removeAttribute('hidden');
  } catch (err) {
    error.textContent = err.message;
    error.removeAttribute('hidden');
    answer.setAttribute('hidden', '');
  } finally {
    button.disabled = false;
    button.textContent = 'ask PieGeni';
  }
}

$('#piegeni-run-btn').addEventListener('click', runPieGeni);

async function runBackward() {
  showSandboxError(null);
  const btn = $('#run-btn');
  btn.disabled = true;
  btn.textContent = 'running…';

  try {
    const expr = $('#expr-input').value.trim();
    if (!expr) throw new Error('enter an expression first');

    const equation = expr.match(/^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$/);
    const computeExpr = equation ? equation[2].replace(/\*\*/g, '^') : expr;
    const body = { expr: computeExpr };
    for (const row of varRows) {
      if (!row.name) continue;
      if (Number.isNaN(row.value)) throw new Error(`variable "${row.name}" needs a numeric value`);
      body[row.name] = row.value;
    }
    if (equation) {
      const knownFunctions = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'cosec', 'tanh', 'exp', 'log', 'relu']);
      const inferredVariable = equation[2].match(/[A-Za-z_]\w*/g)?.find((name) => name !== equation[1] && !knownFunctions.has(name));
      if (inferredVariable && !(inferredVariable in body)) body[inferredVariable] = 1;
    }

    renderEquationPlot(expr);

    const data = await apiPost('/api/compute', body);

    $('#result-readout').removeAttribute('hidden');
    $('#result-data').textContent = formatNumber(data.result.data);
    $('#result-grad').textContent = formatNumber(data.result.grad);
    $('#graph-empty').setAttribute('hidden', '');

    let graph = data.graph;
    // The lightweight local API returns only variable-to-result links. Expand
    // that response into an equation tree so every operator is visible.
    try {
      graph = equationGraph(computeExpr, body, data.result.data);
    } catch (error) {
      console.warn('Could not expand equation graph; showing API graph instead.', error);
    }
    renderGraph(graph.nodes, graph.edges);
  } catch (err) {
    showSandboxError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'run backward()';
  }
}

$('#run-btn').addEventListener('click', runBackward);
$('#expr-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') runBackward(); });

// ----------------------------------------------------------------------------
// Graph layout + SVG rendering
// ----------------------------------------------------------------------------

const NODE_W = 118;
const NODE_H = 56;
const COL_GAP = 170;
const ROW_GAP = 76;
// Keep the first column fully inside the SVG. The old margin placed leaf
// cards at x=-19, which clipped their left edge on every graph.
const MARGIN = NODE_W / 2 + 24;

// Deterministic small jitter per node id, so the "hand-drawn" wobble is
// stable across re-renders of the same graph instead of reshuffling every
// time (which would feel jumpy, not chalky).
function jitterFor(id) {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
  return ((hash % 100) / 100 - 0.5) * 14; // +/- 7px
}

function layoutGraph(nodes, edges) {
  // childrenOf[targetId] = [childId, ...]  (edge.source is the child, per
  // the backend's documented edge direction: source (child) -> target (parent))
  const childrenOf = new Map();
  for (const e of edges) {
    if (!childrenOf.has(e.target)) childrenOf.set(e.target, []);
    childrenOf.get(e.target).push(e.source);
  }

  // `nodes` arrives in topological order (children before parents; see
  // backend/src/graph_json.cpp), so a single forward pass is enough to
  // compute every node's depth = 1 + max(depth of its children), 0 for leaves.
  const depth = new Map();
  for (const n of nodes) {
    const kids = childrenOf.get(n.id) || [];
    if (kids.length === 0) {
      depth.set(n.id, 0);
    } else {
      depth.set(n.id, 1 + Math.max(...kids.map((k) => depth.get(k))));
    }
  }

  const maxDepth = Math.max(0, ...nodes.map((n) => depth.get(n.id)));
  const columns = new Map(); // depth -> [nodeId, ...] in encounter order
  for (const n of nodes) {
    const d = depth.get(n.id);
    if (!columns.has(d)) columns.set(d, []);
    columns.get(d).push(n.id);
  }
  const maxColSize = Math.max(1, ...Array.from(columns.values()).map((c) => c.length));

  const width = MARGIN * 2 + (maxDepth + 1) * COL_GAP;
  const height = MARGIN * 2 + maxColSize * ROW_GAP;

  const positions = new Map();
  // Leaves (depth 0) on the left, root (max depth) on the right — data
  // flows left-to-right, matching how the expression was written; the
  // gradient pulse then visibly runs right-to-left, root back to leaves.
  for (const [d, ids] of columns.entries()) {
    const colHeight = ids.length * ROW_GAP;
    const startY = (height - colHeight) / 2 + ROW_GAP / 2;
    ids.forEach((id, i) => {
      const x = MARGIN + d * COL_GAP;
      const y = startY + i * ROW_GAP + jitterFor(id);
      positions.set(id, { x, y, depth: d });
    });
  }

  return { positions, width: Math.max(width, 480), height: Math.max(height, 260) };
}

function formatNumber(n) {
  if (typeof n !== 'number' || Number.isNaN(n)) return '—';
  if (Object.is(n, -0)) n = 0;
  if (Math.abs(n) < 1e-4 && n !== 0) return n.toExponential(2);
  return (Math.round(n * 10000) / 10000).toString();
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function graphLabel(label, maxLength = 17) {
  const text = String(label);
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function equationGraph(expr, values, result) {
  const tokens = expr.match(/[A-Za-z_][A-Za-z0-9_]*|\d*\.?\d+(?:[eE][+-]?\d+)?|[()+\-*/^,]/g) || [];
  let cursor = 0;
  const peek = () => tokens[cursor];
  const take = () => tokens[cursor++];

  function parsePrimary() {
    const token = take();
    if (token === '(') {
      const node = parseExpression();
      if (take() !== ')') throw new Error('unclosed parenthesis in expression');
      return node;
    }
    if (token && /^\d/.test(token)) return { label: token, op: null, value: Number(token), leaf: true };
    if (token && /^[A-Za-z_]/.test(token)) {
      if (peek() === '(') {
        take();
        const args = [];
        if (peek() !== ')') {
          args.push(parseExpression());
          while (peek() === ',') { take(); args.push(parseExpression()); }
        }
        if (take() !== ')') throw new Error('unclosed function call in expression');
        return { label: token, op: token, args };
      }
      return { label: token, op: null, value: Number(values[token] ?? 0), leaf: true };
    }
    throw new Error(`unexpected token "${token || 'end'}"`);
  }

  function parsePower() {
    const left = parsePrimary();
    if (peek() === '^') { const op = take(); return { label: op, op, args: [left, parsePower()] }; }
    return left;
  }

  function parseUnary() {
    if (peek() === '-') { take(); return { label: 'neg', op: 'neg', args: [parseUnary()] }; }
    return parsePower();
  }

  function parseTerm() {
    let node = parseUnary();
    while (peek() === '*' || peek() === '/') {
      const op = take();
      node = { label: op, op, args: [node, parseUnary()] };
    }
    return node;
  }

  function parseExpression() {
    let node = parseTerm();
    while (peek() === '+' || peek() === '-') {
      const op = take();
      node = { label: op, op, args: [node, parseTerm()] };
    }
    return node;
  }

  const root = parseExpression();
  if (cursor !== tokens.length) throw new Error(`unexpected token "${peek()}"`);
  const nodes = [];
  const edges = [];
  let sequence = 0;
  function visit(node) {
    if (node.leaf) {
      const id = `leaf_${node.label}`;
      if (!nodes.some((item) => item.id === id)) {
        nodes.push({ id, label: node.label, op: null, data: node.value, grad: 0, is_leaf: true });
      }
      return id;
    }
    const childIds = node.args.map(visit);
    const id = `op_${sequence++}`;
    const args = node.args.map((item) => item.value ?? 0);
    if (node.op === '+') node.value = args[0] + args[1];
    else if (node.op === '-') node.value = args[0] - args[1];
    else if (node.op === '*') node.value = args[0] * args[1];
    else if (node.op === '/') node.value = args[0] / args[1];
    else if (node.op === '^') node.value = args[0] ** args[1];
    else if (node.op === 'neg') node.value = -args[0];
    else if (node.op === 'sin') node.value = Math.sin(args[0]);
    else if (node.op === 'cos') node.value = Math.cos(args[0]);
    else if (node.op === 'tan') node.value = Math.tan(args[0]);
    else if (node.op === 'cot') node.value = Math.cos(args[0]) / Math.sin(args[0]);
    else if (node.op === 'sec') node.value = 1 / Math.cos(args[0]);
    else if (node.op === 'cosec') node.value = 1 / Math.sin(args[0]);
    else if (node.op === 'tanh') node.value = Math.tanh(args[0]);
    else if (node.op === 'exp') node.value = Math.exp(args[0]);
    else if (node.op === 'log') node.value = Math.log(args[0]);
    else if (node.op === 'relu') node.value = Math.max(0, args[0]);
    nodes.push({ id, label: node.label, op: node.op, data: node.value, grad: 0, is_leaf: false });
    childIds.forEach((childId) => edges.push({ source: childId, target: id }));
    return id;
  }
  const rootId = visit(root);
  const rootNode = nodes.find((node) => node.id === rootId);
  if (rootNode) { rootNode.label = expr; rootNode.op = null; rootNode.data = result; rootNode.grad = 1; }
  return { nodes, edges };
}

function symbolicDerivative(expr, variable) {
  const tokens = expr.match(/[A-Za-z_]\w*|\d*\.?\d+(?:[eE][+-]?\d+)?|[()+\-*/^,]/g) || [];
  let cursor = 0;
  const peek = () => tokens[cursor];
  const take = () => tokens[cursor++];
  const number = (value) => Number(value) === 0 ? '0' : String(Number(value));
  const simplify = (value) => value.replace(/\b1\s*\*\s*/g, '').replace(/\s*\*\s*1\b/g, '').replace(/\+\s*0\b/g, '').replace(/\b0\s*\*\s*[A-Za-z_(]/g, '0');

  function primary() {
    const token = take();
    if (token === '(') {
      const result = expression();
      if (take() !== ')') throw new Error('unclosed parenthesis');
      return result;
    }
    if (/^\d/.test(token || '')) return { text: token, derivative: '0' };
    if (/^[A-Za-z_]/.test(token || '')) {
      if (peek() === '(') {
        take();
        const arg = expression();
        if (take() !== ')') throw new Error('unclosed function');
        const rules = {
          sin: `cos(${arg.text}) * ${arg.derivative}`,
          cos: `-sin(${arg.text}) * ${arg.derivative}`,
          tan: `sec(${arg.text})^2 * ${arg.derivative}`,
          cot: `-cosec(${arg.text})^2 * ${arg.derivative}`,
          sec: `sec(${arg.text}) * tan(${arg.text}) * ${arg.derivative}`,
          cosec: `-cosec(${arg.text}) * cot(${arg.text}) * ${arg.derivative}`,
          tanh: `(1 - tanh(${arg.text})^2) * ${arg.derivative}`,
          exp: `exp(${arg.text}) * ${arg.derivative}`,
          log: `(1 / (${arg.text})) * ${arg.derivative}`,
          relu: `relu'(${arg.text}) * ${arg.derivative}`,
        };
        if (!(token in rules)) throw new Error(`unsupported function ${token}`);
        return { text: `${token}(${arg.text})`, derivative: rules[token] };
      }
      return { text: token, derivative: token === variable ? '1' : '0' };
    }
    throw new Error('invalid expression');
  }
  function power() {
    const base = primary();
    if (peek() !== '^') return base;
    take();
    const exponent = primary();
    if (exponent.derivative !== '0') throw new Error('variable powers are not supported symbolically');
    return { text: `(${base.text})^${exponent.text}`, derivative: `${exponent.text} * (${base.text})^(${Number(exponent.text) - 1}) * (${base.derivative})` };
  }
  function unary() {
    if (peek() === '-') { take(); const item = unary(); return { text: `-(${item.text})`, derivative: `-(${item.derivative})` }; }
    return power();
  }
  function term() {
    let result = unary();
    while (peek() === '*' || peek() === '/') {
      const op = take();
      const right = unary();
      result = op === '*'
        ? { text: `(${result.text}) * (${right.text})`, derivative: `(${result.derivative}) * (${right.text}) + (${result.text}) * (${right.derivative})` }
        : { text: `(${result.text}) / (${right.text})`, derivative: `((${result.derivative}) * (${right.text}) - (${result.text}) * (${right.derivative})) / (${right.text})^2` };
    }
    return result;
  }
  function expression() {
    let result = term();
    while (peek() === '+' || peek() === '-') {
      const op = take();
      const right = term();
      result = { text: `(${result.text}) ${op} (${right.text})`, derivative: `(${result.derivative}) ${op} (${right.derivative})` };
    }
    return result;
  }

  const result = expression();
  if (cursor !== tokens.length) throw new Error('unexpected token');
  return simplify(result.derivative);
}

function renderEquationPlot(expr) {
  const svg = $('#equation-plot');
  const empty = $('#equation-plot-empty');
  const solution = $('#equation-solution');
  const solutionContent = $('#equation-solution-content');
  svg.innerHTML = '';
  empty.removeAttribute('hidden');
  solution.setAttribute('hidden', '');
  solutionContent.innerHTML = '';

  const match = expr.match(/^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$/);
  const dependent = match ? match[1] : 'y';
  const rhs = (match ? match[2] : expr).replace(/\*\*/g, '^');
  const functionNames = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'cosec', 'tanh', 'exp', 'log', 'relu']);
  const variableMatch = rhs.match(/\b([A-Za-z_]\w*)\b/g)?.find((name) => name !== dependent && !functionNames.has(name));
  if (!variableMatch) return;

  const independent = variableMatch;
  let evaluate;
  try {
    const jsRhs = rhs.replace(/\^/g, '**')
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
    evaluate = new Function(independent, `return (${jsRhs});`);
  } catch (error) {
    return;
  }

  const samples = [];
  for (let i = 0; i <= 240; i++) {
    const input = -4 + (i / 240) * 8;
    try {
      const output = Number(evaluate(input));
      if (Number.isFinite(output)) samples.push({ input, output });
    } catch (error) {
      // Symbolic parameters, such as n in x^n, cannot be plotted numerically.
    }
  }
  if (samples.length < 2) return;

  const width = 640;
  const height = 360;
  const pad = { left: 52, right: 18, top: 18, bottom: 38 };
  const xValues = dependent === 'x' ? samples.map((p) => p.output) : samples.map((p) => p.input);
  const yValues = dependent === 'y' ? samples.map((p) => p.output) : samples.map((p) => p.input);
  const minX = Math.min(-1, ...xValues);
  const maxX = Math.max(1, ...xValues);
  const minY = Math.min(-1, ...yValues);
  const maxY = Math.max(1, ...yValues);
  const scaleX = (value) => pad.left + ((value - minX) / (maxX - minX)) * (width - pad.left - pad.right);
  const scaleY = (value) => height - pad.bottom - ((value - minY) / (maxY - minY)) * (height - pad.top - pad.bottom);
  const make = (tag, attrs) => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    svg.appendChild(element);
    return element;
  };

  if (minX < 0 && maxX > 0) make('line', { x1: scaleX(0), y1: pad.top, x2: scaleX(0), y2: height - pad.bottom, class: 'plot-axis' });
  if (minY < 0 && maxY > 0) make('line', { x1: pad.left, y1: scaleY(0), x2: width - pad.right, y2: scaleY(0), class: 'plot-axis' });
  make('text', { x: width - pad.right - 4, y: scaleY(0) - 8, class: 'plot-axis-label', 'text-anchor': 'end' }).textContent = 'x';
  make('text', { x: scaleX(0) + 8, y: pad.top + 12, class: 'plot-axis-label' }).textContent = 'y';

  const path = samples.map((point, index) => {
    const x = dependent === 'x' ? point.output : point.input;
    const y = dependent === 'y' ? point.output : point.input;
    return `${index === 0 ? 'M' : 'L'} ${scaleX(x).toFixed(2)} ${scaleY(y).toFixed(2)}`;
  }).join(' ');
  make('path', { d: path, class: 'equation-curve' });
  make('text', { x: pad.left, y: height - 10, class: 'plot-caption' }).textContent = `${dependent} = ${rhs}`;
  empty.setAttribute('hidden', '');

  const derivativeAt = (input) => {
    const h = 1e-5;
    return (evaluate(input + h) - evaluate(input - h)) / (2 * h);
  };
  const integrate = (start, end, intervals = 200) => {
    const width = (end - start) / intervals;
    let total = 0;
    for (let i = 0; i <= intervals; i++) {
      const value = Number(evaluate(start + i * width));
      if (!Number.isFinite(value)) return NaN;
      total += (i === 0 || i === intervals ? 1 : i % 2 === 0 ? 2 : 4) * value;
    }
    return total * width / 3;
  };
  const numericDerivative = derivativeAt(2);
  const numericIntegral = integrate(-2, 2);
  let derivativeText = '';
  try { derivativeText = symbolicDerivative(rhs, independent); } catch (error) { derivativeText = 'not available for this expression'; }

  const normalized = rhs.replace(/\s+/g, '').replace(/\*\*/g, '^');
  const squareMatch = normalized.match(/^([A-Za-z_]\w*)\^2$/);
  if (squareMatch && dependent === 'x') {
    const variable = squareMatch[1];
    solutionContent.innerHTML = `
      <p><strong>Given:</strong> <code>x = ${variable}^2</code></p>
      <p>Take the square root of both sides. A square root has two signs:</p>
      <p class="solution-equation"><code>${variable} = &plusmn;&radic;x</code></p>
      <p><strong>Domain:</strong> <code>x &ge; 0</code> &nbsp; <strong>Range:</strong> all real <code>${variable}</code> values.</p>
      <p>The <strong>+</strong> sign makes the upper branch and the <strong>&minus;</strong> sign makes the lower branch. Together they form a parabola opening to the right.</p>
      <p><strong>Calculus:</strong> <code>dx/d${variable} = 2${variable}</code>; <code>&int; x d${variable} = ${variable}^3/3 + C</code>.</p>
    `;
    solution.removeAttribute('hidden');
  } else if (squareMatch && dependent === 'y') {
    const variable = squareMatch[1];
    solutionContent.innerHTML = `
      <p><strong>Given:</strong> <code>y = ${variable}^2</code></p>
      <p>This is already solved for <code>y</code>, so each input <code>${variable}</code> gives one output.</p>
      <p><strong>Domain:</strong> all real <code>${variable}</code> values. &nbsp; <strong>Range:</strong> <code>y &ge; 0</code>.</p>
      <p>The graph is a parabola opening upward with vertex at <code>(0, 0)</code>.</p>
      <p><strong>Calculus:</strong> <code>dy/d${variable} = 2${variable}</code>; <code>&int; y d${variable} = ${variable}^3/3 + C</code>.</p>
    `;
    solution.removeAttribute('hidden');
  } else {
    const linearMatch = normalized.match(/^([+-]?\d*\.?\d*)\*?([A-Za-z_]\w*)([+-]\d*\.?\d+)?$/);
    const coefficientText = linearMatch?.[1];
    const coefficient = coefficientText === '' || coefficientText === '+' ? 1 : coefficientText === '-' ? -1 : Number(coefficientText);
    const constant = linearMatch?.[3] ? Number(linearMatch[3]) : 0;
    const isLinear = linearMatch && linearMatch[2] === independent && Number.isFinite(coefficient) && Number.isFinite(constant);
    let sampleOutput = Number(evaluate(2));
    if (!Number.isFinite(sampleOutput)) sampleOutput = Number(evaluate(1));

    if (isLinear) {
      const signedConstant = constant < 0 ? ` - ${Math.abs(constant)}` : constant > 0 ? ` + ${constant}` : '';
      solutionContent.innerHTML = `
        <p><strong>Step 1: Identify the function.</strong> The equation is linear because <code>${independent}</code> has power 1.</p>
        <p><strong>Step 2: Read the form.</strong> <code>${dependent} = ${coefficient}(${independent})${signedConstant}</code></p>
        <p><strong>Step 3: Find the slope and intercept.</strong> Slope = <code>${coefficient}</code>; y-intercept = <code>${constant}</code>.</p>
        <p><strong>Step 4: Check one point.</strong> Put <code>${independent} = 2</code> into the equation: <code>${dependent} = ${coefficient} &times; 2${signedConstant} = ${formatNumber(sampleOutput)}</code>.</p>
        <p><strong>Calculus:</strong> derivative = <code>${coefficient}</code>; an antiderivative is <code>${coefficient}${independent}^2/2 + ${constant}${independent} + C</code>.</p>
        <p><strong>Conclusion:</strong> every input has exactly one output, so the graph is a straight line.</p>
      `;
    } else {
      solutionContent.innerHTML = `
        <p><strong>Step 1: Identify the function.</strong> <code>${dependent}</code> is expressed in terms of <code>${independent}</code>.</p>
        <p><strong>Step 2: Substitute a value.</strong> Choose <code>${independent} = 2</code> and evaluate the right-hand side.</p>
        <p><strong>Step 3: Calculate:</strong> <code>${dependent} = ${escapeHtml(rhs)} = ${formatNumber(sampleOutput)}</code>.</p>
        <p><strong>Step 4: Repeat</strong> for many input values. The plotted points are the resulting ordered pairs.</p>
        <p><strong>Derivative:</strong> <code>d${dependent}/d${independent} = ${escapeHtml(derivativeText)}</code>.</p>
        <p><strong>Numerical check:</strong> near <code>${independent} = 2</code>, the derivative is approximately <code>${formatNumber(numericDerivative)}</code>. Simpson integration from <code>-2</code> to <code>2</code> gives approximately <code>${formatNumber(numericIntegral)}</code>.</p>
        <p class="solution-note">This expression is plotted and evaluated numerically. Symbolic rearrangement is available for the square and linear forms above.</p>
      `;
    }
    solution.removeAttribute('hidden');
  }
}

function renderGraph(nodes, edges) {
  const svg = $('#graph-svg');
  const { positions, width, height } = layoutGraph(nodes, edges);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = '';

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  // --- edges (drawn first, so nodes sit on top) ---
  edges.forEach((e, i) => {
    const p1 = positions.get(e.source); // child (right edge of its box)
    const p2 = positions.get(e.target); // parent (left edge of its box)
    if (!p1 || !p2) return;

    const x1 = p1.x + NODE_W / 2, y1 = p1.y;
    const x2 = p2.x - NODE_W / 2, y2 = p2.y;
    const midX = (x1 + x2) / 2;
    const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;

    const childNode = nodeById.get(e.source);
    // Approximation: colors by the CHILD's total accumulated gradient
    // (sum of every parent's contribution to it), not this single edge's
    // isolated contribution -- the engine sums contributions per the
    // multivariable chain rule, so an exact per-edge split isn't a
    // meaningful single number when a node has more than one parent.
    let cls = 'edge-path';
    if (childNode) {
      if (childNode.grad > 1e-9) cls += ' grad-pos';
      else if (childNode.grad < -1e-9) cls += ' grad-neg';
    }

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d);
    path.setAttribute('class', cls);
    path.setAttribute('id', `edge-${i}`);
    svg.appendChild(path);
  });

  // --- nodes ---
  for (const n of nodes) {
    const pos = positions.get(n.id);
    if (!pos) continue;
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', `node-card${n.is_leaf ? ' is-leaf' : ''}`);
    g.setAttribute('transform', `translate(${pos.x - NODE_W / 2}, ${pos.y - NODE_H / 2})`);
    const label = graphLabel(n.label);
    g.innerHTML = `
      <rect width="${NODE_W}" height="${NODE_H}" rx="8"></rect>
      <title>${escapeHtml(n.label)}</title>
      <text class="node-label" x="10" y="17" font-size="12">${escapeHtml(label)}${n.op && n.label !== n.op ? ` <tspan fill="var(--chalk-white-dim)">(${escapeHtml(n.op)})</tspan>` : ''}</text>
      <text class="node-data" x="10" y="33" font-size="11">data ${formatNumber(n.data)}</text>
      <text class="node-grad" x="10" y="47" font-size="11">grad ${formatNumber(n.grad)}</text>
    `;
    svg.appendChild(g);
  }

  animateGradientPulse(nodes, edges);
}

// The signature interaction: after backward() runs, send a small chalk
// pulse traveling from the root back toward the leaves along each edge,
// staggered by depth -- so the animation traces the SAME reverse
// topological order Value::backward() actually walks, not a generic
// transition.
function animateGradientPulse(nodes, edges) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const svg = $('#graph-svg');
  const depthOf = new Map();
  {
    const childrenOf = new Map();
    for (const e of edges) {
      if (!childrenOf.has(e.target)) childrenOf.set(e.target, []);
      childrenOf.get(e.target).push(e.source);
    }
    for (const n of nodes) {
      const kids = childrenOf.get(n.id) || [];
      depthOf.set(n.id, kids.length === 0 ? 0 : 1 + Math.max(...kids.map((k) => depthOf.get(k))));
    }
  }
  const maxDepth = Math.max(0, ...nodes.map((n) => depthOf.get(n.id)));
  const STAGGER_MS = 260;
  const PULSE_DUR_MS = 480;

  edges.forEach((e, i) => {
    const targetDepth = depthOf.get(e.target) ?? 0;
    const delayMs = (maxDepth - targetDepth) * STAGGER_MS;

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('r', '4');
    circle.setAttribute('class', 'pulse-dot');
    circle.setAttribute('opacity', '0');
    svg.appendChild(circle);

    const anim = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
    anim.setAttribute('dur', `${PULSE_DUR_MS}ms`);
    anim.setAttribute('begin', `${delayMs}ms`);
    anim.setAttribute('fill', 'freeze');
    // keyPoints 1 -> 0 traverses the path backwards: the path itself runs
    // child -> parent (source -> target), but the gradient physically
    // flows parent -> child, so the pulse must travel target -> source.
    anim.setAttribute('keyPoints', '1;0');
    anim.setAttribute('keyTimes', '0;1');
    anim.setAttribute('calcMode', 'linear');

    const mpath = document.createElementNS('http://www.w3.org/2000/svg', 'mpath');
    mpath.setAttributeNS('http://www.w3.org/1999/xlink', 'href', `#edge-${i}`);
    anim.appendChild(mpath);
    circle.appendChild(anim);

    const opacityAnim = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
    opacityAnim.setAttribute('attributeName', 'opacity');
    opacityAnim.setAttribute('values', '0;1;1;0');
    opacityAnim.setAttribute('keyTimes', '0;0.05;0.85;1');
    opacityAnim.setAttribute('dur', `${PULSE_DUR_MS}ms`);
    opacityAnim.setAttribute('begin', `${delayMs}ms`);
    opacityAnim.setAttribute('fill', 'freeze');
    circle.appendChild(opacityAnim);

    setTimeout(() => circle.remove(), delayMs + PULSE_DUR_MS + 400);
  });
}

// ----------------------------------------------------------------------------
// Training dashboard
// ----------------------------------------------------------------------------

let lossChart = null;

function initSliders() {
  const epochsInput = $('#epochs-input');
  const lrInput = $('#lr-input');
  epochsInput.addEventListener('input', () => { $('#epochs-value').textContent = epochsInput.value; });
  lrInput.addEventListener('input', () => { $('#lr-value').textContent = parseFloat(lrInput.value).toFixed(3); });
}

function downsample(arr, maxPoints) {
  if (arr.length <= maxPoints) return arr.map((v, i) => ({ x: i, y: v }));
  const step = arr.length / maxPoints;
  const out = [];
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.floor(i * step);
    out.push({ x: idx, y: arr[idx] });
  }
  out.push({ x: arr.length - 1, y: arr[arr.length - 1] });
  return out;
}

function renderLossChart(losses) {
  const canvas = $('#loss-chart');
  const ctx = canvas.getContext('2d');
  const points = downsample(losses, 300);

  if (lossChart) lossChart.destroy();
  if (typeof Chart === 'undefined') {
    const width = canvas.width = canvas.clientWidth || 640;
    const height = canvas.height = canvas.clientHeight || 280;
    const padding = 36;
    const maxLoss = Math.max(...losses, 1);
    const x = (index) => padding + (index / Math.max(1, losses.length - 1)) * (width - padding * 1.5);
    const y = (value) => height - padding - (value / maxLoss) * (height - padding * 1.5);

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(237,234,224,0.12)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const gridY = padding + (i / 4) * (height - padding * 1.5);
      ctx.beginPath();
      ctx.moveTo(padding, gridY);
      ctx.lineTo(width - padding / 2, gridY);
      ctx.stroke();
    }

    ctx.strokeStyle = '#6FC7C0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, index) => {
      const pointX = x(point.x);
      const pointY = y(point.y);
      if (index === 0) ctx.moveTo(pointX, pointY);
      else ctx.lineTo(pointX, pointY);
    });
    ctx.stroke();
    return;
  }

  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'MSE loss',
        data: points,
        borderColor: '#6FC7C0',
        backgroundColor: 'rgba(111, 199, 192, 0.12)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'epoch', color: '#EDEAE099', font: { family: 'IBM Plex Mono', size: 11 } },
          ticks: { color: '#EDEAE099', font: { family: 'IBM Plex Mono', size: 10 } },
          grid: { color: 'rgba(237,234,224,0.08)' },
        },
        y: {
          title: { display: true, text: 'loss', color: '#EDEAE099', font: { family: 'IBM Plex Mono', size: 11 } },
          ticks: { color: '#EDEAE099', font: { family: 'IBM Plex Mono', size: 10 } },
          grid: { color: 'rgba(237,234,224,0.08)' },
          beginAtZero: true,
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          titleFont: { family: 'IBM Plex Mono' },
          bodyFont: { family: 'IBM Plex Mono' },
          callbacks: {
            title: (items) => `epoch ${items[0].parsed.x}`,
            label: (item) => `loss ${formatNumber(item.parsed.y)}`,
          },
        },
      },
    },
  });
}

function renderPredictions(predictions) {
  const body = $('#predictions-body');
  body.innerHTML = '';
  predictions.forEach((p) => {
    const rounded = Math.round(p.predicted);
    const correct = rounded === p.target;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${formatNumber(p.input[0])}</td>
      <td>${formatNumber(p.input[1])}</td>
      <td>${formatNumber(p.target)}</td>
      <td>${formatNumber(p.predicted)}</td>
      <td><span class="pred-tag ${correct ? 'pred-tag--correct' : 'pred-tag--off'}">${correct ? 'match' : 'off'}</span></td>
    `;
    body.appendChild(tr);
  });
}

function showTrainerError(message) {
  const banner = $('#trainer-error');
  if (!message) {
    banner.setAttribute('hidden', '');
    banner.textContent = '';
    return;
  }
  banner.textContent = message;
  banner.removeAttribute('hidden');
}

function parseHiddenLayers(text) {
  const parts = text.split(',').map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) throw new Error('enter at least one hidden layer width, e.g. "4, 4"');
  const widths = parts.map((p) => {
    const n = parseInt(p, 10);
    if (!Number.isFinite(n) || n <= 0) throw new Error(`"${p}" isn't a valid layer width`);
    return n;
  });
  return widths;
}

async function trainXor() {
  showTrainerError(null);
  const btn = $('#train-btn');
  const status = $('#trainer-status');
  btn.disabled = true;
  btn.textContent = 'training…';
  status.textContent = 'training in progress — this happens in one request, so the page will pause briefly for larger epoch counts.';

  try {
    const hiddenLayers = parseHiddenLayers($('#hidden-layers-input').value);
    const epochs = parseInt($('#epochs-input').value, 10);
    const learningRate = parseFloat($('#lr-input').value);

    const data = await apiPost('/api/train-xor', {
      epochs,
      learning_rate: learningRate,
      hidden_layers: hiddenLayers,
    });

    renderLossChart(data.losses);
    renderPredictions(data.final_predictions);
    const finalLoss = data.losses[data.losses.length - 1];
    status.textContent = `trained ${data.epochs} epochs · final loss ${formatNumber(finalLoss)}`;
  } catch (err) {
    showTrainerError(err.message);
    status.textContent = '';
  } finally {
    btn.disabled = false;
    btn.textContent = 'train on XOR';
  }
}

$('#train-btn').addEventListener('click', trainXor);

// ----------------------------------------------------------------------------
// Boot
// ----------------------------------------------------------------------------

function init() {
  initSettingsPanel();
  initSliders();
  renderVarRows();
  renderPresets();
  checkHealth();
}

document.addEventListener('DOMContentLoaded', init);
