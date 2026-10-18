"use strict";
// Plain JavaScript, no build step. All text from the server is inserted with textContent,
// never innerHTML, so report content cannot inject markup or scripts.

const EXAMPLES = [
  "Why did net revenue drop?",
  "Why did the refund rate rise in September 2025?",
  "Why did the discount rate change in November 2025?",
  "What is the weather in Paris?",
];
const POLL_MS = 700;
const POLL_LIMIT_MS = 150000;

const $ = (id) => document.getElementById(id);
const form = $("ask-form"), questionEl = $("question"), askBtn = $("ask-btn");
const statusEl = $("status"), errorEl = $("error"), reportEl = $("report");
const historyEl = $("history"), historyEmpty = $("history-empty");

let activeToken = 0; // lets a newer request cancel an older poll loop

function el(tag, opts = {}, ...children) {
  const node = document.createElement(tag);
  if (opts.className) node.className = opts.className;
  if (opts.text !== undefined) node.textContent = opts.text;
  if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
  for (const c of children) if (c) node.append(c);
  return node;
}
const label = (s) => String(s).replaceAll("_", " ");
const badge = (status) => el("span", { className: `badge b-${status}`, text: label(status) });

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}
function clearMessages() { errorEl.hidden = true; statusEl.hidden = true; }
function showStatus(text) {
  statusEl.replaceChildren(el("span", { className: "spinner" }), el("span", { text }));
  statusEl.hidden = false;
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) {
    let message = `Request failed (${res.status}).`;
    if (body && body.error && body.error.message) message = body.error.message;
    else if (body && Array.isArray(body.detail) && body.detail[0]) message = "Please check the question: " + body.detail[0].msg;
    if (res.status === 429) message += " Try again in a few seconds.";
    throw new Error(message);
  }
  return body;
}

// ------------------------------------------------------------------ rendering
function evidenceRefs(keys) {
  const frag = document.createDocumentFragment();
  for (const key of keys || []) {
    frag.append(el("a", { className: "ref", text: key, attrs: { href: `#ev-${key}`, title: "Jump to evidence" } }));
  }
  return frag;
}

function section(title, content) {
  return [el("h3", { text: title }), content];
}

function renderReport(inv, steps) {
  const r = inv.report || {};
  const parts = [];
  parts.push(el("div", { className: "report-head" },
    el("p", { className: "report-q", text: inv.question }), badge(inv.status)));

  if (r.message) {
    const cls = inv.status === "failed" ? "error" : "notice";
    parts.push(el("p", { className: cls, text: r.message }));
  }
  if (r.summary) parts.push(el("p", { className: "summary", text: r.summary }));

  if (r.findings && r.findings.length) {
    const ul = el("ul", { className: "plain" });
    for (const f of r.findings) ul.append(el("li", {}, el("span", { text: f.text }), evidenceRefs(f.evidence_keys)));
    parts.push(...section("Findings", ul));
  }

  if (r.hypotheses && r.hypotheses.length) {
    const box = el("div");
    for (const h of r.hypotheses) {
      box.append(el("div", { className: "hyp" },
        el("div", {}, badge(h.status), el("span", { text: ` ${h.statement}` })),
        el("p", { className: "muted small" }, el("span", { text: h.reason }), evidenceRefs(h.evidence_keys))));
    }
    parts.push(...section("Hypotheses tested", box));
  }

  if (r.evidence && r.evidence.length) {
    const table = el("table");
    for (const e of r.evidence) {
      table.append(el("tr", { attrs: { id: `ev-${e.key}` } },
        el("td", { className: "key", text: e.key }), el("td", { text: e.description })));
    }
    parts.push(...section("Evidence", table));
  }

  for (const [title, items] of [["Caveats", r.caveats], ["Limitations", r.limitations]]) {
    if (items && items.length) {
      const ul = el("ul", { className: "plain muted" });
      for (const c of items) ul.append(el("li", { text: c }));
      parts.push(el("details", {}, el("summary", { text: `${title} (${items.length})` }), ul));
    }
  }

  if (steps && steps.length) {
    const ul = el("ul", { className: "plain muted small" });
    for (const s of steps) ul.append(el("li", { text: `${s.seq}. ${s.node} - ${label(s.status)} (${s.duration_ms} ms)` }));
    parts.push(el("details", {}, el("summary", { text: "How it worked (workflow trace)" }), ul));
  }

  if (r.stats) {
    const s = r.stats;
    const tokens = s.tokens_in != null ? `, ${s.tokens_in}+${s.tokens_out ?? 0} tokens` : "";
    parts.push(el("div", { className: "stats", text:
      `${s.iterations} round(s), ${s.queries_run} queries, ${s.llm_calls} model call(s)${tokens}, ${s.duration_ms} ms` +
      (s.model ? ` - model: ${s.model}` : "") }));
  }
  reportEl.replaceChildren(...parts);
  reportEl.hidden = false;
}

// -------------------------------------------------------------------- history
function when(iso) {
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function loadHistory() {
  try {
    const data = await api("/investigations?limit=10");
    historyEl.replaceChildren();
    historyEmpty.hidden = data.items.length > 0;
    for (const item of data.items) {
      const btn = el("button", { attrs: { type: "button", title: item.question } },
        el("span", { className: "q", text: item.question }),
        el("span", { className: "meta" }, badge(item.status), el("span", { text: when(item.created_at) })));
      btn.addEventListener("click", () => openInvestigation(item.id));
      historyEl.append(el("li", {}, btn));
    }
  } catch (err) {
    historyEl.replaceChildren();
    historyEmpty.textContent = "History is unavailable right now.";
    historyEmpty.hidden = false;
  }
}

// ------------------------------------------------------------------- workflow
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const isDone = (s) => s !== "queued" && s !== "running";

async function pollUntilDone(id, token) {
  const started = Date.now();
  for (;;) {
    const inv = await api(`/investigations/${id}`);
    if (token !== activeToken) return null; // a newer request took over
    if (isDone(inv.status)) return inv;
    showStatus(inv.status === "queued" ? "Queued..." : "Investigating: planning, querying, checking evidence...");
    if (Date.now() - started > POLL_LIMIT_MS) throw new Error("This is taking longer than expected. Check the history list later.");
    await sleep(POLL_MS);
  }
}

async function showInvestigation(id, token) {
  const inv = await pollUntilDone(id, token);
  if (!inv) return;
  let steps = [];
  try { steps = await api(`/investigations/${id}/steps`); } catch { /* trace is optional */ }
  clearMessages();
  renderReport(inv, steps);
  loadHistory();
}

async function openInvestigation(id) {
  const token = ++activeToken;
  clearMessages();
  reportEl.hidden = true;
  try { await showInvestigation(id, token); } catch (err) { if (token === activeToken) { clearMessages(); showError(err.message); } }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;
  const token = ++activeToken;
  clearMessages();
  reportEl.hidden = true;
  askBtn.disabled = true;
  showStatus("Starting...");
  try {
    const accepted = await api("/investigations", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
    });
    await showInvestigation(accepted.id, token);
  } catch (err) {
    if (token === activeToken) { clearMessages(); showError(err.message); }
  } finally {
    if (token === activeToken) askBtn.disabled = false;
  }
});

questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) form.requestSubmit();
});

for (const text of EXAMPLES) {
  const chip = el("button", { className: "chip", text, attrs: { type: "button" } });
  chip.addEventListener("click", () => { questionEl.value = text; questionEl.focus(); });
  $("examples").append(chip);
}
$("refresh").addEventListener("click", loadHistory);
loadHistory();
