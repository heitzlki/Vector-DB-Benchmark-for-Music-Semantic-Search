const form = document.getElementById('search-form');
const statusEl = document.getElementById('status');

const dbKeys = ['qdrant', 'milvus', 'weaviate'];
const latEls = Object.fromEntries(dbKeys.map(k => [k, document.getElementById(`lat-${k}`)]));
const listEls = Object.fromEntries(dbKeys.map(k => [k, document.getElementById(`list-${k}`)]));
const errEls = Object.fromEntries(dbKeys.map(k => [k, document.getElementById(`err-${k}`)]));

function clearOutputs() {
  dbKeys.forEach((k) => {
    latEls[k].textContent = '';
    listEls[k].innerHTML = '';
    errEls[k].textContent = '';
  });
}

function renderDBResults(db, data) {
  if (!data) return;
  if (!data.ok) {
    errEls[db].textContent = data.error || 'Error';
    return;
  }
  latEls[db].textContent = `${data.latency_ms?.toFixed(1)} ms`;
  const items = data.results || [];
  listEls[db].innerHTML = items.map((it) => `
    <li>
      <div class="item-title">${escapeHtml(it.track || '(no title)')} – ${escapeHtml(it.artist || 'unknown')}<span class="score">score: ${fmtScore(it.score)}</span></div>
      <div class="item-sub">${escapeHtml(it.genre || '')}</div>
      <div class="item-sub">${escapeHtml(it.text || '')}</div>
    </li>
  `).join('');
}

function fmtScore(s) {
  if (s === undefined || s === null) return '-';
  if (typeof s === 'number') return s.toFixed(4);
  return String(s);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearOutputs();
  const query = document.getElementById('query').value.trim();
  const topk = parseInt(document.getElementById('topk').value || '10', 10);
  const model = document.getElementById('model').value.trim();

  const checkedDbs = Array.from(form.querySelectorAll('input[name="db"]:checked')).map(el => el.value);
  const selected = checkedDbs.slice(0, 3); // keep at most 3

  if (!query) return;
  if (selected.length === 0) {
    statusEl.textContent = 'Select at least one DB';
    return;
  }

  statusEl.textContent = 'Searching…';
  try {
    const res = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, topk, dbs: selected, model }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Render each of the 3 columns if present
    dbKeys.forEach((db) => {
      renderDBResults(db, data.by_db?.[db]);
    });
    statusEl.textContent = 'Done';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Request failed';
    dbKeys.forEach((db) => {
      errEls[db].textContent = 'Request failed';
    });
  }
});

