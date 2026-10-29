/* Folio: framework-free vanilla JavaScript */
const $ = (s, r = document) => r.querySelector(s);
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : 'id-' + Math.random().toString(36).slice(2) + Date.now());
const BULLET = '\u2022 ';
const DOT = ' \u00b7 ';

const state = {
  sid: localStorage.getItem('sid') || uid(),
  lang: 'en', mode: 'short', ocr: false, busy: false, ctrl: null,
};
localStorage.setItem('sid', state.sid);

const el = {
  feed: $('#feed'), hero: $('#hero'), q: $('#q'), send: $('#send'), docs: $('#docs'),
  status: $('#status'), drop: $('#drop'), file: $('#file'), prog: $('#prog'),
  progBar: $('#progBar'), progMsg: $('#progMsg'), progPct: $('#progPct'), upErr: $('#upErr'),
  side: $('#side'), scrim: $('#scrim'), ocrHint: $('#ocrHint'),
};

const SEND_ICON = el.send.innerHTML;
const STOP_ICON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

/* ---------- SSE reader ---------- */
async function readSSE(res, on) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let ev = 'message', data = '';
      for (const l of block.split('\n')) {
        if (l.startsWith('event:')) ev = l.slice(6).trim();
        else if (l.startsWith('data:')) data += l.slice(5).trim();
      }
      if (data) on(ev, JSON.parse(data));
    }
  }
}

async function failText(res) {
  try { const j = await res.json(); return j.detail || 'Something went wrong.'; } catch { return 'Something went wrong.'; }
}

/* ---------- answer rendering (always textContent, so it is safe) ---------- */
function renderAnswer(box, text, caret) {
  box.textContent = '';
  let ul = null;
  for (const raw of text.replace(/\r/g, '').split('\n')) {
    const line = raw.trim();
    if (!line) { ul = null; continue; }
    if (line.startsWith(BULLET)) {
      if (!ul) { ul = document.createElement('ul'); box.appendChild(ul); }
      const li = document.createElement('li');
      li.textContent = line.slice(BULLET.length);
      ul.appendChild(li);
      continue;
    }
    ul = null;
    if (line.length < 90 && /:$/.test(line)) {
      const h = document.createElement('h4');
      h.textContent = line.replace(/:$/, '');
      box.appendChild(h);
    } else {
      const p = document.createElement('p');
      if (/^\d+\.\s/.test(line)) p.className = 'num';
      p.textContent = line;
      box.appendChild(p);
    }
  }
  if (caret) {
    const c = document.createElement('span');
    c.className = 'caret';
    (box.lastElementChild && box.lastElementChild.tagName !== 'UL' ? box.lastElementChild : box).appendChild(c);
  }
}

function scrollDown() { el.feed.scrollTop = el.feed.scrollHeight; }

function dropHero() { if (el.hero) { el.hero.remove(); el.hero = null; } }

function addUser(text) {
  dropHero();
  const m = document.createElement('div');
  m.className = 'msg user';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  m.appendChild(b);
  el.feed.appendChild(m);
  scrollDown();
}

function addBot() {
  dropHero();
  const m = document.createElement('div');
  m.className = 'msg bot';
  m.innerHTML = '<div class="page"><div class="ans"><span class="think"><i></i><i></i><i></i></span></div><div class="meta" hidden></div></div>';
  el.feed.appendChild(m);
  scrollDown();
  return { root: m, ans: $('.ans', m), meta: $('.meta', m) };
}

function addMeta(bot, sources, llm, text) {
  bot.meta.hidden = false;
  bot.meta.textContent = '';
  for (const s of sources || []) {
    const c = document.createElement('span');
    c.className = 'chip src';
    c.textContent = `${s.doc}${DOT}page ${s.page}`;
    bot.meta.appendChild(c);
  }
  const cp = document.createElement('button');
  cp.className = 'copy'; cp.type = 'button'; cp.textContent = 'Copy';
  cp.onclick = async () => {
    try { await navigator.clipboard.writeText(text); cp.textContent = 'Copied'; }
    catch { cp.textContent = 'Copy failed'; }
    setTimeout(() => (cp.textContent = 'Copy'), 1600);
  };
  bot.meta.appendChild(cp);
  if (llm) {
    const l = document.createElement('span');
    l.className = 'chip llm';
    l.textContent = `${llm.llm}${DOT}${llm.model}`;
    bot.meta.appendChild(l);
  }
}

/* ---------- chat ---------- */
function setBusy(b) {
  state.busy = b;
  el.send.innerHTML = b ? STOP_ICON : SEND_ICON;
  el.send.classList.toggle('stop', b);
  el.send.setAttribute('aria-label', b ? 'Stop' : 'Send');
}

async function ask(question) {
  if (state.busy) return;
  question = question.trim();
  if (!question) return;
  addUser(question);
  el.q.value = ''; autosize();
  const bot = addBot();
  setBusy(true);
  state.ctrl = new AbortController();

  let text = '', sources = [], llm = null, raf = 0;
  const paint = caret => { renderAnswer(bot.ans, text, caret); scrollDown(); };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: state.ctrl.signal,
      body: JSON.stringify({ session_id: state.sid, question, language: state.lang, mode: state.mode }),
    });
    if (!res.ok) throw new Error(await failText(res));
    await readSSE(res, (ev, d) => {
      if (ev === 'token') {
        text += d.t;
        if (!raf) raf = requestAnimationFrame(() => { raf = 0; paint(true); });
      } else if (ev === 'sources') sources = d.sources;
      else if (ev === 'provider') llm = d;
      else if (ev === 'reset') { text = ''; paint(true); }
      else if (ev === 'error') throw new Error(d.message);
    });
    cancelAnimationFrame(raf);
    paint(false);
    if (text) addMeta(bot, sources, llm, text);
  } catch (e) {
    cancelAnimationFrame(raf);
    if (e.name === 'AbortError') {
      paint(false);
      if (text) addMeta(bot, sources, llm, text); else bot.root.remove();
    } else {
      bot.root.classList.add('error');
      bot.ans.textContent = e.message || 'Something went wrong. Please try again.';
    }
  } finally {
    setBusy(false);
    state.ctrl = null;
    loadStatus();
  }
}

/* ---------- upload ---------- */
function setProg(p, msg) {
  el.progBar.style.width = p + '%';
  el.progPct.textContent = p + '%';
  el.progMsg.textContent = msg;
}

async function uploadFile(file) {
  el.upErr.hidden = true;
  el.prog.hidden = false;
  setProg(1, `Uploading ${file.name}`);
  const fd = new FormData();
  fd.append('file', file);
  fd.append('ocr', state.ocr ? 'true' : 'false');
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await failText(res));
    await readSSE(res, (ev, d) => {
      if (ev === 'progress') setProg(d.pct, d.msg);
      else if (ev === 'error') throw new Error(d.message);
    });
    setProg(100, `${file.name} added`);
  } catch (e) {
    el.upErr.textContent = `${file.name}: ${e.message}`;
    el.upErr.hidden = false;
  }
  await Promise.all([loadDocs(), loadStatus()]);
}

async function uploadMany(files) {
  const pdfs = [...files].filter(f => /\.pdf$/i.test(f.name) || f.type === 'application/pdf');
  if (!pdfs.length) { el.upErr.textContent = 'Please choose PDF files only.'; el.upErr.hidden = false; return; }
  for (const f of pdfs) await uploadFile(f);
  setTimeout(() => { el.prog.hidden = true; }, 1800);
}

/* ---------- documents and status ---------- */
async function loadDocs() {
  const docs = await (await fetch('/api/documents')).json();
  el.docs.textContent = '';
  if (!docs.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No documents added yet.';
    el.docs.appendChild(li);
    return;
  }
  for (const d of docs) {
    const li = document.createElement('li');
    const nm = document.createElement('div');
    nm.className = 'nm'; nm.title = d.name; nm.textContent = d.name;
    const mt = document.createElement('div');
    mt.className = 'mt';
    mt.textContent = `${d.pages} pages${DOT}${d.chunks} chunks`;
    if (d.ocr) { const t = document.createElement('span'); t.className = 'tag'; t.textContent = 'OCR'; mt.appendChild(t); }
    const rm = document.createElement('button');
    rm.className = 'rm'; rm.type = 'button'; rm.setAttribute('aria-label', 'Remove document'); rm.title = 'Remove';
    rm.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 7h14M10 11v6M14 11v6M6 7l1 12h10l1-12M9 7V4h6v3"/></svg>';
    rm.onclick = async () => {
      if (!confirm(`Remove "${d.name}" from the knowledge base?`)) return;
      await fetch('/api/documents/' + d.id, { method: 'DELETE' });
      loadDocs(); loadStatus();
    };
    li.append(nm, rm, mt);
    el.docs.appendChild(li);
  }
}

function statusRow(label, value, ok) {
  const row = document.createElement('div');
  row.className = 'st-row';
  const k = document.createElement('span');
  k.textContent = label;
  const v = document.createElement('b');
  const dot = document.createElement('i');
  dot.className = 'dot' + (ok ? '' : ' off');
  v.append(dot, value);
  row.append(k, v);
  return row;
}

async function loadStatus() {
  try {
    const s = await (await fetch('/api/status')).json();
    el.status.textContent = '';
    el.status.append(
      statusRow('Answer model', s.llm || 'none', s.ready),
      statusRow('Embedding', s.embedding || 'none', s.ready),
      statusRow('Knowledge base', `${s.documents} PDFs${DOT}${s.chunks} chunks`, true),
    );
    const sub = document.createElement('div');
    sub.className = 'st-sub';
    sub.textContent = 'Fallback order: ' + s.providers.map(p => p.name + (p.available ? '' : ' (down)')).join(' > ');
    el.status.appendChild(sub);
  } catch {
    el.status.textContent = 'Cannot reach the server.';
  }
}

async function loadHistory() {
  try {
    const msgs = await (await fetch('/api/history/' + state.sid)).json();
    for (const m of msgs) {
      if (m.role === 'user') addUser(m.content);
      else { const b = addBot(); renderAnswer(b.ans, m.content, false); }
    }
  } catch { /* an empty history is fine */ }
}

/* ---------- events ---------- */
function autosize() {
  el.q.style.height = 'auto';
  el.q.style.height = Math.min(el.q.scrollHeight, 160) + 'px';
}

function bindSeg(id, onPick) {
  const seg = $(id);
  seg.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    seg.querySelectorAll('button').forEach(x => { x.classList.toggle('on', x === b); x.setAttribute('aria-checked', String(x === b)); });
    onPick(b);
  });
}

bindSeg('#langSeg', b => (state.lang = b.dataset.v));
bindSeg('#modeSeg', b => (state.mode = b.dataset.v));
bindSeg('#ocrSeg', b => {
  state.ocr = b.dataset.ocr === 'true';
  el.ocrHint.textContent = state.ocr
    ? 'For scanned or image-based PDFs. Takes longer.'
    : 'Best for PDFs with selectable text. Fast.';
});

el.send.onclick = () => (state.busy ? state.ctrl && state.ctrl.abort() : ask(el.q.value));
el.q.addEventListener('input', autosize);
el.q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(el.q.value); }
});

if (el.hero) el.hero.querySelectorAll('[data-q]').forEach(b => (b.onclick = () => ask(b.dataset.q)));

el.file.onchange = () => { uploadMany(el.file.files); el.file.value = ''; };
el.drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); } });
['dragenter', 'dragover'].forEach(t => el.drop.addEventListener(t, e => { e.preventDefault(); el.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => el.drop.addEventListener(t, e => { e.preventDefault(); el.drop.classList.remove('over'); }));
el.drop.addEventListener('drop', e => uploadMany(e.dataTransfer.files));

$('#clearBtn').onclick = async () => {
  if (state.busy && state.ctrl) state.ctrl.abort();
  await fetch('/api/history/' + state.sid, { method: 'DELETE' });
  state.sid = uid(); localStorage.setItem('sid', state.sid);
  location.reload();
};

const toggleSide = open => { el.side.classList.toggle('open', open); el.scrim.classList.toggle('show', open); };
$('#menuBtn').onclick = () => toggleSide(true);
el.scrim.onclick = () => toggleSide(false);

loadDocs(); loadStatus(); loadHistory();
