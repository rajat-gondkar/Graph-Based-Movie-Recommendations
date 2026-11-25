/* Movie KG front end: browse and like films, then get explained recommendations from the knowledge graph. */
(() => {
  "use strict";

  const PAGE = 36;
  const MIN_LIKES = 3;
  const COUNTS = [5, 10, 15, 20];

  const $ = (sel, el = document) => el.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
    },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
  };

  const ICONS = {
    heart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/></svg>',
    heartFilled: '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/></svg>',
    chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>',
    shuffle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg>',
  };

  const state = {
    status: null,
    catalog: [],
    byId: new Map(),
    genres: [],
    themes: [],
    users: [],
    liked: store.get("liked", []),
    k: store.get("k", 10),
    alpha: store.get("alpha_v2", null),   // null until /api/status gives the default (D50)
    userId: store.get("userId", 92),
    view: "browse",
    browseBy: "genre",
    facet: "All",
    query: "",
    sort: "popular",
    shown: PAGE,
    results: new Map(),   // request key -> response
    request: 0,           // ignores stale responses
  };

  const main = $("#main");
  const aside = $("#aside");
  const mobilebar = $("#mobilebar");

  // ---------------------------------------------------------------- API
  async function api(path, options = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    if (!res.ok) {
      let message = `Something went wrong (HTTP ${res.status}).`;
      try {
        const body = await res.json();
        if (typeof body.detail === "string") message = body.detail;
      } catch { /* keep the generic message */ }
      throw new Error(message);
    }
    return res.json();
  }

  // ---------------------------------------------------------------- helpers
  const likedSet = () => new Set(state.liked);
  const year = (y) => (y ? String(y) : "");
  const formatText = (text) => esc(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
  const joinTitles = (titles) => titles.length <= 1 ? titles.join("")
    : `${titles.slice(0, -1).join(", ")} and ${titles[titles.length - 1]}`;

  function posterImg(film, alt = "") {
    if (film.poster) return `<img src="${esc(film.poster)}" alt="${esc(alt)}" loading="lazy" decoding="async">`;
    return `<span class="fallback">${esc(film.title)}</span>`;
  }

  function counted(list, key) {
    const counts = new Map();
    for (const film of list) for (const v of film[key]) counts.set(v, (counts.get(v) || 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }

  // ---------------------------------------------------------------- routing
  function route() {
    const hash = location.hash.replace(/^#\/?/, "");
    if (hash === "users") state.view = "users";
    else if (hash === "results" && state.liked.length >= MIN_LIKES) state.view = "results";
    else state.view = "browse";
    render();
    main.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }

  function render() {
    for (const a of document.querySelectorAll("[data-nav]")) {
      const active = (a.dataset.nav === "users") === (state.view === "users");
      if (active) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    }
    if (state.view === "browse") renderBrowse();
    else if (state.view === "results") renderResults();
    else renderUsers();
    renderAside();
    renderMobilebar();
  }

  // ---------------------------------------------------------------- browse
  function renderBrowse() {
    main.innerHTML = `
      <div class="page-head">
        <h1>Pick films you love</h1>
        <p class="lede">Browse by genre or theme and tap the heart on films you enjoyed. Pick at least ${MIN_LIKES},
          then get recommendations explained through the themes they share.</p>
      </div>
      <div class="controls">
        <div class="segmented" role="group" aria-label="Browse by">
          <button type="button" data-browse="genre" aria-pressed="${state.browseBy === "genre"}">Genre</button>
          <button type="button" data-browse="theme" aria-pressed="${state.browseBy === "theme"}">Theme</button>
        </div>
        <input class="field search" type="search" placeholder="Search all films" aria-label="Search films by title"
               value="${esc(state.query)}" data-search>
        <select class="field" aria-label="Sort" data-sort>
          <option value="popular">Most popular</option>
          <option value="rated">Highest rated</option>
          <option value="newest">Newest</option>
        </select>
      </div>
      <div class="rail" data-rail role="group" aria-label="Filter"></div>
      <p class="meta-line" data-count></p>
      <div class="grid" data-grid></div>
      <div class="more" data-more></div>
      ${footer()}`;
    $("[data-sort]", main).value = state.sort;
    updateBrowse();
  }

  function filtered() {
    let list = state.catalog;
    const q = state.query.trim().toLowerCase();
    if (q) list = list.filter((f) => f.title.toLowerCase().includes(q));
    else if (state.facet !== "All") {
      const key = state.browseBy === "genre" ? "genres" : "themes";
      list = list.filter((f) => f[key].includes(state.facet));
    }
    const sorters = {
      popular: (a, b) => b.n - a.n || b.avg - a.avg,
      rated: (a, b) => b.avg - a.avg || b.n - a.n,
      newest: (a, b) => (b.year || 0) - (a.year || 0) || b.n - a.n,
    };
    return [...list].sort(sorters[state.sort]);
  }

  function updateBrowse() {
    const facets = state.browseBy === "genre" ? state.genres : state.themes;
    const rail = $("[data-rail]", main);
    const searching = state.query.trim() !== "";
    rail.hidden = searching;
    rail.innerHTML = [["All", state.catalog.length], ...facets].map(([name, n]) =>
      `<button type="button" class="chip" data-facet="${esc(name)}" aria-pressed="${state.facet === name}">${esc(name)}<span class="count">${n}</span></button>`
    ).join("");

    const list = filtered();
    const shown = list.slice(0, state.shown);
    const where = searching ? `matching “${esc(state.query.trim())}”`
      : state.facet === "All" ? "in the catalogue" : `in ${esc(state.facet)}`;
    $("[data-count]", main).innerHTML = `${list.length.toLocaleString()} films ${where}`;

    const liked = likedSet();
    $("[data-grid]", main).innerHTML = shown.length ? shown.map((f) => filmTile(f, liked.has(f.id))).join("")
      : `<p class="nothing">No films match that search.</p>`;
    $("[data-more]", main).innerHTML = list.length > shown.length
      ? `<button type="button" class="btn" data-more-btn>Show more films</button>` : "";
  }

  function filmTile(f, liked) {
    const meta = [year(f.year), `★ ${f.avg.toFixed(1)}`].filter(Boolean).join(" · ");
    return `
      <article class="film${liked ? " liked" : ""}" data-film="${f.id}">
        <button type="button" class="poster" data-like="${f.id}" aria-pressed="${liked}"
                aria-label="${liked ? "Remove" : "Like"} ${esc(f.title)}">
          ${posterImg(f)}
          <span class="heart">${liked ? ICONS.heartFilled : ICONS.heart}</span>
        </button>
        <h3 class="film-title">${esc(f.title)}</h3>
        <p class="film-meta">${meta}</p>
      </article>`;
  }

  function toggleLike(id) {
    const i = state.liked.indexOf(id);
    if (i >= 0) state.liked.splice(i, 1); else state.liked.push(id);
    store.set("liked", state.liked);
    const tile = main.querySelector(`[data-film="${id}"]`);
    if (tile) {
      const liked = i < 0;
      tile.classList.toggle("liked", liked);
      const btn = $("[data-like]", tile);
      btn.setAttribute("aria-pressed", String(liked));
      btn.setAttribute("aria-label", `${liked ? "Remove" : "Like"} ${state.byId.get(id).title}`);
      $(".heart", tile).innerHTML = liked ? ICONS.heartFilled : ICONS.heart;
    }
    renderAside();
    renderMobilebar();
  }

  // ---------------------------------------------------------------- results (own profile)
  function renderResults() {
    const liked = [...state.liked].sort((a, b) => a - b);
    const key = `p:${liked.join(",")}:${state.k}:${state.alpha}`;
    const titles = state.liked.map((id) => state.byId.get(id)?.title).filter(Boolean);
    const head = `
      <a class="back" href="#/browse">← Back to browsing</a>
      <div class="page-head">
        <h1>Recommended for you</h1>
        <p class="lede">Because you liked ${esc(joinTitles(titles.slice(0, 4)))}${titles.length > 4 ? ` and ${titles.length - 4} more` : ""}.</p>
        <p class="stats" data-stats></p>
      </div>`;
    showResult(key, head, () => api("/api/recommend", {
      method: "POST", body: JSON.stringify({ liked, k: state.k, alpha: state.alpha }),
    }));
  }

  // ---------------------------------------------------------------- MovieLens users
  function renderUsers() {
    const id = state.userId;
    const key = `u:${id}:${state.k}:${state.alpha}`;
    const head = `
      <div class="page-head">
        <h1>MovieLens user ${id}</h1>
        <p class="lede">One of the ${state.users.length.toLocaleString()} people in the dataset. These picks come from the films they rated four stars or more.</p>
        <p class="stats" data-stats></p>
        <div data-favourites></div>
      </div>`;
    showResult(key, head, () => api(`/api/users/${id}/recommend?k=${state.k}&alpha=${state.alpha}`));
  }

  async function showResult(key, head, fetcher) {
    const token = ++state.request;
    const cached = state.results.get(key);
    if (cached) { paintResult(head, cached.data, cached.ms); return; }
    main.innerHTML = `${head}
      <div class="loading" role="status">
        <div class="progress"></div>
        Walking the knowledge graph and writing explanations…
      </div>`;
    const started = performance.now();
    try {
      const data = await fetcher();
      const ms = performance.now() - started;
      state.results.set(key, { data, ms });
      if (token === state.request) paintResult(head, data, ms);
    } catch (err) {
      if (token === state.request) main.innerHTML = `${head}<div class="banner" role="alert">${formatCode(err.message)}</div>`;
    }
  }

  const formatCode = (text) => esc(text).replace(/`([^`]+)`/g, "<code>$1</code>");

  function paintResult(head, data, ms) {
    const recs = data.recommendations;
    main.innerHTML = `${head}
      <section class="profile" aria-label="Taste profile">
        <div><p class="label">Themes in this taste</p>${bars(data.profile.themes)}</div>
        <div><p class="label">Genres</p>${bars(data.profile.genres)}</div>
      </section>
      <ol class="recs">${recs.map(recItem).join("")}</ol>
      ${footer()}`;
    const filled = recs.filter((r) => r.filled).length;
    const time = ms < 100 ? "under 0.1 s" : `${(ms / 1000).toFixed(1)} s`;
    $("[data-stats]", main).textContent = `${recs.length} recommendations · ${time}`
      + (filled ? ` · ${filled} popular pick${filled > 1 ? "s" : ""} added` : "");
    const fav = $("[data-favourites]", main);
    if (fav && data.topRated) {
      fav.innerHTML = `<div class="favourites" aria-label="Their favourites">${data.topRated.map((m) => `
        <figure class="fav" style="margin:0">${m.poster ? `<img src="${esc(m.poster)}" alt="${esc(m.title)}" loading="lazy">` : `<span class="thumb"></span>`}
          <span>${esc(m.rating)}★</span></figure>`).join("")}</div>`;
    }
  }

  function bars(items) {
    if (!items.length) return `<p class="note">No films rated four stars or more yet.</p>`;
    const max = Math.max(...items.map((i) => i.value));
    return `<ul class="bars">${items.map((i) => `
      <li class="bar"><span class="bar-name" title="${esc(i.name)}">${esc(i.name)}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${(100 * i.value / max).toFixed(1)}%"></span></span></li>`).join("")}</ul>`;
  }

  function recItem(r, i) {
    const match = r.filled ? "Popular pick" : `${Math.round(r.score * 100)}% match`;
    const tags = r.themes.map((t) => `<span class="tag">${esc(t)}</span>`).join("");
    const note = r.source === "template-fallback" ? `<p class="source">The language model was unavailable, so this uses the fixed template.</p>`
      : r.source === "template" && !r.filled ? `<p class="source">Template explanation.</p>` : "";
    const similar = r.evidence.liked_by_the_same_people || [];
    const path = r.evidence.evidence.length || similar.length ? `
      <button type="button" class="toggle" aria-expanded="false" data-path="${i}">Show graph path ${ICONS.chevron}</button>
      <div class="path" hidden data-path-panel="${i}"></div>` : "";
    return `
      <li class="rec">
        <div class="rec-poster">${posterImg(r, r.title)}</div>
        <div class="rec-head">
          <span class="rank">${String(i + 1).padStart(2, "0")}</span>
          <h2>${esc(r.title)}</h2>
          <div class="rec-meta">${[year(r.year), match].filter(Boolean).map(esc).join(" · ")}${tags}</div>
        </div>
        <div class="rec-body">
          <p class="explanation">${formatText(r.explanation)}</p>
          ${note}
          ${path}
        </div>
      </li>`;
  }

  // Graph path: You -> films you liked -> shared themes <- recommended film,
  // plus dashed SIMILAR links: films you liked that the same people also loved (D50)
  function pathSVG(ev) {
    const W = 800, ROW = 40, TOP = 34, LEGEND = 30;
    const liked = [];
    const edges = [];
    for (const item of ev.evidence) {
      for (const m of item.because_you_liked) {
        if (!liked.some((l) => l.title === m.title)) liked.push(m);
        edges.push([m.title, item.theme]);
      }
    }
    const similar = ev.liked_by_the_same_people || [];
    for (const m of similar) if (!liked.some((l) => l.title === m.title)) liked.push(m);
    const fans = new Map(similar.map((m) => [m.title, m.shared_fans]));
    const themes = ev.evidence.map((e) => e.theme);
    const rows = Math.max(liked.length, themes.length, 1);
    const H = TOP + rows * ROW + 8 + LEGEND;
    const yAt = (i, n) => TOP + (rows - n) * ROW / 2 + i * ROW + ROW / 2;
    const cut = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
    const width = (s, max) => Math.min(max, s.length * 6.6 + 24);
    const col = { you: 40, liked: 230, theme: 470, rec: 690 };
    const pos = {};
    liked.forEach((m, i) => { pos[`m:${m.title}`] = { x: col.liked, y: yAt(i, liked.length), w: width(cut(m.title, 26), 190) }; });
    themes.forEach((t, i) => { pos[`t:${t}`] = { x: col.theme, y: yAt(i, themes.length), w: width(cut(t, 22), 160) }; });
    const recTitle = cut(ev.recommended.title, 30);
    const rec = { x: col.rec, y: yAt(0, 1), w: width(recTitle, 216) };
    const you = { x: col.you, y: yAt(0, 1) };
    const curve = (x1, y1, x2, y2, cls) => {
      const mx = (x1 + x2) / 2;
      return `<path class="${cls}" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none"/>`;
    };
    const lines = [
      ...liked.map((m) => { const p = pos[`m:${m.title}`]; return curve(you.x + 18, you.y, p.x - p.w / 2, p.y, "e1"); }),
      ...edges.map(([m, t]) => { const a = pos[`m:${m}`], b = pos[`t:${t}`]; return curve(a.x + a.w / 2, a.y, b.x - b.w / 2, b.y, "e2"); }),
      ...themes.map((t) => { const p = pos[`t:${t}`]; return curve(rec.x - rec.w / 2, rec.y, p.x + p.w / 2, p.y, "e3"); }),
    ].join("");
    const ly = H - 12;
    const legend = `
      <g class="legend">
        <line x1="230" y1="${ly}" x2="262" y2="${ly}" class="e3"/><text x="270" y="${ly + 4}">shares a theme</text>
        ${similar.length ? `<rect x="400" y="${ly - 8}" width="32" height="16" rx="8" class="fan-key"/><text x="440" y="${ly + 4}">also loved by people who loved the recommendation</text>` : ""}
      </g>`;
    const pill = (p, text, cls, title) => `
      <g class="${cls}"><title>${esc(title)}</title>
        <rect x="${p.x - p.w / 2}" y="${p.y - 14}" width="${p.w}" height="28" rx="14"/>
        <text x="${p.x}" y="${p.y + 4}" text-anchor="middle">${esc(text)}</text></g>`;
    const heads = [["You", col.you], ["Films you liked", col.liked], ["Shared themes", col.theme], ["Recommended", col.rec]]
      .map(([t, x]) => `<text class="colhead" x="${x}" y="14" text-anchor="middle">${t.toUpperCase()}</text>`).join("");
    return `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Graph path from the films you liked through shared themes to ${esc(ev.recommended.title)}">
        <style>
          .colhead{font-size:10px;font-weight:600;letter-spacing:.08em;fill:var(--faint)}
          .e1{stroke:var(--line);stroke-width:1.5}.e2{stroke:var(--faint);stroke-width:1.2;opacity:.7}
          .e3{stroke:var(--accent);stroke-width:1.5;opacity:.65}
          .n-fan rect,.fan-key{fill:var(--surface);stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 3}.n-fan text{fill:var(--text);font-size:12px}
          .legend text{font-size:11px;fill:var(--muted)}
          .n-movie rect{fill:var(--surface);stroke:var(--line)}.n-movie text{fill:var(--text);font-size:12px}
          .n-theme rect{fill:var(--accent-soft)}.n-theme text{fill:var(--accent);font-size:12px;font-weight:500}
          .n-rec rect{fill:var(--accent)}.n-rec text{fill:var(--on-accent);font-size:12px;font-weight:600}
          .n-you circle{fill:var(--text)}.n-you text{fill:var(--bg);font-size:11px;font-weight:600}
        </style>
        ${heads}${lines}${legend}
        <g class="n-you"><circle cx="${you.x}" cy="${you.y}" r="18"/><text x="${you.x}" y="${you.y + 4}" text-anchor="middle">You</text></g>
        ${liked.map((m) => pill(pos[`m:${m.title}`], cut(m.title, 26), fans.has(m.title) ? "n-fan" : "n-movie",
          `${m.title}: you rated it ${m.rating}★${fans.has(m.title) ? ` · ${fans.get(m.title)} people loved both` : ""}`)).join("")}
        ${themes.map((t) => pill(pos[`t:${t}`], cut(t, 22), "n-theme", `Theme: ${t}`)).join("")}
        ${pill(rec, recTitle, "n-rec", ev.recommended.title)}
      </svg>`;
  }

  // ---------------------------------------------------------------- sidebar
  function renderAside() {
    const settings = `
      <section class="side-section">
        <p class="label">Settings</p>
        <div class="setting">
          <div class="setting-row"><span class="setting-name">Recommendations</span></div>
          <div class="segmented full" role="group" aria-label="Number of recommendations">
            ${COUNTS.map((n) => `<button type="button" data-k="${n}" aria-pressed="${state.k === n}">${n}</button>`).join("")}
          </div>
        </div>
        <div class="setting">
          <div class="setting-row">
            <label class="setting-name" for="alpha">Graph match vs popularity</label>
            <span class="setting-value" data-alpha-value>${state.alpha.toFixed(1)}</span>
          </div>
          <input id="alpha" type="range" min="0" max="1" step="0.1" value="${state.alpha}" data-alpha>
          <div class="range-ends"><span>Popular</span><span>Graph only</span></div>
        </div>
        <div class="setting">
          <p class="note">${state.status?.llm
            ? "Explanations are written by a language model using only the graph path behind each pick."
            : "No language-model key is set, so explanations use a fixed template."}</p>
        </div>
      </section>`;

    if (state.view === "users") {
      aside.innerHTML = `
        <section class="side-section">
          <p class="label">Choose a user</p>
          <div class="user-picker">
            <select class="field" aria-label="MovieLens user" data-user>
              ${state.users.map((u) => `<option value="${u}"${u === state.userId ? " selected" : ""}>User ${u}</option>`).join("")}
            </select>
            <button type="button" class="icon-btn" data-random title="Random user" aria-label="Pick a random user">${ICONS.shuffle}</button>
          </div>
        </section>${settings}`;
      return;
    }

    const picks = state.liked.map((id) => state.byId.get(id)).filter(Boolean);
    const ready = picks.length >= MIN_LIKES;
    const onResults = state.view === "results";
    aside.innerHTML = `
      <section class="side-section">
        <div class="side-head"><p class="label">Your picks · ${picks.length}</p>
          ${picks.length ? `<button type="button" class="link" data-clear>Clear</button>` : ""}</div>
        ${picks.length ? `<ul class="picks">${picks.map((f) => `
          <li class="pick">${f.poster ? `<img src="${esc(f.poster)}" alt="" loading="lazy">` : `<span class="thumb"></span>`}
            <span class="pick-title">${esc(f.title)}</span>
            <button type="button" class="remove" data-like="${f.id}" aria-label="Remove ${esc(f.title)}">×</button></li>`).join("")}</ul>`
          : `<p class="empty">Nothing yet. Tap the heart on films you enjoyed.</p>`}
        ${onResults ? `<a class="btn btn-block" style="display:grid;place-items:center;text-decoration:none" href="#/browse">Add more films</a>`
          : `<button type="button" class="btn btn-primary btn-block" data-go ${ready ? "" : "disabled"}>Get recommendations</button>
             ${ready ? "" : `<p class="hint">Pick ${MIN_LIKES - picks.length} more to continue</p>`}`}
      </section>${settings}`;
  }

  function renderMobilebar() {
    const show = state.view === "browse";
    mobilebar.hidden = !show;
    if (!show) return;
    const n = state.liked.length;
    mobilebar.innerHTML = `
      <span class="small">${n} picked${n < MIN_LIKES ? ` · ${MIN_LIKES - n} more to go` : ""}</span>
      <button type="button" class="btn btn-primary" data-go ${n >= MIN_LIKES ? "" : "disabled"}>Get recommendations</button>`;
  }

  function footer() {
    return `
      <footer class="foot">
        <p>Every explanation is built from real paths in the knowledge graph: films you liked, the themes they share with
          the recommended film, and films that the same people also loved. The language model only rephrases that evidence
          and is told not to add any other facts.</p>
        <p>Ratings from MovieLens 32M (GroupLens). Plots and posters from TMDB. This product uses the TMDB API but is not
          endorsed or certified by TMDB.</p>
      </footer>`;
  }

  // ---------------------------------------------------------------- events
  document.addEventListener("click", (e) => {
    const t = e.target.closest("button, a");
    if (!t) return;
    if (t.dataset.like) {
      toggleLike(Number(t.dataset.like));
      if (state.view === "results") {
        if (state.liked.length < MIN_LIKES) location.hash = "#/browse"; else renderResults();
      }
      return;
    }
    if (t.dataset.browse) { state.browseBy = t.dataset.browse; state.facet = "All"; state.shown = PAGE; renderBrowse(); return; }
    if (t.dataset.facet) { state.facet = t.dataset.facet; state.shown = PAGE; updateBrowse(); return; }
    if (t.hasAttribute("data-more-btn")) { state.shown += PAGE; updateBrowse(); return; }
    if (t.hasAttribute("data-go")) { location.hash = "#/results"; return; }
    if (t.hasAttribute("data-clear")) {
      state.liked = []; store.set("liked", state.liked);
      if (state.view === "results") location.hash = "#/browse"; else { updateBrowse(); renderAside(); renderMobilebar(); }
      return;
    }
    if (t.dataset.k) { state.k = Number(t.dataset.k); store.set("k", state.k); settingsChanged(); return; }
    if (t.hasAttribute("data-random")) {
      state.userId = state.users[Math.floor(Math.random() * state.users.length)];
      store.set("userId", state.userId); render(); return;
    }
    if (t.dataset.path !== undefined) {
      const panel = main.querySelector(`[data-path-panel="${t.dataset.path}"]`);
      const open = t.getAttribute("aria-expanded") !== "true";
      t.setAttribute("aria-expanded", String(open));
      t.firstChild.textContent = open ? "Hide graph path " : "Show graph path ";
      if (open && !panel.innerHTML) {
        const ev = currentEvidence(Number(t.dataset.path));
        const fans = (ev.liked_by_the_same_people || [])
          .map((m) => `${esc(m.title)} <span class="faint">(${Number(m.shared_fans).toLocaleString()} loved both)</span>`).join(" · ");
        panel.innerHTML = pathSVG(ev) + (fans ? `<p class="path-note">People who loved ${esc(ev.recommended.title)} also loved: ${fans}</p>` : "");
      }
      panel.hidden = !open;
    }
  });

  function currentEvidence(i) {
    const key = state.view === "users" ? `u:${state.userId}:${state.k}:${state.alpha}`
      : `p:${[...state.liked].sort((a, b) => a - b).join(",")}:${state.k}:${state.alpha}`;
    return state.results.get(key).data.recommendations[i].evidence;
  }

  function settingsChanged() {
    if (state.view === "browse") renderAside(); else render();
  }

  let searchTimer;
  document.addEventListener("input", (e) => {
    if (e.target.matches("[data-search]")) {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => { state.query = e.target.value; state.shown = PAGE; updateBrowse(); }, 120);
    } else if (e.target.matches("[data-alpha]")) {
      $("[data-alpha-value]", aside).textContent = Number(e.target.value).toFixed(1);
    }
  });

  document.addEventListener("change", (e) => {
    if (e.target.matches("[data-sort]")) { state.sort = e.target.value; state.shown = PAGE; updateBrowse(); }
    else if (e.target.matches("[data-alpha]")) {
      state.alpha = Math.round(Number(e.target.value) * 10) / 10; store.set("alpha_v2", state.alpha); settingsChanged();
    } else if (e.target.matches("[data-user]")) {
      state.userId = Number(e.target.value); store.set("userId", state.userId); render();
    }
  });

  window.addEventListener("hashchange", route);

  // ---------------------------------------------------------------- start
  async function start() {
    main.innerHTML = `<div class="loading" role="status"><div class="progress"></div>Loading the catalogue…</div>`;
    try {
      const [status, catalog, users] = await Promise.all([api("/api/status"), api("/api/catalog"), api("/api/users")]);
      state.status = status;
      if (state.alpha === null) state.alpha = status.defaultAlpha;
      state.catalog = catalog;
      state.byId = new Map(catalog.map((f) => [f.id, f]));
      state.genres = counted(catalog, "genres");
      state.themes = counted(catalog, "themes");
      state.users = users;
      state.liked = state.liked.filter((id) => state.byId.has(id));
      if (!users.includes(state.userId)) state.userId = users.includes(92) ? 92 : users[0];
      if (status.graphMode !== "full") {
        document.querySelector(".shell").insertAdjacentHTML("beforebegin",
          `<div class="shell"><div class="banner" style="margin-top:var(--s5)">The graph holds training ratings only (eval mode). For the demo, rebuild it with <code>python scripts/05_build_graph.py --mode full --wipe</code>.</div></div>`);
      }
      route();
    } catch (err) {
      main.innerHTML = `<div class="banner" role="alert">${formatCode(err.message)}</div>`;
    }
  }

  start();
})();
