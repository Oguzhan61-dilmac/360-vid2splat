/* Miras Gezgini: tekrar ziyaretlerde sayfaların anında açılması için önbellek.
 *
 * - Sayfalar (HTML): önbellekten hemen aç, arka planda tazele (yeni sürüm bir sonraki açılışta).
 * - Kendi verimiz (data/*.json): önbellekten hemen ver, arka planda tazele.
 * - Kitaplıklar ve yazı tipleri (jsDelivr, cdnjs, Google Fonts): sürümlü, önbellekten.
 * - Harita döşemeleri (Esri uydu, AWS arazi, OpenFreeMap): önbellekten, en fazla 3000 döşeme.
 */
const VERSION = "v3";
const PAGES = `pages-${VERSION}`, DATA = `data-${VERSION}`, LIBS = `libs-${VERSION}`, TILES = `tiles-${VERSION}`;
const TILE_LIMIT = 3000;

// İlk ziyarette arka planda: sayfaların iskeleti, küre verisi ve harita kitaplığı önceden önbelleğe
const SHELL = ["./", "index.html", "miras.html", "data/globe.json"];
const SHELL_LIBS = ["https://cdn.jsdelivr.net/npm/maplibre-gl@5.24.0/dist/maplibre-gl.js", "https://cdn.jsdelivr.net/npm/maplibre-gl@5.24.0/dist/maplibre-gl.css"];
self.addEventListener("install", (e) => {
  self.skipWaiting();
  e.waitUntil((async () => {
    try { await (await caches.open(PAGES)).addAll(SHELL.filter((u) => !u.includes("data/"))); } catch {}
    try { await (await caches.open(DATA)).add("data/globe.json"); } catch {}
    try { await (await caches.open(LIBS)).addAll(SHELL_LIBS); } catch {}
  })());
});
self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keep = new Set([PAGES, DATA, LIBS, TILES]);
    for (const k of await caches.keys()) if (!keep.has(k)) await caches.delete(k);
    await self.clients.claim();
  })());
});

const LIB_HOSTS = ["cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com", "fonts.gstatic.com", "static.sketchfab.com"];
const TILE_HOSTS = ["server.arcgisonline.com", "s3.amazonaws.com", "tiles.openfreemap.org"];

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === self.location.origin) {
    if (req.mode === "navigate" || url.pathname.endsWith(".html") || url.pathname.endsWith("/")) e.respondWith(staleWhileRevalidate(req, PAGES));
    else if (url.pathname.includes("/data/")) e.respondWith(staleWhileRevalidate(req, DATA));
    return;
  }
  if (LIB_HOSTS.includes(url.hostname)) e.respondWith(cacheFirst(req, LIBS));
  else if (TILE_HOSTS.includes(url.hostname) && !url.pathname.endsWith("/planet")) e.respondWith(cacheFirst(req, TILES, TILE_LIMIT));
});

async function networkFirst(req, name) {
  const cache = await caches.open(name);
  try {
    const res = await Promise.race([fetch(req), new Promise((_, rej) => setTimeout(() => rej(new Error("yavaş")), 4000))]);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch {
    return (await cache.match(req, { ignoreSearch: true })) || fetch(req);
  }
}

async function staleWhileRevalidate(req, name) {
  const cache = await caches.open(name);
  const key = new Request(req.url.split("?")[0]);
  const hit = await cache.match(key);
  const fresh = fetch(req.url, { cache: "no-cache" }).then((res) => { if (res.ok) cache.put(key, res.clone()); return res; }).catch(() => null);
  return hit || (await fresh) || new Response("{}", { status: 503, headers: { "Content-Type": "application/json" } });
}

async function cacheFirst(req, name, limit) {
  const cache = await caches.open(name);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === "opaque") {
    cache.put(req, res.clone());
    if (limit) trim(cache, limit);
  }
  return res;
}

let trimming = false;
async function trim(cache, limit) {
  if (trimming) return;
  trimming = true;
  try {
    const keys = await cache.keys();
    for (let i = 0; i < keys.length - limit; i++) await cache.delete(keys[i]);
  } finally { trimming = false; }
}
