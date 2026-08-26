"""
Live-updating knowledge graph view. Run this alongside extract_kg.py (in a
second terminal) and watch the graph grow in your browser as each passage
finishes -- no need to run build_graph.py / visualize_graph.py manually.

It works by re-reading output/extractions.json and rebuilding the graph
fresh on every request to /graph.json (extract_kg.py already checkpoints
that file after every passage, so this always reflects the latest state).
The page polls that endpoint every few seconds and upserts into the live
vis-network graph -- new nodes/edges animate in via physics, existing node
positions are left alone so the layout doesn't jump around.

Fully offline: vis-network's JS/CSS is served locally from pyvis's bundled
copy, no internet connection needed.

Usage:
    python live_server.py
Then open: http://localhost:8765
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyvis

from graph_lib import build_graph_from_extractions, to_node_link, TYPE_COLORS

OUT_DIR = Path(__file__).resolve().parent / "output"
PASSAGES_PATH = OUT_DIR / "subset_passages.json"
EXTRACTIONS_PATH = OUT_DIR / "extractions.json"
PORT = 8765

# Locate vis-network's bundled JS/CSS inside the installed pyvis package (offline, no CDN needed)
_pyvis_lib_dir = Path(pyvis.__file__).resolve().parent / "lib"
_vis_versions = sorted(p for p in _pyvis_lib_dir.glob("vis-*") if p.is_dir())
VIS_DIR = _vis_versions[-1] if _vis_versions else None

HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Live Knowledge Graph</title>
<link rel="stylesheet" href="/vis-network.css">
<script src="/vis-network.min.js"></script>
<style>
  html, body { margin:0; height:100%; background:#111318; color:#e8e8e8;
               font-family: system-ui, sans-serif; overflow:hidden; }
  #status { position:fixed; top:0; left:0; right:0; padding:10px 16px; background:#1b1e26;
            border-bottom:1px solid #333; font-size:14px; z-index:10; display:flex;
            align-items:center; gap:16px; flex-wrap:wrap; }
  #network { position:absolute; top:44px; left:0; right:0; bottom:0; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#4caf50;
         animation: pulse 1.4s infinite; }
  .dot.stale { background:#e07a5f; animation:none; }
  @keyframes pulse { 0%{opacity:1;} 50%{opacity:.25;} 100%{opacity:1;} }
  #legend { display:flex; gap:10px; font-size:12px; color:#aaa; margin-left:auto; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
  #bar { position:fixed; bottom:0; left:0; right:0; height:4px; background:#222; z-index:10; }
  #bar-fill { height:100%; background:#4caf50; width:0%; transition: width .4s ease; }
</style>
</head>
<body>
<div id="status">
  <span class="dot" id="dot"></span>
  <span id="progress-text">connecting...</span>
  <span id="graph-text"></span>
  <span id="updated-text" style="color:#777;"></span>
  <div id="legend"></div>
</div>
<div id="network"></div>
<div id="bar"><div id="bar-fill"></div></div>
<script>
const TYPE_COLORS = __TYPE_COLORS__;

const legend = document.getElementById('legend');
Object.entries(TYPE_COLORS).forEach(([t, c]) => {
  const s = document.createElement('span');
  s.innerHTML = `<span class="swatch" style="background:${c}"></span>${t}`;
  legend.appendChild(s);
});

const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const network = new vis.Network(
  document.getElementById('network'),
  { nodes, edges },
  {
    physics: {
      barnesHut: { gravitationalConstant: -8000, centralGravity: 0.3,
                   springLength: 120, springConstant: 0.02, damping: 0.5 },
      stabilization: false,
    },
    edges: { arrows: 'to', font: { size: 10, color: '#aaa', strokeWidth: 0 },
             color: { color: '#555', highlight: '#fff' }, smooth: { type: 'continuous' } },
    nodes: { font: { color: '#e8e8e8' }, shape: 'dot' },
    interaction: { hover: true },
  }
);

async function poll() {
  const dot = document.getElementById('dot');
  try {
    const res = await fetch('/graph.json', { cache: 'no-store' });
    const g = await res.json();
    dot.classList.remove('stale');

    const deg = {};
    g.edges.forEach(e => {
      deg[e.source] = (deg[e.source] || 0) + 1;
      deg[e.target] = (deg[e.target] || 0) + 1;
    });

    nodes.update(g.nodes.map(n => ({
      id: n.id,
      label: n.name || n.id,
      color: TYPE_COLORS[n.type] || TYPE_COLORS.OTHER,
      size: 10 + 3 * (deg[n.id] || 0),
      title: `<b>${n.name}</b><br>Type: ${n.type}<br>Degree: ${deg[n.id] || 0}<br>` +
             `Sources: ${(n.sources || []).length} passage(s)<br>${(n.descriptions || []).slice(0, 3).join(' | ')}`,
    })));

    edges.update(g.edges.map(e => ({
      id: `${e.source}|${e.target}|${e.key}`,
      from: e.source,
      to: e.target,
      label: (e.relation || '').slice(0, 20),
      title: `${e.relation || ''}<br>${e.description || ''}<br><i>source: ${e.source_passage || ''}</i>`,
    })));

    const meta = g._meta || {};
    const pct = meta.passages_total ? (100 * meta.passages_done / meta.passages_total) : 0;
    document.getElementById('progress-text').textContent =
      `${meta.passages_done ?? 0} / ${meta.passages_total ?? '?'} passages extracted (${pct.toFixed(1)}%)`;
    document.getElementById('graph-text').textContent =
      `${g.nodes.length} entities, ${g.edges.length} relations`;
    document.getElementById('updated-text').textContent =
      `updated ${new Date().toLocaleTimeString()}`;
    document.getElementById('bar-fill').style.width = pct.toFixed(1) + '%';
  } catch (e) {
    dot.classList.add('stale');
    document.getElementById('progress-text').textContent = 'server unreachable -- is live_server.py still running?';
  }
  setTimeout(poll, 3000);
}
poll();
</script>
</body>
</html>
"""


def build_graph_json() -> bytes:
    extractions = []
    if EXTRACTIONS_PATH.exists():
        with open(EXTRACTIONS_PATH, encoding="utf-8") as f:
            extractions = json.load(f)

    passages_total = None
    if PASSAGES_PATH.exists():
        with open(PASSAGES_PATH, encoding="utf-8") as f:
            passages_total = len(json.load(f))

    G = build_graph_from_extractions(extractions)
    payload = to_node_link(G)
    payload["_meta"] = {
        "passages_done": len(extractions),
        "passages_total": passages_total,
    }
    return json.dumps(payload).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - silence default per-request logging
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_PAGE.replace("__TYPE_COLORS__", json.dumps(TYPE_COLORS))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/graph.json":
            self._send(200, build_graph_json(), "application/json", no_cache=True)
        elif self.path == "/vis-network.min.js" and VIS_DIR:
            self._send_file(VIS_DIR / "vis-network.min.js", "application/javascript")
        elif self.path == "/vis-network.css" and VIS_DIR:
            self._send_file(VIS_DIR / "vis-network.css", "text/css")
        else:
            self._send(404, b"Not found", "text/plain")

    def _send_file(self, path: Path, content_type: str):
        try:
            self._send(200, path.read_bytes(), content_type)
        except FileNotFoundError:
            self._send(404, b"Not found", "text/plain")

    def _send(self, code: int, body: bytes, content_type: str, no_cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    if VIS_DIR is None:
        print("[WARN] Could not find pyvis's bundled vis-network assets -- "
              "run `pip install pyvis` first.")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Live graph view running at http://localhost:{PORT}")
    print("Leave this running in its own terminal while extract_kg.py runs in another.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
