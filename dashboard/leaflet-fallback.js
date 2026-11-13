/* Offline fallback: if Leaflet could not load from the internet, draw the map as plain SVG
   (no street basemap, no zoom) so the dashboard still works in a classroom without Wi-Fi. */
(function () {
  if (window.L) return;
  const NS = 'http://www.w3.org/2000/svg';
  const B = { s: 12.80, n: 13.16, w: 77.42, e: 77.82 };
  function el(t, a) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }
  function Map(id) {
    const host = typeof id === 'string' ? document.getElementById(id) : id;
    host.style.position = 'relative'; host.style.background = '#eef2f7';
    const W = 1000, H = Math.round(1000 * (B.n - B.s) * 110.6 / ((B.e - B.w) * 108.3));
    this.svg = el('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'xMidYMid meet', style: 'width:100%;height:100%' });
    host.appendChild(this.svg);
    const note = document.createElement('div');
    note.textContent = 'Offline map (no internet: street basemap hidden)';
    note.style.cssText = 'position:absolute;top:8px;left:8px;background:#fff;padding:4px 8px;border-radius:6px;font-size:11px;color:#64748b';
    host.appendChild(note);
    this.pop = document.createElement('div');
    this.pop.style.cssText = 'position:absolute;top:8px;right:8px;max-width:260px;background:#fff;padding:8px 10px;border-radius:8px;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.2);display:none';
    host.appendChild(this.pop);
    this.host = host;
    this.px = (lat, lon) => [(lon - B.w) / (B.e - B.w) * W, (B.n - lat) / (B.n - B.s) * H];
  }
  Map.prototype = {
    setView() { return this; }, invalidateSize() { return this; },
    removeLayer(l) { l._remove(); return this; },
    showPopup(html) { this.pop.innerHTML = html + '<div style="text-align:right;margin-top:4px"><a href="#" style="font-size:11px">close</a></div>'; this.pop.style.display = 'block';
      this.pop.querySelector('a').onclick = e => { e.preventDefault(); this.pop.style.display = 'none'; }; }
  };
  function Layer(make) { this._make = make; this._nodes = []; }
  Layer.prototype = {
    addTo(m) { this._map = m; this._nodes = this._make(m); this._nodes.forEach(n => { m.svg.appendChild(n); this._wire(n); }); if (this._style) this.setStyle(this._style); return this; },
    _wire(n) { if (this._tip) { const t = el('title', {}); t.textContent = this._tip; n.appendChild(t); }
      if (this._popup !== undefined) n.onclick = () => this._map.showPopup(this._popup); n.style.cursor = 'pointer'; },
    _remove() { this._nodes.forEach(n => n.remove()); this._nodes = []; },
    bindTooltip(t) { this._tip = t; return this; }, openTooltip() { return this; },
    bindPopup(h) { this._popup = h; this._nodes.forEach(n => this._wire(n)); return this; },
    setStyle(s) { this._style = Object.assign(this._style || {}, s); this._nodes.forEach(n => {
      if (s.fillColor) n.setAttribute('fill', s.fillColor); if (s.fillOpacity !== undefined) n.setAttribute('fill-opacity', s.fillOpacity); }); return this; }
  };
  function Group(layers) { this.layers = layers; }
  Group.prototype = { addTo(m) { this.layers.forEach(l => l.addTo(m)); return this; }, _remove() { this.layers.forEach(l => l._remove()); } };
  window.L = {
    map: id => new Map(id),
    tileLayer: () => ({ addTo() { return this; } }),
    rectangle: (b, o) => { const l = new Layer(m => { const [x1, y1] = m.px(b[1][0], b[0][1]); const [x2, y2] = m.px(b[0][0], b[1][1]);
      return [el('rect', { x: x1, y: y1, width: x2 - x1 + 0.4, height: y2 - y1 + 0.4, fill: '#ccc', 'fill-opacity': o.fillOpacity || 0.6 })]; }); return l; },
    circleMarker: (ll, o) => new Layer(m => { const [x, y] = m.px(ll[0], ll[1]);
      return [el('circle', { cx: x, cy: y, r: (o.radius || 4) * 1.3, stroke: o.color || '#000', 'stroke-width': o.weight || 1, fill: o.fillColor || 'none', 'fill-opacity': o.fillOpacity || 0 })]; }),
    marker: (ll) => new Layer(m => { const [x, y] = m.px(ll[0], ll[1]);
      const g = el('g', {}); g.appendChild(el('rect', { x: x - 7, y: y - 7, width: 14, height: 14, rx: 3, fill: '#15803d' }));
      const t = el('text', { x: x, y: y + 4.5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: '#fff' }); t.textContent = '+'; g.appendChild(t); return [g]; }),
    divIcon: o => o,
    layerGroup: ls => new Group(ls),
    control: () => ({ onAdd: null, addTo(m) { const d = this.onAdd(m); d.style.cssText += ';position:absolute;bottom:8px;right:8px'; m.host.appendChild(d); return this; } }),
    DomUtil: { create: (t, c) => { const d = document.createElement(t); d.className = c; return d; } }
  };
})();
