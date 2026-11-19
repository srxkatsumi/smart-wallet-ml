// Helpers shared across the site's pages.

function dirCell(direction, confidence) {
  const up = direction === 'up';
  const cls = up ? 'dir-up' : 'dir-down';
  const arrow = up ? '▲' : '▼';
  return `<span class="dir ${cls}">${arrow} ${Math.round(confidence * 100)}%</span>`;
}

// Confiança calibrada (a principal, igual ao email) com a bruta em letra pequena.
function dirCellCal(direction, conf, raw) {
  const extra = (raw !== undefined && raw !== null && Math.abs(raw - conf) >= 0.005)
    ? `<div style="font-size:9.5px;color:var(--ink-ghost);font-family:var(--mono);margin-top:2px">${document.documentElement.lang === 'pt' ? 'bruta' : 'raw'} ${Math.round(raw * 100)}</div>` : '';
  return dirCell(direction, conf) + extra;
}

function fmtPrice(v) {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(v, digits = 1) {
  if (v === null || v === undefined) return '-';
  return (v * 100).toFixed(digits) + '%';
}


const _MONTHS_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function fmtDate(iso) {
  if (!iso) return '-';
  const [y, m, d] = iso.split('-');
  return `${parseInt(d, 10)} ${_MONTHS_EN[parseInt(m, 10) - 1]} ${y}`;
}

function setActiveNav(page) {
  document.querySelectorAll('.nav-links a').forEach(a => {
    if (a.dataset.page === page) a.setAttribute('aria-current', 'page');
  });
}

// Badge do card Carteira: em coleta até MIN_DIAS dias alinhados, depois o veredito do teste.
function significanceBadge(acc, lang = 'en') {
  const pt = lang === 'pt';
  const est = Object.values(acc.estado || {});
  if (!est.length) return `<span class="badge badge-quiet">${pt ? 'sem significância comprovada' : 'no proven significance'}</span>`;
  if (est.some(e => !e.ready)) {
    const dias = Math.max(...est.map(e => e.n_dias));
    return `<span class="badge badge-quiet">${pt ? 'em coleta' : 'collecting'}: ${dias} ${pt ? 'de' : 'of'} ${acc.min_dias} ${pt ? 'dias para o teste' : 'days for the test'}</span>`;
  }
  if (est.some(e => e.sig)) return `<span class="badge badge-up">${pt ? 'significância comprovada' : 'proven significance'}</span>`;
  return `<span class="badge badge-quiet">${pt ? 'sem significância comprovada' : 'no proven significance'}</span>`;
}

// ✅ predicted ▲ and it rose · 📉 predicted ▼ and it fell · ❌ missed yesterday's D+1 call
function acertouOntemIcon(status) {
  if (status === 'up_correct')   return '✅';
  if (status === 'down_correct') return '📉';
  if (status === 'wrong')        return '❌';
  return '';
}

function trendHtml(delta, { positiveIsGood = true, digits = 1, lang = 'en' } = {}) {
  const pt = lang === 'pt';
  if (delta === null || delta === undefined) {
    return `<span style="color:var(--ink-ghost);font-size:11px">${pt ? 'sem período anterior comparável' : 'no comparable prior period'}</span>`;
  }
  const good = positiveIsGood ? delta >= 0 : delta <= 0;
  const arrow = delta >= 0 ? '↑' : '↓';
  const color = good ? 'var(--up)' : 'var(--down)';
  return `<span style="color:${color};font-size:11px;font-weight:600">${arrow} ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(digits)}pp ${pt ? 'vs. período anterior' : 'vs. prior period'}</span>`;
}

// ── General chart component: plain SVG, no library.
// Supports one or more line series, an optional shaded band (e.g. Bollinger),
// an optional bar series (e.g. MACD histogram), fixed or auto y-domain, and
// horizontal reference lines (e.g. RSI 30/70). One shared crosshair + tooltip
// reads out every series at the hovered X. See dataviz skill: marks-and-anatomy
// (2px lines, ≥8px end marker, hairline recessive gridlines, 10% area fill)
// and interaction (crosshair finds X, one tooltip lists every series).
function renderChart(container, config) {
  const {
    series = [],          // [{name, data:[{date,value}], color, dashed}]
    band = null,          // {upper:[{date,value}], lower:[...], color}
    bars = null,          // {name, data:[{date,value}], color}
    yDomain = 'auto',     // [min,max] or 'auto'
    refLines = [],         // [{value, label}]
    valueFmt = v => v.toFixed(2),
    height = 150,
  } = config;

  const allData = series.length ? series[0].data : (bars ? bars.data : []);
  if (!allData || allData.length < 2) {
    container.innerHTML = '<p style="font-size:11px;color:var(--ink-ghost)">Not enough history yet.</p>';
    return;
  }

  const width = 560;
  const pad = { top: 14, right: 14, bottom: 22, left: 46 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const n = allData.length;

  let yMin, yMax;
  if (yDomain === 'auto') {
    const allVals = [
      ...series.flatMap(s => s.data.map(d => d.value)),
      ...(band ? [...band.upper.map(d => d.value), ...band.lower.map(d => d.value)] : []),
      ...(bars ? bars.data.map(d => d.value) : []),
      ...refLines.map(r => r.value),
    ];
    const min = Math.min(...allVals), max = Math.max(...allVals);
    const p = (max - min) * 0.1 || Math.abs(max) * 0.05 || 1;
    yMin = min - p; yMax = max + p;
  } else {
    [yMin, yMax] = yDomain;
  }

  const x = i => pad.left + (i / (n - 1)) * innerW;
  const y = v => pad.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH;
  const path = data => data.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(d.value).toFixed(1)}`).join(' ');

  const gridVals = [yMin + (yMax - yMin) * 0.15, (yMin + yMax) / 2, yMax - (yMax - yMin) * 0.15];
  const gridlines = gridVals.map(v => `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="#e6e3dc" stroke-width="1"/>
    <text x="${pad.left - 6}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end" font-size="9" font-family="var(--mono)" fill="#a8a39a">${valueFmt(v)}</text>
  `).join('');

  const refLineSvg = refLines.map(r => `
    <line x1="${pad.left}" x2="${width - pad.right}" y1="${y(r.value).toFixed(1)}" y2="${y(r.value).toFixed(1)}"
          stroke="#c9c2b3" stroke-width="1" stroke-dasharray="3,3"/>
    <text x="${width - pad.right}" y="${(y(r.value) - 3).toFixed(1)}" text-anchor="end" font-size="8" fill="#a8a39a">${r.label}</text>
  `).join('');

  const bandSvg = band ? (() => {
    const upperPath = band.upper.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(d.value).toFixed(1)}`).join(' ');
    const lowerPath = band.lower.slice().reverse().map((d, i) => `L${x(band.lower.length - 1 - i).toFixed(1)},${y(d.value).toFixed(1)}`).join(' ');
    return `<path d="${upperPath} ${lowerPath} Z" fill="${band.color}" fill-opacity="0.1" stroke="none"/>`;
  })() : '';

  const barsSvg = bars ? bars.data.map((d, i) => {
    const bw = Math.max(1, innerW / n * 0.6);
    const yZero = y(0);
    const yv = y(d.value);
    const barH = Math.abs(yv - yZero);
    return `<rect x="${(x(i) - bw / 2).toFixed(1)}" y="${Math.min(yv, yZero).toFixed(1)}" width="${bw.toFixed(1)}" height="${barH.toFixed(1)}" fill="${d.value >= 0 ? bars.upColor || 'var(--up)' : bars.downColor || 'var(--down)'}" opacity="0.55"/>`;
  }).join('') : '';

  const linesSvg = series.map(s => `
    <path d="${path(s.data)}" fill="none" stroke="${s.color}" stroke-width="2"
          stroke-linejoin="round" stroke-linecap="round" ${s.dashed ? 'stroke-dasharray="4,3"' : ''}/>
    <circle cx="${x(s.data.length - 1).toFixed(1)}" cy="${y(s.data[s.data.length - 1].value).toFixed(1)}" r="4"
            fill="${s.color}" stroke="var(--card)" stroke-width="2"/>
  `).join('');

  const legendSvg = series.length > 1 ? `
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:4px">
      ${series.map(s => `<span style="font-size:9.5px;color:var(--ink-muted)">
        <span style="display:inline-block;width:10px;height:2px;background:${s.color};vertical-align:middle;margin-right:4px"></span>${s.name}
      </span>`).join('')}
    </div>` : '';

  const uid = 'ch' + Math.random().toString(36).slice(2, 9);

  container.innerHTML = `
    ${legendSvg}
    <div style="position:relative">
      <svg viewBox="0 0 ${width} ${height}" style="width:100%;height:auto;display:block" role="img">
        ${gridlines}
        ${refLineSvg}
        ${bandSvg}
        ${barsSvg}
        ${linesSvg}
        <text x="${pad.left}" y="${height - 6}" font-size="9" fill="#a8a39a">${allData[0].date}</text>
        <text x="${width - pad.right}" y="${height - 6}" text-anchor="end" font-size="9" fill="#a8a39a">${allData[n - 1].date}</text>
        <line id="${uid}-cross" x1="0" x2="0" y1="${pad.top}" y2="${pad.top + innerH}" stroke="#a8a39a" stroke-width="1" opacity="0" pointer-events="none"/>
        <rect x="${pad.left}" y="${pad.top}" width="${innerW}" height="${innerH}" fill="transparent" id="${uid}-hit"/>
      </svg>
      <div id="${uid}-tip" style="position:absolute;top:0;pointer-events:none;opacity:0;transition:opacity .1s;
           background:var(--brand);color:#fff;font-size:10px;font-family:var(--mono);padding:5px 8px;
           border-radius:4px;white-space:nowrap;transform:translate(-50%,-115%);line-height:1.5;z-index:2"></div>
    </div>`;

  const svg = container.querySelector('svg');
  const hit = container.querySelector(`#${uid}-hit`);
  const cross = container.querySelector(`#${uid}-cross`);
  const tip = container.querySelector(`#${uid}-tip`);

  function move(evt) {
    const rect = svg.getBoundingClientRect();
    const scale = width / rect.width;
    const px = (evt.clientX - rect.left) * scale;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - pad.left) / innerW) * (n - 1))));
    const cx = x(i);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('opacity', 1);
    const parts = [allData[i].date];
    series.forEach(s => { if (s.data[i]) parts.push(`${s.name}: ${valueFmt(s.data[i].value)}`); });
    if (bars && bars.data[i]) parts.push(`${bars.name}: ${valueFmt(bars.data[i].value)}`);
    tip.textContent = parts.join(' · ');
    tip.style.opacity = 1;

    // Clamp in real pixels so the tooltip never spills past its own chart
    // container (small multiples are narrow, a naive %-based center easily
    // overflows near either edge and gets clipped by .card's overflow:hidden).
    const tipW = tip.offsetWidth || 90;
    let pxLeft = (cx / width) * rect.width;
    pxLeft = Math.max(tipW / 2 + 2, Math.min(rect.width - tipW / 2 - 2, pxLeft));
    tip.style.left = pxLeft + 'px';
    tip.style.top = Math.max(18, (pad.top / height) * rect.height) + 'px';
  }
  function leave() {
    cross.setAttribute('opacity', 0);
    tip.style.opacity = 0;
  }
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerleave', leave);
}

function corrCellColor(v) {
  let r, g, b;
  if (v >= 0) {
    r = Math.round(0xf6 + (0x1e - 0xf6) * v);
    g = Math.round(0xf3 + (0x7a - 0xf3) * v);
    b = Math.round(0xeb + (0x4c - 0xeb) * v);
  } else {
    const t = Math.abs(v);
    r = Math.round(0xf6 + (0xb8 - 0xf6) * t);
    g = Math.round(0xf3 + (0x45 - 0xf3) * t);
    b = Math.round(0xeb + (0x3a - 0xeb) * t);
  }
  const hex = n => n.toString(16).padStart(2, '0');
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

// ── Detalhe de um ativo da Watchlist: os mesmos gráficos dos Favoritos ──
// Os indicadores são calculados aqui a partir dos fechos, com as mesmas
// fórmulas de docs/export_data.py:technical_indicators (pandas: rolling,
// std com ddof=1, ewm com adjust=False).
function _rolling(arr, n, fn) {
  return arr.map((_, i) => (i + 1 < n ? null : fn(arr.slice(i + 1 - n, i + 1))));
}
function _mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function _std(a) { const m = _mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); }
function _ewm(arr, span) {
  const alpha = 2 / (span + 1); const out = []; let e = null;
  arr.forEach(v => { e = e === null ? v : alpha * v + (1 - alpha) * e; out.push(e); });
  return out;
}

function computeIndicators(dates, closes, keep = 90) {
  const sma20 = _rolling(closes, 20, _mean), sma50 = _rolling(closes, 50, _mean), std20 = _rolling(closes, 20, _std);
  const delta = closes.map((v, i) => (i === 0 ? null : v - closes[i - 1]));
  const gain = _rolling(delta.map(d => (d === null ? null : Math.max(d, 0))), 14, a => (a.includes(null) ? null : _mean(a)));
  const loss = _rolling(delta.map(d => (d === null ? null : Math.max(-d, 0))), 14, a => (a.includes(null) ? null : _mean(a)));
  const rsi = gain.map((g, i) => (g === null || !loss[i] ? null : 100 - 100 / (1 + g / loss[i])));
  const ema12 = _ewm(closes, 12), ema26 = _ewm(closes, 26);
  const macd = ema12.map((v, i) => v - ema26[i]);
  const sig = _ewm(macd, 9);
  const start = Math.max(0, closes.length - keep);
  const s = (arr) => dates.slice(start).map((d, i) => ({ date: d, value: arr[start + i] })).filter(p => p.value !== null && !Number.isNaN(p.value));
  return {
    price: s(closes), sma20: s(sma20), sma50: s(sma50),
    bb_upper: s(sma20.map((m, i) => (m === null ? null : m + 2 * std20[i]))),
    bb_lower: s(sma20.map((m, i) => (m === null ? null : m - 2 * std20[i]))),
    rsi14: s(rsi), macd: s(macd), macd_sig: s(sig), macd_hist: s(macd.map((v, i) => v - sig[i])),
  };
}

// Converte o formato compacto do watchlist_series.json (deslocamento em dias a partir de "base").
function seriesDates(base, offsets) {
  const b = new Date(base + 'T00:00:00Z');
  return offsets.map(o => new Date(b.getTime() + o * 86400000).toISOString().slice(0, 10));
}

// root: elemento onde desenhar; d: entrada de watchlist_series.json; pt: textos em português
function renderTickerDetail(root, base, d, pt = true) {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const brand = css('--brand'), up = css('--up'), down = css('--down');
  const dates = seriesDates(base, d.o);
  const chart = computeIndicators(dates, d.c);
  const accCurve = (d.a || []).map(([o, v]) => ({ date: seriesDates(base, [o])[0], value: v }));
  const L = pt
    ? { price: 'Preço · SMA20 · SMA50 · Bandas de Bollinger (EUR)', rsi: 'RSI (14)', macd: 'MACD', acc: 'Acurácia D+1 (janela de 15)', p: 'Preço', s: 'Sinal', h: 'Histograma', ob: 'sobrecomprado', os: 'sobrevendido', rnd: 'aleatório', none: 'Ainda sem histórico validado suficiente.' }
    : { price: 'Price · SMA20 · SMA50 · Bollinger Bands (EUR)', rsi: 'RSI (14)', macd: 'MACD', acc: 'D+1 accuracy (15-window)', p: 'Price', s: 'Signal', h: 'Histogram', ob: 'overbought', os: 'oversold', rnd: 'random', none: 'Not enough validated history yet.' };
  const h = d.h || {};
  const prev = ['d1', 'd2', 'd3'].map(k => `<div><span class="dir-label">D+${k[1]}</span>${h[k] ? dirCellCal(h[k][0], h[k][1], h[k][2]) : '-'}</div>`).join('');
  root.innerHTML = `
    <div class="panel-label">${L.price}</div><div class="wd-price"></div>
    <div class="panel-grid">
      <div><div class="panel-label">${L.rsi}</div><div class="wd-rsi"></div></div>
      <div><div class="panel-label">${L.macd}</div><div class="wd-macd"></div></div>
      <div><div class="panel-label">${L.acc}</div><div class="wd-acc"></div></div>
    </div>
    <div class="tile-forecasts" style="margin-top:18px;gap:24px">${prev}</div>`;
  renderChart(root.querySelector('.wd-price'), {
    height: 180,
    series: [{ name: L.p, data: chart.price, color: brand }, { name: 'SMA20', data: chart.sma20, color: '#8a7a3a' }, { name: 'SMA50', data: chart.sma50, color: '#5a6472' }],
    band: { upper: chart.bb_upper, lower: chart.bb_lower, color: brand },
    valueFmt: v => '€' + v.toFixed(2),
  });
  renderChart(root.querySelector('.wd-rsi'), {
    height: 130, series: [{ name: 'RSI', data: chart.rsi14, color: brand }], yDomain: [0, 100],
    refLines: [{ value: 70, label: L.ob }, { value: 30, label: L.os }], valueFmt: v => v.toFixed(0),
  });
  renderChart(root.querySelector('.wd-macd'), {
    height: 130,
    series: [{ name: 'MACD', data: chart.macd, color: brand }, { name: L.s, data: chart.macd_sig, color: '#c98a2e', dashed: true }],
    bars: { name: L.h, data: chart.macd_hist, upColor: up, downColor: down }, valueFmt: v => v.toFixed(2),
  });
  const accEl = root.querySelector('.wd-acc');
  if (accCurve.length >= 2) {
    renderChart(accEl, { height: 130, series: [{ name: 'Acc', data: accCurve, color: up }], yDomain: [0, 100], refLines: [{ value: 50, label: L.rnd }], valueFmt: v => v.toFixed(0) + '%' });
  } else {
    accEl.innerHTML = `<p style="font-size:11px;color:var(--ink-ghost)">${L.none}</p>`;
  }
}
