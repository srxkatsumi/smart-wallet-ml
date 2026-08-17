// Helpers shared across the site's pages.

function dirCell(direction, confidence) {
  const up = direction === 'up';
  const cls = up ? 'dir-up' : 'dir-down';
  const arrow = up ? '▲' : '▼';
  return `<span class="dir ${cls}">${arrow} ${Math.round(confidence * 100)}%</span>`;
}

function fmtPrice(v) {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(v, digits = 1) {
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

// ✅ predicted ▲ and it rose · 📉 predicted ▼ and it fell · ❌ missed yesterday's D+1 call
function acertouOntemIcon(status) {
  if (status === 'up_correct')   return '✅';
  if (status === 'down_correct') return '📉';
  if (status === 'wrong')        return '❌';
  return '';
}

function trendHtml(delta, { positiveIsGood = true, digits = 1 } = {}) {
  if (delta === null || delta === undefined) {
    return '<span style="color:var(--ink-ghost);font-size:11px">no comparable prior period</span>';
  }
  const good = positiveIsGood ? delta >= 0 : delta <= 0;
  const arrow = delta >= 0 ? '↑' : '↓';
  const color = good ? 'var(--up)' : 'var(--down)';
  return `<span style="color:${color};font-size:11px;font-weight:600">${arrow} ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(digits)}pp vs. prior period</span>`;
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
