// Sobreposições PT sobre ../assets/site.js (carregado primeiro nesta página).
// Só formatação dependente de idioma/locale vive aqui: dirCell, setActiveNav,
// renderChart e corrCellColor ficam no arquivo partilhado, sem duplicação.

function fmtPrice(v) {
  return v.toLocaleString('pt-PT', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtDate(iso) {
  if (!iso) return '-';
  const [y, m, d] = iso.split('-');
  return `${d}/${m}/${y}`;
}

// ✅ previu ▲ e subiu · 📉 previu ▼ e caiu · ❌ errou a previsão D+1 do dia anterior
function acertouOntemIcon(status) {
  if (status === 'up_correct')   return '✅';
  if (status === 'down_correct') return '📉';
  if (status === 'wrong')        return '❌';
  return '';
}

function trendHtml(delta, { positiveIsGood = true, digits = 1 } = {}) {
  if (delta === null || delta === undefined) {
    return '<span style="color:var(--ink-ghost);font-size:11px">sem período anterior comparável</span>';
  }
  const good = positiveIsGood ? delta >= 0 : delta <= 0;
  const arrow = delta >= 0 ? '↑' : '↓';
  const color = good ? 'var(--up)' : 'var(--down)';
  return `<span style="color:${color};font-size:11px;font-weight:600">${arrow} ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(digits)}pp vs. período anterior</span>`;
}
