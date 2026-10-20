'use strict';
/* ============================================================
   Hypertension Digital Twin v2 – Dashboard Controller
   ============================================================ */

// ── State ──────────────────────────────────────────────────────
const S = {
  patients:      [],
  currentPid:    null,
  currentData:   null,
  gaugeChart:    null,
  bpChart:       null,
  alertIds:      [],
};

// ── Helpers ────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const fmt = (v, d=1) => v != null ? Number(v).toFixed(d) : '--';

function setNow() {
  const now = new Date();
  const iso = new Date(now - now.getTimezoneOffset() * 60000)
    .toISOString().slice(0, 16);
  $('r-ts').value = iso;
}

function updateClock() {
  $('clock').textContent = new Date().toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Patient list ───────────────────────────────────────────────
async function loadPatients() {
  const res = await fetch('/api/patients');
  S.patients = await res.json();
  renderPatientList();
  populatePatientSelector();
  if (S.patients.length > 0 && !S.currentPid) {
    loadPatient(S.patients[0].id);
  }
}

function renderPatientList() {
  const el = $('patient-list');
  if (!S.patients.length) {
    el.innerHTML = '<div style="color:var(--text-3);font-size:.78rem;text-align:center;padding:1rem">No patients yet</div>';
    return;
  }
  el.innerHTML = S.patients.map(p => {
    const hasAlert = p.alert_count > 0;
    const active   = p.id === S.currentPid ? 'active' : '';
    const lastSBP  = p.last_sbp  ? `${Math.round(p.last_sbp)}` : '--';
    const lastDBP  = p.last_dbp  ? `${Math.round(p.last_dbp)}` : '--';
    const alertChip = hasAlert
      ? `<div class="pt-alert-chip">&#9888; ${p.alert_count} alert${p.alert_count>1?'s':''}</div>`
      : '';
    return `<div class="pt-card ${active}" onclick="loadPatient(${p.id})" id="ptcard-${p.id}">
      <div class="pt-name">${p.name}</div>
      <div class="pt-meta">${p.age} yrs &nbsp;&#183;&nbsp; ${p.sex} &nbsp;&#183;&nbsp; BMI ${fmt(p.bmi)}</div>
      <div class="pt-last" style="color:${lastBPColor(parseFloat(p.last_sbp))}">
        ${lastSBP} / ${lastDBP} mmHg
      </div>
      <div style="font-size:.65rem;color:var(--text-3)">${p.reading_count || 0} readings</div>
      ${alertChip}
    </div>`;
  }).join('');
}

function lastBPColor(sbp) {
  if (!sbp) return 'var(--text-2)';
  if (sbp < 120)  return 'var(--green)';
  if (sbp < 130)  return 'var(--cyan)';
  if (sbp < 140)  return 'var(--amber)';
  if (sbp < 180)  return 'var(--orange)';
  return 'var(--red)';
}

function populatePatientSelector() {
  const sel = $('patient-select');
  const cur = sel.value;
  sel.innerHTML = '<option value="">— Select Patient —</option>' +
    S.patients.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
  if (cur) sel.value = cur;
}

// ── Load patient ───────────────────────────────────────────────
async function loadPatient(pid) {
  if (!pid) return;
  S.currentPid = parseInt(pid);
  $('patient-select').value = pid;

  const res  = await fetch(`/api/patients/${pid}`);
  const data = await res.json();
  S.currentData = data;

  // Show twin content
  $('empty-state').style.display  = 'none';
  $('twin-content').style.display = 'flex';

  updatePatientCards();
  renderTwinDashboard(data);
  renderReadingHistory(data.readings);
  renderAlerts(data.alerts);
  renderPatientProfile(data.patient);
  updateAlertBadge(data.alert_count);
  S.alertIds = data.alerts.map(a => a.id);
  renderPatientList(); // refresh active state
}

// ── Twin dashboard render ──────────────────────────────────────
function renderTwinDashboard(data) {
  const readings = data.readings || [];
  const trend    = data.trend    || {};
  const last     = readings.length ? readings[readings.length - 1] : null;

  // Stats strip
  if (last) {
    $('s-sbp').textContent     = Math.round(last.systolic);
    $('s-dbp').textContent     = Math.round(last.diastolic);
    $('s-sbp-lbl').textContent = data.last_bp_label || 'mmHg';
  }
  $('s-readings').textContent = readings.length;
  $('s-days').textContent     = trend.n ? `readings` : 'readings';

  const slope = trend.sbp_slope || 0;
  $('s-trend-val').textContent = (slope >= 0 ? '+' : '') + fmt(slope, 2);
  $('s-trend-dir').textContent = slope >= 0 ? 'mmHg/day ↑' : 'mmHg/day ↓';
  $('s-trend-val').style.color = slope > 0.5 ? 'var(--red)' : slope < -0.5 ? 'var(--green)' : 'var(--blue)';

  // Gauge
  const riskScore = last?.risk_score ?? null;
  const riskLevel = last?.risk_level ?? 'Awaiting Data';
  updateGauge(riskScore, riskLevel);

  // BP stage label
  $('bp-stage').textContent = data.last_bp_label || '—';

  // Trend metadata
  $('tm-avg-sbp').textContent = trend.avg_sbp ? `${trend.avg_sbp}` : '--';
  $('tm-avg-dbp').textContent = trend.avg_dbp ? `${trend.avg_dbp}` : '--';
  $('tm-max-sbp').textContent = trend.max_sbp ? `${trend.max_sbp}` : '--';
  $('tm-slope').textContent   = trend.sbp_slope != null ? (slope >= 0 ? '+' : '') + fmt(slope,2) : '--';

  // Trend direction badge
  const dirBadge = $('trend-dir-badge');
  const dir      = trend.direction || 'Stable';
  const icon     = trend.icon || 'right';
  const arrow    = { up: '↑ Rising', down: '↓ Falling', right: '→ Stable' }[icon] || dir;
  dirBadge.textContent = arrow;
  dirBadge.className   = `trend-dir ${icon}`;

  // BP chart
  buildBPChart(readings);
}

// ── Gauge chart ────────────────────────────────────────────────
function riskColor(level) {
  const map = {
    Low: '#22c55e',
    Moderate: '#f59e0b',
    High: '#f97316',
    Critical: '#ef4444',
    'Lower predicted likelihood': '#22c55e',
    'Moderate predicted likelihood': '#f59e0b',
    'Higher predicted likelihood': '#f97316'
  };
  return map[level] || '#3b82f6';
}

function updateGauge(score, level) {
  const prob  = score ?? 0;
  const color = riskColor(level);

  if (S.gaugeChart) {
    S.gaugeChart.data.datasets[0].data = [prob, 100 - prob];
    S.gaugeChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.05)'];
    S.gaugeChart.update('active');
  } else {
    const ctx = $('gauge-canvas').getContext('2d');
    S.gaugeChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [prob, 100 - prob],
          backgroundColor: [color, 'rgba(255,255,255,0.05)'],
          borderWidth: 0, borderRadius: 4,
        }],
      },
      options: {
        rotation: -90, circumference: 180, cutout: '72%', responsive: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { duration: 900, easing: 'easeInOutQuart' },
      },
    });
  }

  $('gauge-value').textContent = score != null ? `${fmt(score)}%` : '--';

  const badge = $('risk-badge');
  badge.textContent   = level;
  badge.style.color        = color;
  badge.style.borderColor  = color;
  badge.style.background   = color + '18';

  const descs = {
    Low:      'Risk appears within normal range. Maintain healthy lifestyle habits.',
    Moderate: 'Moderate risk detected. Lifestyle modifications recommended.',
    High:     'Elevated risk. Clinical evaluation strongly advised.',
    Critical: 'Critical risk level. Immediate medical consultation required.',
    'Lower predicted likelihood': 'Profile features indicate a lower statistical probability of Stage 1+ elevated BP.',
    'Moderate predicted likelihood': 'Profile features indicate a moderate statistical probability of Stage 1+ elevated BP.',
    'Higher predicted likelihood': 'Profile features indicate a higher statistical probability of Stage 1+ elevated BP.',
    'Awaiting Data': 'Add a BP reading to update the AI risk estimate.',
  };
  $('risk-desc').textContent = descs[level] || '';

  // Glow the gauge card border
  const gc = $('gauge-card');
  gc.style.borderColor = color;
  setTimeout(() => { gc.style.borderColor = ''; }, 2000);
}

// ── BP Timeline chart ──────────────────────────────────────────
function buildBPChart(readings) {
  const ctx    = $('bp-chart').getContext('2d');
  const labels = readings.map(r => {
    try { return new Date(r.timestamp).toLocaleDateString('en-IN', { month:'short', day:'numeric' }); }
    catch { return r.timestamp; }
  });
  const sbpData  = readings.map(r => r.systolic);
  const dbpData  = readings.map(r => r.diastolic);
  const riskData = readings.map(r => r.risk_score);

  // Reference lines
  const sbp130 = Array(readings.length).fill(130);
  const dbp80  = Array(readings.length).fill(80);

  if (S.bpChart) {
    S.bpChart.data.labels = labels;
    S.bpChart.data.datasets[0].data = sbpData;
    S.bpChart.data.datasets[1].data = dbpData;
    S.bpChart.data.datasets[2].data = riskData;
    S.bpChart.data.datasets[3].data = sbp130;
    S.bpChart.data.datasets[4].data = dbp80;
    S.bpChart.update('active');
    return;
  }

  const gradSBP = ctx.createLinearGradient(0, 0, 0, 200);
  gradSBP.addColorStop(0, 'rgba(249,115,22,0.35)');
  gradSBP.addColorStop(1, 'rgba(249,115,22,0)');

  const gradDBP = ctx.createLinearGradient(0, 0, 0, 200);
  gradDBP.addColorStop(0, 'rgba(59,130,246,0.25)');
  gradDBP.addColorStop(1, 'rgba(59,130,246,0)');

  S.bpChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'SBP (mmHg)', data: sbpData,
          borderColor: '#f97316', backgroundColor: gradSBP,
          pointBackgroundColor: '#f97316', pointRadius: 5, pointHoverRadius: 7,
          fill: true, tension: 0.35, borderWidth: 2, yAxisID: 'y',
        },
        {
          label: 'DBP (mmHg)', data: dbpData,
          borderColor: '#3b82f6', backgroundColor: gradDBP,
          pointBackgroundColor: '#3b82f6', pointRadius: 5, pointHoverRadius: 7,
          fill: true, tension: 0.35, borderWidth: 2, yAxisID: 'y',
        },
        {
          label: 'Risk %', data: riskData,
          borderColor: '#8b5cf6', backgroundColor: 'transparent',
          pointBackgroundColor: '#8b5cf6', pointRadius: 3, pointHoverRadius: 6,
          fill: false, tension: 0.35, borderWidth: 1.5,
          borderDash: [4, 3], yAxisID: 'y2',
        },
        {
          label: 'SBP 130 threshold', data: sbp130,
          borderColor: 'rgba(249,115,22,0.3)', borderWidth: 1,
          borderDash: [6, 4], pointRadius: 0, fill: false, yAxisID: 'y',
        },
        {
          label: 'DBP 80 threshold', data: dbp80,
          borderColor: 'rgba(59,130,246,0.3)', borderWidth: 1,
          borderDash: [6, 4], pointRadius: 0, fill: false, yAxisID: 'y',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12, padding: 10 },
        },
        tooltip: {
          backgroundColor: 'rgba(8,13,24,0.95)',
          titleColor: '#f1f5f9', bodyColor: '#94a3b8',
          borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1,
        },
      },
      scales: {
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => v + ' mmHg' },
          title: { display: false },
        },
        y2: {
          position: 'right', min: 0, max: 100,
          grid: { display: false },
          ticks: { color: '#8b5cf6', font: { size: 10 }, callback: v => v + '%' },
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 40 },
        },
      },
      animation: { duration: 800 },
    },
  });
}

// ── Reading history ────────────────────────────────────────────
function renderReadingHistory(readings) {
  const el = $('reading-history');
  if (!readings.length) {
    el.innerHTML = '<div style="color:var(--text-3);font-size:.78rem;text-align:center;padding:1rem">No readings yet</div>';
    return;
  }
  el.innerHTML = [...readings].reverse().map((r, i) => {
    const ts = (() => {
      try { return new Date(r.timestamp).toLocaleString('en-IN', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }); }
      catch { return r.timestamp; }
    })();
    return `<div class="hist-row ${i === 0 ? 'hist-new' : ''}">
      <div>
        <div class="hist-bp" style="color:${lastBPColor(r.systolic)}">${Math.round(r.systolic)} / ${Math.round(r.diastolic)}</div>
        <div class="hist-date">${ts}${r.heart_rate ? ' &nbsp;&#9829; ' + Math.round(r.heart_rate) + ' bpm' : ''}</div>
        ${r.notes ? `<div style="font-size:.65rem;color:var(--text-3)">${r.notes}</div>` : ''}
      </div>
      <span class="hist-pill ${r.risk_level || ''}">${r.risk_level || '--'}</span>
    </div>`;
  }).join('');
}

// ── Alerts render ──────────────────────────────────────────────
function renderAlerts(alerts) {
  const el = $('alerts-list');
  $('alert-count-lbl').textContent = alerts.length ? `(${alerts.length} active)` : '';

  if (!alerts.length) {
    el.innerHTML = '<div class="no-alerts">&#10003; No active alerts — patient is being monitored</div>';
    return;
  }

  el.innerHTML = alerts.map(a => {
    const icon = { danger: '&#128680;', warning: '&#9888;', info: '&#8505;' }[a.severity] || '&#8505;';
    const ts = (() => {
      try { return new Date(a.created_at).toLocaleString('en-IN', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }); }
      catch { return a.created_at; }
    })();
    return `<div class="alert-item ${a.severity}" id="alert-${a.id}">
      <span class="alert-icon">${icon}</span>
      <div style="flex:1">
        <div class="alert-msg">${a.message}</div>
        <div class="alert-time">${ts}</div>
      </div>
      <button class="btn-ack" onclick="ackAlert(${a.id})">Dismiss</button>
    </div>`;
  }).join('');
}

function updateAlertBadge(count) {
  const badge = $('alert-badge');
  badge.textContent = count;
  badge.style.display = count > 0 ? 'flex' : 'none';
}

async function ackAlert(aid) {
  await fetch(`/api/alerts/${aid}/ack`, { method: 'POST' });
  const el = $(`alert-${aid}`);
  if (el) { el.style.opacity = '0'; el.style.transform = 'translateX(10px)'; el.style.transition = '.3s'; setTimeout(() => el.remove(), 300); }
  const remaining = document.querySelectorAll('.alert-item').length - 1;
  updateAlertBadge(remaining);
  $('alert-count-lbl').textContent = remaining > 0 ? `(${remaining} active)` : '';
  if (remaining === 0) {
    $('alerts-list').innerHTML = '<div class="no-alerts">&#10003; No active alerts</div>';
  }
}

async function ackAllAlerts() {
  await Promise.all(S.alertIds.map(id => fetch(`/api/alerts/${id}/ack`, { method: 'POST' })));
  $('alerts-list').innerHTML = '<div class="no-alerts">&#10003; All alerts dismissed</div>';
  $('alert-count-lbl').textContent = '';
  updateAlertBadge(0);
}

function scrollToAlerts() {
  $('alerts-section')?.scrollIntoView({ behavior: 'smooth' });
}

// ── Patient profile card ───────────────────────────────────────
function renderPatientProfile(pt) {
  $('pt-profile-card').style.display = 'block';
  const whr = pt.waist_cm && pt.height_cm ? (pt.waist_cm / pt.height_cm).toFixed(3) : '--';
  $('pt-profile-body').innerHTML = [
    ['&#128100; Age',      `${pt.age} years`],
    ['&#9889; Sex',        pt.sex],
    ['&#129436; BMI',      `${fmt(pt.bmi)} kg/m²`],
    ['&#128084; Waist',    `${pt.waist_cm} cm`],
    ['&#128208; WHR',      whr],
    ['&#129657; Diabetes', pt.diabetes],
    ['&#128138; BP Med',   pt.bp_medication],
    ['&#128197; Enrolled', new Date(pt.created_at).toLocaleDateString('en-IN')],
  ].map(([label, val]) => `
    <div style="display:flex;justify-content:space-between;padding:.25rem 0;border-bottom:1px solid rgba(255,255,255,.04)">
      <span style="color:var(--text-3)">${label}</span>
      <span style="font-weight:600">${val}</span>
    </div>
  `).join('');
}

// ── Patient cards (stats strip) top-level ─────────────────────
function updatePatientCards() {}

// ── Add Reading ────────────────────────────────────────────────
async function addReading() {
  if (!S.currentPid) { alert('Please select a patient first.'); return; }

  const btn = $('btn-add-reading');
  const fb  = $('reading-feedback');
  btn.classList.add('loading');
  btn.innerHTML = '<span class="spin"></span>Saving…';
  fb.style.display = 'none';

  const sbp = parseFloat($('r-sbp-num').value || $('r-sbp').value);
  const dbp = parseFloat($('r-dbp-num').value || $('r-dbp').value);

  if (!sbp || !dbp) {
    btn.classList.remove('loading'); btn.innerHTML = '&#10133; Add Reading';
    return;
  }

  const ts    = $('r-ts').value
    ? new Date($('r-ts').value).toISOString().replace('T', ' ').slice(0, 19)
    : new Date().toISOString().replace('T', ' ').slice(0, 19);

  const payload = {
    systolic:   sbp,
    diastolic:  dbp,
    heart_rate: $('r-hr').value ? parseFloat($('r-hr').value) : null,
    notes:      $('r-notes').value,
    timestamp:  ts,
  };

  try {
    const res  = await fetch(`/api/patients/${S.currentPid}/readings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    // Refresh patient data
    await loadPatient(S.currentPid);
    await loadPatients(); // refresh sidebar

    // Show feedback
    const fc = { Low: 'var(--green)', Moderate: 'var(--amber)', High: 'var(--orange)', Critical: 'var(--red)' };
    fb.style.display = 'block';
    fb.style.color   = fc[data.risk_level] || 'var(--text-2)';
    fb.innerHTML     = `&#10003; Reading saved &nbsp;|&nbsp; Risk: <strong>${data.risk_level}</strong> (${fmt(data.risk_score)}%) &nbsp;|&nbsp; Stage: ${data.bp_label}`;

    // Show any new alerts
    if (data.new_alerts && data.new_alerts.length) {
      fb.innerHTML += `<br/>&#9888; ${data.new_alerts.length} new alert(s) generated`;
    }

    // Reset form
    $('r-hr').value = ''; $('r-notes').value = ''; setNow();

  } catch(e) {
    fb.style.display = 'block'; fb.style.color = 'var(--red)';
    fb.textContent = 'Error: ' + e.message;
  } finally {
    btn.classList.remove('loading'); btn.innerHTML = '&#10133; Add Reading';
  }
}

// ── New patient form ───────────────────────────────────────────
function toggleNewPatientForm() {
  const f = $('new-pt-form');
  f.classList.toggle('open');
}

async function createPatient() {
  const name = $('np-name').value.trim();
  if (!name) { alert('Please enter a patient name.'); return; }

  const payload = {
    name,
    age:           parseFloat($('np-age').value),
    sex:           $('np-sex').value,
    bmi:           parseFloat($('np-bmi').value),
    waist_cm:      parseFloat($('np-waist').value),
    height_cm:     parseFloat($('np-height').value),
    poverty_ratio: parseFloat($('np-poverty').value),
    diabetes:      $('np-diabetes').value,
    bp_medication: $('np-bpmed').value,
  };

  const res  = await fetch('/api/patients', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  $('np-name').value = '';
  toggleNewPatientForm();
  await loadPatients();
  loadPatient(data.id);
}

// ── Model info ─────────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/api/model_info');
    const info = await res.json();
    if ($('mi-auc')) $('mi-auc').textContent = info.roc_auc.toFixed(3);
    if ($('mi-f1'))  $('mi-f1').textContent  = info.f1_score.toFixed(3);
    if ($('mi-rec')) $('mi-rec').textContent  = info.recall.toFixed(3);
  } catch(e) { console.warn('Model info error', e); }
}

// ── Init ───────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  setNow();
  loadPatients();
  loadModelInfo();
});
