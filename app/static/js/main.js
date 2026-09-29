/* Shared helpers used by every page. */

async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

async function apiGet(path) {
  const response = await fetch(path);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
