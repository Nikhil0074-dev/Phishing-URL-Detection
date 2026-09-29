/* Logic for the URL checker on the home page. */

(function () {
  const input = document.getElementById("url-input");
  const modelSelect = document.getElementById("model-select");
  const checkButton = document.getElementById("check-button");
  const compareButton = document.getElementById("compare-button");
  const message = document.getElementById("message");
  const result = document.getElementById("result");
  const verdictPanel = document.getElementById("verdict-panel");
  const riskScore = document.getElementById("risk-score");
  const verdictLabel = document.getElementById("verdict-label");
  const resultUrl = document.getElementById("result-url");
  const riskMeter = document.getElementById("risk-meter");
  const resultMeta = document.getElementById("result-meta");
  const factorsBody = document.getElementById("factors");
  const explanationNote = document.getElementById("explanation-note");
  const indicatorsList = document.getElementById("indicators");
  const noIndicators = document.getElementById("no-indicators");
  const modelsPanel = document.getElementById("models-panel");
  const modelRowsBody = document.getElementById("model-rows");
  const consensusNote = document.getElementById("consensus-note");

  document.querySelectorAll(".sample").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.textContent.trim();
      input.focus();
    });
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") runCheck();
  });
  checkButton.addEventListener("click", runCheck);
  compareButton.addEventListener("click", runCompare);

  function setBusy(busy) {
    checkButton.disabled = busy;
    compareButton.disabled = busy;
    checkButton.textContent = busy ? "Checking..." : "Check URL";
  }

  function renderVerdict(prediction, riskPercent, url, meta) {
    verdictPanel.classList.remove("verdict-legitimate", "verdict-phishing");
    verdictPanel.classList.add(
      prediction === "phishing" ? "verdict-phishing" : "verdict-legitimate"
    );
    riskScore.textContent = `${riskPercent}%`;
    verdictLabel.textContent = prediction === "phishing" ? "Phishing" : "Legitimate";
    resultUrl.textContent = url;
    riskMeter.style.width = `${riskPercent}%`;
    resultMeta.textContent = meta;
  }

  function renderFactors(factors) {
    factorsBody.innerHTML = "";
    factors.forEach((factor) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${escapeHtml(factor.label)}</td>
        <td class="numeric">${formatNumber(factor.value, 2)}</td>
        <td><span class="factor-level level-${factor.level}">${factor.level}</span></td>
        <td class="numeric">${formatNumber(factor.importance, 3)}</td>
      `;
      factorsBody.appendChild(row);
    });
  }

  function renderIndicators(indicators) {
    indicatorsList.innerHTML = "";
    if (!indicators.length) {
      noIndicators.textContent = "No notable warning patterns were found in this URL.";
      return;
    }
    noIndicators.textContent = "";
    indicators.forEach((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      indicatorsList.appendChild(item);
    });
  }

  function renderModelRows(modelResults, consensus) {
    modelRowsBody.innerHTML = "";
    modelResults.forEach((item) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${escapeHtml(item.model_display_name)}</td>
        <td><span class="factor-level level-${item.prediction === "phishing" ? "HIGH" : "LOW"}">${item.prediction}</span></td>
        <td class="numeric">${item.risk_percent}%</td>
        <td class="numeric">${formatNumber(item.prediction_time_ms, 2)}</td>
      `;
      modelRowsBody.appendChild(row);
    });
    consensusNote.textContent =
      `Consensus: ${consensus.prediction} ` +
      `(${consensus.phishing_votes} of ${consensus.total_models} models flagged phishing, ` +
      `mean risk ${formatPercent(consensus.mean_risk_score)}).`;
    modelsPanel.classList.remove("hidden");
  }

  async function runCheck() {
    const url = input.value.trim();
    message.textContent = "";
    if (!url) {
      message.textContent = "Enter a URL first.";
      return;
    }
    modelsPanel.classList.add("hidden");
    setBusy(true);
    try {
      const payload = { url, model: modelSelect.value || undefined };
      const data = await apiPost("/api/predict", payload);
      result.classList.remove("hidden");
      renderVerdict(
        data.prediction,
        data.risk_percent,
        data.url,
        `${data.model_display_name} - risk band: ${data.risk_band} - ${formatNumber(data.prediction_time_ms, 2)} ms`
      );
      renderFactors(data.contributing_factors);
      explanationNote.textContent = data.explanation_note;
      renderIndicators(data.indicators);
    } catch (error) {
      message.textContent = error.message;
    } finally {
      setBusy(false);
    }
  }

  async function runCompare() {
    const url = input.value.trim();
    message.textContent = "";
    if (!url) {
      message.textContent = "Enter a URL first.";
      return;
    }
    setBusy(true);
    try {
      const data = await apiPost("/api/predict/compare", { url });
      result.classList.remove("hidden");
      const consensus = data.consensus;
      renderVerdict(
        consensus.prediction,
        Math.round(consensus.mean_risk_score * 100),
        data.url,
        `Consensus across ${consensus.total_models} models`
      );
      renderFactors(data.contributing_factors);
      explanationNote.textContent =
        "Factors shown use the first model's importance; every model's vote is listed below.";
      renderIndicators(data.indicators);
      renderModelRows(data.model_results, consensus);
    } catch (error) {
      message.textContent = error.message;
    } finally {
      setBusy(false);
    }
  }
})();
