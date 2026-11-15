const $ = (id) => document.getElementById(id);
const btn = $("checkBtn");

btn.addEventListener("click", checkUrl);
$("urlInput").addEventListener("keydown", (e) => { if (e.key === "Enter") checkUrl(); });
$("refreshMetrics").addEventListener("click", loadMetrics);

async function checkUrl() {
  $("error").textContent = "";
  const url = $("urlInput").value.trim();
  if (!url) { $("error").textContent = "Please enter a URL."; return; }
  btn.disabled = true; btn.textContent = "Analyzing...";
  try {
    const response = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    $("result").classList.remove("hidden");
    $("prediction").textContent = data.prediction;
    $("prediction").className = data.prediction === "Phishing" ? "danger" : "safe";
    $("score").textContent = `${data.phishing_probability}%`;
    $("modelName").textContent = `Model used: ${data.model}`;
    $("explanationType").textContent = data.explanation_type;
    $("reasons").innerHTML = "";
    data.explanations.forEach(item => {
      const li = document.createElement("li");
      li.textContent = `${item.feature}: ${item.impact}`;
      $("reasons").appendChild(li);
    });
    $("features").innerHTML = "";
    Object.entries(data.features).forEach(([key, value]) => {
      const div = document.createElement("div"); div.className = "feature";
      const label = document.createElement("b"); label.textContent = key.replaceAll("_", " ");
      const val = document.createElement("span"); val.textContent = value;
      div.append(label, val); $("features").appendChild(div);
    });
    $("disclaimer").textContent = data.notice;
  } catch (err) { $("error").textContent = err.message; }
  finally { btn.disabled = false; btn.textContent = "Check URL"; }
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    const rows = await response.json();
    const body = $("metricsBody"); body.innerHTML = "";
    if (!rows.length) { body.innerHTML = '<tr><td colspan="5">No metrics yet. Run train_model.py.</td></tr>'; return; }
    rows.forEach(row => {
      const tr = document.createElement("tr");
      [row.model, row.accuracy, row.precision, row.recall, row.f1].forEach(value => {
        const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  } catch { $("metricsBody").innerHTML = '<tr><td colspan="5">Could not load metrics.</td></tr>'; }
}
loadMetrics();
