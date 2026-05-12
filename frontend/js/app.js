const dropZone     = document.getElementById("drop-zone");
const fileInput    = document.getElementById("file-input");
const filePreview  = document.getElementById("file-preview");
const fileNameEl   = document.getElementById("file-name");
const fileSizeEl   = document.getElementById("file-size");
const clearBtn     = document.getElementById("clear-btn");
const audioPlayer  = document.getElementById("audio-player");
const errorMsg     = document.getElementById("error-msg");
const analyzeBtn   = document.getElementById("analyze-btn");

const uploadCard   = document.getElementById("upload-card");
const loadingCard  = document.getElementById("loading-card");
const resultsCard  = document.getElementById("results-card");

const resultGenre      = document.getElementById("result-genre");
const resultConfidence = document.getElementById("result-confidence");
const probList         = document.getElementById("prob-list");
const resetBtn         = document.getElementById("reset-btn");

let selectedFile = null;

// Fetch which models are available and disable CNN if not ready
fetch("/models")
  .then((r) => r.json())
  .then((data) => {
    if (!data.cnn) {
      const cnnOpt = document.getElementById("opt-cnn");
      cnnOpt.classList.add("is-disabled");
      cnnOpt.querySelector("input").disabled = true;
      cnnOpt.querySelector(".model-option__desc").textContent = "not available";
    }
  })
  .catch(() => {});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("is-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("is-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatBytes(file.size);
  filePreview.classList.add("is-visible");
  analyzeBtn.disabled = false;
  hideError();

  const url = URL.createObjectURL(file);
  audioPlayer.src = url;
  audioPlayer.classList.add("is-visible");
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  filePreview.classList.remove("is-visible");
  audioPlayer.classList.remove("is-visible");
  audioPlayer.src = "";
  analyzeBtn.disabled = true;
  hideError();
}

clearBtn.addEventListener("click", clearFile);
analyzeBtn.addEventListener("click", analyze);

async function analyze() {
  if (!selectedFile) return;

  hideError();
  showCard("loading");

  const modelType = document.querySelector('input[name="model"]:checked').value;

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("model_type", modelType);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    showCard("results");
    launchConfetti();
  } catch (err) {
    showCard("upload");
    showError(err.message || "Something went wrong. Is the backend running?");
  }
}

function renderResults(data) {
  resultGenre.textContent = data.predicted_genre;
  resultConfidence.textContent = pct(data.confidence);

  probList.innerHTML = "";

  const sorted = Object.entries(data.all_probabilities).sort(
    ([, a], [, b]) => b - a
  );

  sorted.forEach(([genre, prob], idx) => {
    const isTop = idx === 0;
    const item = document.createElement("div");
    item.className = "prob-item";
    item.innerHTML = `
      <span class="prob-item__label">${genre}</span>
      <div class="prob-item__track">
        <div class="prob-item__fill ${isTop ? "prob-item__fill--top" : ""}"
             data-width="${pct(prob, false)}"></div>
      </div>
      <span class="prob-item__value">${pct(prob)}</span>
    `;
    probList.appendChild(item);
  });

  // double rAF so bars start from 0% before transitioning to final width
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll(".prob-item__fill").forEach((bar) => {
        bar.style.width = bar.dataset.width;
      });
    });
  });
}

resetBtn.addEventListener("click", () => {
  clearFile();
  showCard("upload");
});

function showCard(name) {
  uploadCard.hidden  = name !== "upload";
  loadingCard.hidden = name !== "loading";
  resultsCard.hidden = name !== "results";
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("is-visible");
}

function hideError() {
  errorMsg.classList.remove("is-visible");
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 ** 2) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 ** 2).toFixed(1) + " MB";
}

function pct(value, withSign = true) {
  const n = (value * 100).toFixed(1);
  return withSign ? n + "%" : n + "%";
}

function launchConfetti() {
  const COLORS = ["#a855f7", "#7c3aed", "#60a5fa", "#10b981", "#f59e0b", "#f87171"];
  const COUNT  = 120;
  const canvas = document.createElement("canvas");

  Object.assign(canvas.style, {
    position: "fixed", inset: "0", width: "100%", height: "100%",
    pointerEvents: "none", zIndex: "9999",
  });
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d");

  const pieces = Array.from({ length: COUNT }, () => ({
    x:    Math.random() * canvas.width,
    y:    Math.random() * -canvas.height,
    w:    6 + Math.random() * 8,
    h:    10 + Math.random() * 6,
    r:    Math.random() * Math.PI * 2,
    dr:   (Math.random() - 0.5) * 0.2,
    vy:   3 + Math.random() * 4,
    vx:   (Math.random() - 0.5) * 2,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
  }));

  let frame;
  const start = performance.now();

  function draw(now) {
    const elapsed = now - start;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let alive = false;
    for (const p of pieces) {
      p.y  += p.vy;
      p.x  += p.vx;
      p.r  += p.dr;
      if (p.y < canvas.height + 20) alive = true;

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.r);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, 1 - elapsed / 3500);
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }

    if (alive && elapsed < 3500) {
      frame = requestAnimationFrame(draw);
    } else {
      canvas.remove();
    }
  }

  frame = requestAnimationFrame(draw);
}
