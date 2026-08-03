const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const hint = document.getElementById("hint");
const idleState = document.getElementById("idleState");
const busyState = document.getElementById("busyState");
const doneState = document.getElementById("doneState");
const errorState = document.getElementById("errorState");
const predClass = document.getElementById("predClass");
const predConf = document.getElementById("predConf");
const bars = document.getElementById("bars");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

let busy = false;
let previewUrl = null;
let activeController = null;

function showState(which) {
  idleState.hidden = which !== "idle";
  busyState.hidden = which !== "busy";
  doneState.hidden = which !== "done";
  errorState.hidden = which !== "error";
}

function setBusy(isBusy) {
  busy = isBusy;
  dropzone.classList.toggle("disabled", isBusy);
  browseBtn.disabled = isBusy;
}

function setStatus(kind, text) {
  statusPill.classList.remove("ready", "error");
  if (kind) statusPill.classList.add(kind);
  statusText.textContent = text;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_ready) {
      setStatus("ready", "Model ready");
    } else {
      setStatus("error", "Model file missing");
    }
  } catch {
    setStatus("error", "Server offline");
  }
}

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!busy) fileInput.click();
});

dropzone.addEventListener("click", () => {
  if (!busy) fileInput.click();
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    if (!busy) dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (file && !busy) handleFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file && !busy) handleFile(file);
  fileInput.value = "";
});

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showState("error");
    errorState.textContent = "Please upload an image file (JPG / PNG).";
    return;
  }

  if (activeController) activeController.abort();
  activeController = new AbortController();

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  previewWrap.hidden = false;
  hint.hidden = true;

  setBusy(true);
  showState("busy");

  const form = new FormData();
  form.append("image", file);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: form,
      signal: activeController.signal,
    });

    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error("Invalid server response");
    }

    if (!res.ok) throw new Error(data.error || "Prediction failed");

    predClass.textContent = data.predicted_class;
    predConf.textContent = `${data.confidence}% confidence`;

    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    bars.innerHTML = entries
      .map(
        ([name, pct]) => `
      <div class="bar-row">
        <span>${name}</span>
        <div class="track"><div class="fill" style="width:0" data-w="${pct}"></div></div>
        <span>${Number(pct).toFixed(1)}%</span>
      </div>`
      )
      .join("");

    showState("done");
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        bars.querySelectorAll(".fill").forEach((el) => {
          el.style.width = `${el.dataset.w}%`;
        });
      });
    });
  } catch (err) {
    if (err.name === "AbortError") return;
    showState("error");
    errorState.textContent = err.message || "Something went wrong.";
  } finally {
    setBusy(false);
    // Safety: never leave spinner visible after request ends
    if (!busyState.hidden && doneState.hidden && errorState.hidden) {
      showState("idle");
    }
  }
}

checkHealth();
