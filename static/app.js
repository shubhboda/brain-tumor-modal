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

function showState(which) {
  idleState.hidden = which !== "idle";
  busyState.hidden = which !== "busy";
  doneState.hidden = which !== "done";
  errorState.hidden = which !== "error";
}

function openPicker(e) {
  e.stopPropagation();
  fileInput.click();
}

browseBtn.addEventListener("click", openPicker);
dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
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
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showState("error");
    errorState.textContent = "Please upload an image file.";
    return;
  }

  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.hidden = false;
  hint.hidden = true;

  showState("busy");

  const form = new FormData();
  form.append("image", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    predClass.textContent = data.predicted_class;
    predConf.textContent = `${data.confidence}% confidence`;

    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    bars.innerHTML = entries
      .map(
        ([name, pct]) => `
      <div class="bar-row">
        <span>${name}</span>
        <div class="track"><div class="fill" data-w="${pct}"></div></div>
        <span>${pct.toFixed(1)}%</span>
      </div>`
      )
      .join("");

    showState("done");
    requestAnimationFrame(() => {
      bars.querySelectorAll(".fill").forEach((el) => {
        el.style.width = `${el.dataset.w}%`;
      });
    });
  } catch (err) {
    showState("error");
    errorState.textContent = err.message || "Something went wrong.";
  }
}
