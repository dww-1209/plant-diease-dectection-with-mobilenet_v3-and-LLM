const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const form = document.getElementById("upload-form");
const resultCard = document.getElementById("result-card");
const adviceCard = document.getElementById("advice-card");
const errorBox = document.getElementById("error-box");
const adviceBtn = document.getElementById("advice-btn");

let lastResult = null;

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach((e) =>
  dropzone.addEventListener(e, (ev) => {
    ev.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((e) =>
  dropzone.addEventListener(e, () => dropzone.classList.remove("drag"))
);
dropzone.addEventListener("drop", (ev) => {
  ev.preventDefault();
  if (ev.dataTransfer.files[0]) {
    fileInput.files = ev.dataTransfer.files;
    showPreview(ev.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) showPreview(fileInput.files[0]);
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.add("show");
  };
  reader.readAsDataURL(file);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  if (!fileInput.files[0]) return showError("请先选择一张图片");
  const fd = new FormData();
  fd.append("image", fileInput.files[0]);

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!data.success) return showError(data.message || "识别失败");
    lastResult = data.data;
    document.getElementById("r-plant").textContent = data.data.plant_class;
    document.getElementById("r-health").textContent = data.data.health_status;
    document.getElementById("r-disease").textContent = data.data.disease_name;
    document.getElementById("r-degree").textContent = data.data.disease_degree;
    document.getElementById("r-prob").textContent =
      (data.data.probability * 100).toFixed(2) + "%";
    resultCard.hidden = false;
    adviceCard.hidden = true;
  } catch (err) {
    showError(err.message);
  }
});

adviceBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  clearError();
  adviceBtn.disabled = true;
  adviceBtn.textContent = "生成中…";
  try {
    const resp = await fetch("/get_treatment_advice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plant_class: lastResult.plant_class,
        disease_name: lastResult.disease_name,
        disease_degree: lastResult.disease_degree,
        health_status: lastResult.health_status,
      }),
    });
    const data = await resp.json();
    if (!data.success) return showError(data.message || "获取建议失败");
    document.getElementById("advice-text").textContent = data.advice;
    adviceCard.hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    adviceBtn.disabled = false;
    adviceBtn.textContent = "获取治理建议";
  }
});
