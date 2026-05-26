const dropzone = document.getElementById("dropzone");
const dropzoneHint = document.getElementById("dropzone-hint");
const fileInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const form = document.getElementById("upload-form");
const submitBtn = document.getElementById("submit-btn");
const resultCard = document.getElementById("result-card");
const adviceCard = document.getElementById("advice-card");
const errorBox = document.getElementById("error-box");
const adviceBtn = document.getElementById("advice-btn");

let lastResult = null;
let currentObjectURL = null;

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

function showPreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("请选择图片文件");
    return;
  }
  // Revoke any previous object URL to avoid memory leaks
  if (currentObjectURL) URL.revokeObjectURL(currentObjectURL);
  currentObjectURL = URL.createObjectURL(file);

  preview.onload = () => {
    dropzone.classList.add("has-image");
  };
  preview.onerror = () => {
    showError("图片加载失败");
    dropzone.classList.remove("has-image");
  };
  preview.src = currentObjectURL;

  // Update hint text below the image (kept hidden by .has-image, but updated for next selection)
  dropzoneHint.textContent = `已选：${file.name}（${formatSize(file.size)}）`;
}

// Click on the dropzone opens the file picker. Input is OUTSIDE the dropzone
// now, so there is no double-fire from <label> default behavior.
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" || ev.key === " ") {
    ev.preventDefault();
    fileInput.click();
  }
});

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

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  if (!fileInput.files[0]) return showError("请先选择一张图片");

  submitBtn.disabled = true;
  submitBtn.textContent = "识别中…";
  resultCard.hidden = true;
  adviceCard.hidden = true;

  const fd = new FormData();
  fd.append("image", fileInput.files[0]);

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    if (resp.status === 413) {
      return showError("图片太大（超过 10MB），请选小一点的图");
    }
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
  } catch (err) {
    showError("网络错误：" + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "识别";
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
    showError("网络错误：" + err.message);
  } finally {
    adviceBtn.disabled = false;
    adviceBtn.textContent = "获取治理建议";
  }
});
