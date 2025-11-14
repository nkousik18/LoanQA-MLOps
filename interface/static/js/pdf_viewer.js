let pdfDoc = null;
const canvas = document.getElementById("pdfCanvas");
const ctx = canvas.getContext("2d");
const fileInput = document.getElementById("fileInput");

pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/pdfjs/pdf.worker.js";

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Upload PDF to backend (for storage)
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/upload", { method: "POST", body: formData });
  const { path } = await res.json();

  const fileURL = path;
  const loadingTask = pdfjsLib.getDocument(fileURL);
  pdfDoc = await loadingTask.promise;
  renderPage(1);
});

async function renderPage(num) {
  const page = await pdfDoc.getPage(num);
  const viewport = page.getViewport({ scale: 1.5 });
  canvas.height = viewport.height;
  canvas.width = viewport.width;
  await page.render({ canvasContext: ctx, viewport }).promise;
}
