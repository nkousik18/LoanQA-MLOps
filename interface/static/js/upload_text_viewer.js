/* ===========================================================
   upload_text_viewer.js
   -----------------------------------------------------------
   Handles document upload → OCR → full text display in viewer.
   Works with Flask routes:
   - POST /api/upload → returns {"content_path": "..."}
   - GET  /api/read_text?path=... → streams full extracted text
   =========================================================== */

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const docContainer = document.getElementById("document-content");

/* ---------------------- Upload Logic ---------------------- */
uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    showModal("⚠️ Please choose a file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  showModal("⏳ Uploading and extracting text...");

  try {
    // Step 1️⃣: Upload to backend for OCR + indexing
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
    const data = await response.json();

    if (data.error) throw new Error(data.error);

    showModal("✅ Document processed successfully. Loading full text...");

    // Step 2️⃣: Fetch the extracted text directly from file path
    if (!data.content_path) {
      throw new Error("Server did not return content_path.");
    }

    const textResponse = await fetch(`/api/read_text?path=${encodeURIComponent(data.content_path)}`);
    if (!textResponse.ok) throw new Error(`Failed to read file: ${textResponse.status}`);

    const fullText = await textResponse.text();

    // Step 3️⃣: Render full text into the viewer
    renderExtractedText(fullText);

  } catch (error) {
    console.error("[UPLOAD ERROR]", error);
    showModal(`❌ Upload failed: ${error.message}`);
  }
});

/* ---------------------- Renderer ---------------------- */
function renderExtractedText(text) {
  if (!docContainer) return;

  // Sanitize and render
  docContainer.textContent = text;

  // Apply styling for readability
  docContainer.style.whiteSpace = "pre-wrap";
  docContainer.style.lineHeight = "1.6";
  docContainer.style.fontSize = "1rem";
  docContainer.style.padding = "1rem";
  docContainer.style.color = "#e0f7fa";
  docContainer.style.overflowY = "auto";
  docContainer.style.maxHeight = "80vh";

  showModal("📄 Full document loaded successfully!");
}

/* ---------------------- Helpers ---------------------- */
function showModal(message) {
  // Simple popup or fallback console log
  const modal = document.getElementById("modal");
  const modalText = document.getElementById("modal-text");

  if (modal && modalText) {
    modalText.textContent = message;
    modal.style.display = "flex";
    setTimeout(() => (modal.style.display = "none"), 3000);
  } else {
    console.log("[INFO]", message);
  }
}
