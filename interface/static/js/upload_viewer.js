/* ===========================================================
   upload_viewer.js
   -----------------------------------------------------------
   Handles user file uploads, sends them to Flask backend,
   and loads the processed document into the PDF.js viewer.
   =========================================================== */

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const pdfViewer = document.getElementById("pdf-viewer");
const imageViewer = document.getElementById("image-viewer");

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) return alert("Please choose a file first.");

  const formData = new FormData();
  formData.append("file", file);

  showModal("⏳ Uploading and processing document...");

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
    const data = await response.json();

    console.log("[UPLOAD SUCCESS]", data);
    showModal("✅ Document processed successfully.<br>You can now interact with it.");

    // Display the uploaded file (PDF or image)
    const fileName = data.filename.toLowerCase();
    displayDocument(fileName, URL.createObjectURL(file));

  } catch (error) {
    console.error("[UPLOAD ERROR]", error);
    showModal(`❌ Upload failed: ${error.message}`);
  }
});

/**
 * Loads the document in viewer.
 * Uses iframe for PDFs and <img> for images.
 */
function displayDocument(fileName, fileURL) {
  if (fileName.endsWith(".pdf")) {
    pdfViewer.src = fileURL;
    pdfViewer.style.display = "block";
    imageViewer.style.display = "none";
  } else if (/\.(jpg|jpeg|png)$/i.test(fileName)) {
    imageViewer.src = fileURL;
    imageViewer.style.display = "block";
    pdfViewer.style.display = "none";
  } else {
    showModal("⚠️ Unsupported file format.");
  }
}
