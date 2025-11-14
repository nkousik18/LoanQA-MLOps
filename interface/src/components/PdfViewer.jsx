import React, { useState, useRef } from "react";
import * as pdfjsLib from "pdfjs-dist";
import { getTextMap } from "../api/api";
import "./../index.css";

/**
 * LoanDoc Intelligence Viewer
 * ---------------------------------------------
 * - Uploads PDF to Flask backend
 * - Fetches glyph geometry from PyMuPDF
 * - Renders pages with selectable text overlay
 */

pdfjsLib.GlobalWorkerOptions.workerSrc = `${window.location.origin}/pdf.worker.min.js`;

const PdfViewer = () => {
  const [fileName, setFileName] = useState("");
  const [debugMode, setDebugMode] = useState(false);
  const viewerRef = useRef(null);

  /** ---------------------------
   * Handle PDF Upload
   * --------------------------- */
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    console.log("📤 Uploading to backend...");

    try {
      console.log("🧩 Fetching text geometry from PyMuPDF service...");
      const mapResponse = await getTextMap(file);

      if (!mapResponse.pages || !mapResponse.pages.length) {
        console.error("❌ No text geometry received.");
        return;
      }

      console.log(`✅ Received ${mapResponse.pages.length} pages of text geometry.`);
      await renderAllPages(file, mapResponse.pages);
    } catch (err) {
      console.error("❌ Error rendering PDF:", err);
    }
  };

  /** ---------------------------
   * Render All PDF Pages + Text Overlay
   * --------------------------- */
  const renderAllPages = async (file, pagesData) => {
    const container = viewerRef.current;
    container.innerHTML = "";

    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    for (let i = 0; i < pdf.numPages; i++) {
      const page = await pdf.getPage(i + 1);
      const scale = 1.3;
      const viewport = page.getViewport({ scale });

      // Page wrapper for proper alignment
      const pageWrapper = document.createElement("div");
      pageWrapper.style.position = "relative";
      pageWrapper.style.margin = "20px auto";
      pageWrapper.style.display = "inline-block";
      pageWrapper.style.borderRadius = "8px";
      pageWrapper.style.background = "transparent";
      pageWrapper.style.boxShadow = "0 0 8px rgba(0,0,0,0.25)";
      pageWrapper.style.overflow = "visible";
      pageWrapper.style.width = `${viewport.width}px`;
      pageWrapper.style.height = `${viewport.height}px`;
      container.appendChild(pageWrapper);

      // Canvas layer (renders PDF visuals)
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.display = "block";
      canvas.style.borderRadius = "8px";
      pageWrapper.appendChild(canvas);
      await page.render({ canvasContext: context, viewport }).promise;

      // Text layer (transparent selectable overlay)
      const textLayer = document.createElement("div");
      textLayer.className = "textLayer";
      textLayer.style.width = `${viewport.width}px`;
      textLayer.style.height = `${viewport.height}px`;
      pageWrapper.appendChild(textLayer);

      // Match PyMuPDF geometry
      const pageData = pagesData.find((p) => p.page === i + 1);
      if (pageData && Array.isArray(pageData.words)) {
        const pageHeight = pageData.height;
        pageData.words.forEach((word) => {
          const span = document.createElement("span");
          span.textContent = word.text;

          // Corrected Y-axis flip
          const flippedY = pageHeight - word.y - word.height;

          span.style.position = "absolute";
          span.style.left = `${word.x * scale}px`;
          span.style.top = `${flippedY * scale}px`;
          span.style.width = `${word.width * scale}px`;
          span.style.height = `${word.height * scale}px`;
          span.style.whiteSpace = "pre";
          span.style.color = "transparent";
          span.style.pointerEvents = "all";
          span.style.userSelect = "text";

          if (debugMode) {
            span.style.outline = "1px solid rgba(0,128,255,0.3)";
            span.style.background = "rgba(0,128,255,0.05)";
          }

          textLayer.appendChild(span);
        });
      }
    }
  };

  /** ---------------------------
   * Component UI
   * --------------------------- */
  return (
    <div>
      {/* Upload Bar */}
      <div className="upload-bar">
        <h2>📄 LoanDoc Intelligence Viewer</h2>
        <div>
          <input type="file" accept="application/pdf" onChange={handleUpload} />
          {fileName && <span style={{ marginLeft: "10px" }}>{fileName}</span>}
          <button
            onClick={() => setDebugMode((prev) => !prev)}
            style={{
              marginLeft: "10px",
              background: debugMode ? "#ffb703" : "white",
              color: debugMode ? "black" : "#004aad",
              border: "1px solid #004aad",
              borderRadius: "6px",
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            {debugMode ? "Hide Debug" : "Show Debug"}
          </button>
        </div>
      </div>

      {/* PDF Viewer */}
      <div
        id="viewerContainer"
        ref={viewerRef}
        style={{
          overflowY: "auto",
          height: "90vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          background: "#f8f9fa",
        }}
      />

      {/* Floating Chatbot Button */}
      <a href="#chatbot" className="chatbot-btn">
        💬
      </a>
    </div>
  );
};

export default PdfViewer;
