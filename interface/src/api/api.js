// src/api/api.js
// Centralized API utilities for LoanDoc Intelligence Interface

const BASE_URL = "http://localhost:8080";

/**
 * Uploads a PDF to backend → triggers extraction + vectorstore index.
 */
export async function processPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/api/process_pdf`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to process PDF");
  return res.json();
}

/**
 * Handles text-level actions: summarize, translate, explain.
 */
export async function processText(text, action, language = "English") {
  const res = await fetch(`${BASE_URL}/api/process_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, action, language }),
  });

  if (!res.ok) throw new Error("Failed to process text action");
  return res.json();
}

/**
 * Requests precise glyph-level text coordinates (for PDF.js overlay)
 * Backend automatically caches identical PDFs using SHA256 hash.
 */
export async function getTextMap(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/api/get_text_map`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to fetch text map");
  return res.json();
}
