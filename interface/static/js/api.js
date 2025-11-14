// src/api/api.js
const BASE_URL = "http://localhost:5001";

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: formData });
  return res.json();
}

export async function processText(text, action, language = "English") {
  const res = await fetch(`${BASE_URL}/process_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, action, language })
  });
  return res.json();
}
