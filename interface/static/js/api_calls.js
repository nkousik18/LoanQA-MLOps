/* ===========================================================
   api_calls.js
   -----------------------------------------------------------
   Handles API calls to Flask backend for:
   - Summary
   - Translation
   - Explanation
   Uses showResponse() (persistent text box) for displaying results.
   =========================================================== */

const BASE_URL = "http://127.0.0.1:8000/api";

/**
 * Generic POST request handler for LLM endpoints.
 * @param {string} endpoint - One of: "summary", "translate", "explain"
 * @param {string} text - The selected text to send to the backend.
 * @param {string} [lang] - Target language (for translate only).
 */
async function callLLM(endpoint, text, lang = "en") {
  if (!text || text.trim().length === 0) {
    console.warn("[API] No text provided to LLM endpoint.");
    showResponse("⚠️ Error", "Please select some text before running this action.");
    return null;
  }

  try {
    // Build request body
    const payload =
      endpoint === "translate" ? { text: text, lang: lang } : { text: text };

    console.log(`[API] Sending request to /${endpoint} ...`);

    const response = await fetch(`${BASE_URL}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    // Parse JSON safely
    const data = await response.json();
    console.log(`[API] ✅ /${endpoint} response:`, data);

    // Display backend response content in persistent response box
    if (endpoint === "translate" && data.translation) {
      showResponse(`🌍 Translation (${lang})`, data.translation);
    } else if (endpoint === "summary" && data.summary) {
      showResponse("🧾 Summary", data.summary);
    } else if (endpoint === "explain" && data.explanation) {
      showResponse("💡 Explanation", data.explanation);
    } else if (data.message) {
      showResponse("✅ Result", data.message);
    } else {
      showResponse("ℹ️ Notice", "Request succeeded, but no text was returned.");
    }

    return data;

  } catch (error) {
    console.error(`[API Error] /${endpoint}:`, error);
    showResponse("❌ Error", `Failed to process ${endpoint} request.\n${error.message}`);
    return null;
  }
}

/* ===========================================================
   showResponse() is provided by index.html
   This script assumes window.showResponse(title, message)
   is defined globally.
   =========================================================== */
