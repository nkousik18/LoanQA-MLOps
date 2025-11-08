/* ===========================================================
   selection_handler.js
   -----------------------------------------------------------
   Handles text selection and button-triggered API calls.
   Works with callLLM() from api_calls.js.
   =========================================================== */

let selectedText = "";

document.addEventListener("DOMContentLoaded", () => {
  console.log("[Selection Handler] ✅ Loaded and ready.");

  const viewer = document.getElementById("text-viewer");
  const toolbar = document.getElementById("toolbar");

  // Detect text selection
  viewer.addEventListener("mouseup", () => {
    const text = window.getSelection().toString().trim();
    if (text.length > 0) {
      selectedText = text;
      console.log(`[Selection Handler] Text selected: "${text.slice(0, 80)}..."`);
      toolbar.style.display = "flex";
    }
  });

  // === Button Event Listeners ===
  document.getElementById("summary-btn").addEventListener("click", async () => {
    if (!selectedText) return showModal("⚠️ Please select text first.");
    console.log("[Button] Summary clicked");
    const res = await callLLM("summary", selectedText);
    if (res?.summary) showModal(res.summary, "🧾 Summary");
  });

  document.getElementById("translate-btn").addEventListener("click", async () => {
    if (!selectedText) return showModal("⚠️ Please select text first.");
    console.log("[Button] Translate clicked");
    const lang = prompt("🌍 Enter target language (e.g., 'French', 'Spanish', 'Hindi'):");
    if (!lang) return;
    const res = await callLLM("translate", selectedText, lang);
    if (res?.translation) showModal(res.translation, `🌍 Translation (${lang})`);
  });

  document.getElementById("explain-btn").addEventListener("click", async () => {
    if (!selectedText) return showModal("⚠️ Please select text first.");
    console.log("[Button] Explain clicked");
    const res = await callLLM("explain", selectedText);
    if (res?.explanation) showModal(res.explanation, "💡 Explanation");
  });
});
