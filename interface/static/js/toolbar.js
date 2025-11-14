const toolbar = document.getElementById("toolbar");
const summaryBtn = document.getElementById("summaryBtn");
const translateBtn = document.getElementById("translateBtn");
const explainBtn = document.getElementById("explainBtn");

document.addEventListener("mouseup", () => {
  const selectedText = window.getSelection().toString().trim();
  if (selectedText.length > 0) {
    const sel = window.getSelection().getRangeAt(0);
    const rect = sel.getBoundingClientRect();
    toolbar.style.left = rect.x + "px";
    toolbar.style.top = rect.y - 40 + "px";
    toolbar.classList.remove("hidden");
  } else {
    toolbar.classList.add("hidden");
  }
});

summaryBtn.addEventListener("click", () => sendAction("summary"));
translateBtn.addEventListener("click", async () => {
  const lang = prompt("Enter target language (e.g., French, Hindi, Spanish):", "French");
  sendAction("translate", lang);
});
explainBtn.addEventListener("click", () => sendAction("explain"));

async function sendAction(action, language = "English") {
  const text = window.getSelection().toString();
  const res = await fetch("/process_text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, action, language })
  });
  const data = await res.json();
  alert(data.result);
  toolbar.classList.add("hidden");
}
