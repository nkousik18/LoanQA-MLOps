////const API_URL = "http://127.0.0.1:8501";
////
////async function callAPI(endpoint, text) {
////    try {
////        const response = await fetch(`${API_URL}/${endpoint}`, {
////            method: "POST",
////            headers: {
////                "Content-Type": "application/json"
////            },
////            body: JSON.stringify({ text: text })
////        });
////        const data = await response.json();
////        document.getElementById("output").innerText = data.result;
////    } catch (error) {
////        document.getElementById("output").innerText = "Error: " + error;
////    }
////}
////
////document.getElementById("translateBtn").addEventListener("click", () => {
////    const text = document.getElementById("inputText").value;
////    callAPI("translate", text);
////});
////
////document.getElementById("summarizeBtn").addEventListener("click", () => {
////    const text = document.getElementById("inputText").value;
////    callAPI("summarize", text);
////});
////
////document.getElementById("ttsBtn").addEventListener("click", () => {
////    const text = document.getElementById("inputText").value;
////    callAPI("tts", text);
////});
////
////document.getElementById("mathBtn").addEventListener("click", () => {
////    const text = document.getElementById("inputText").value;
////    callAPI("math_explain", text);
////});
//
//// URL of your backend
//const API_URL = "http://127.0.0.1:8501";
//
//// Helper function to call backend (e.g., summarize or math)
//async function callAPI(endpoint, text) {
//    try {
//        const response = await fetch(`${API_URL}/${endpoint}`, {
//            method: "POST",
//            headers: {
//                "Content-Type": "application/json"
//            },
//            body: JSON.stringify({ text: text })
//        });
//        const data = await response.json();
//        document.getElementById("output").innerText = data.result;
//
//        // Voice readout
//        const utterance = new SpeechSynthesisUtterance(data.result);
//        speechSynthesis.speak(utterance);
//
//    } catch (error) {
//        document.getElementById("output").innerText = "Error: " + error;
//    }
//}
//
//// Button event listeners
//
//// Summarize
//document.getElementById("summarizeBtn").addEventListener("click", () => {
//    const text = document.getElementById("inputText").value;
//    if (!text) return alert("Please upload a file or enter text!");
//    callAPI("summarize", text);
//});
//
//// Math Explain
//document.getElementById("mathBtn").addEventListener("click", () => {
//    const text = document.getElementById("inputText").value;
//    if (!text) return alert("Please upload a file or enter text!");
//    callAPI("math_explain", text);
//});
//
//// Voice Assistant (reads text aloud)
//document.getElementById("ttsBtn").addEventListener("click", () => {
//    const text = document.getElementById("inputText").value;
//    if (!text) return alert("Please upload a file or enter text!");
//
//    const utterance = new SpeechSynthesisUtterance(text);
//    speechSynthesis.speak(utterance);
//});
//
//// Upload File
//document.getElementById("uploadBtn").addEventListener("click", () => {
//    const fileInput = document.createElement("input");
//    fileInput.type = "file";
//    fileInput.accept = ".txt,.pdf,.docx";
//    fileInput.onchange = async (e) => {
//        const file = e.target.files[0];
//        if (!file) return;
//
//        const reader = new FileReader();
//        reader.onload = function(event) {
//            document.getElementById("inputText").value = event.target.result;
//        };
//        reader.readAsText(file);
//    };
//    fileInput.click();
//});

////Attempt 2
//// Your backend URL
//const API_URL = "http://127.0.0.1:8501"; // update if different
//
//// ---------------- PDF Helper ----------------
//async function readPDF(file) {
//    const typedArray = new Uint8Array(await file.arrayBuffer());
//    const pdf = await pdfjsLib.getDocument(typedArray).promise;
//    let text = "";
//
//    for (let i = 1; i <= pdf.numPages; i++) {
//        const page = await pdf.getPage(i);
//        const content = await page.getTextContent();
//        content.items.forEach(item => text += item.str + " ");
//        text += "\n"; // separate pages
//    }
//    return text;
//}
//
//// ---------------- Upload Button ----------------
//document.getElementById("uploadBtn").addEventListener("click", () => {
//    const fileInput = document.createElement("input");
//    fileInput.type = "file";
//    fileInput.accept = ".txt,.pdf";
//    fileInput.onchange = async (e) => {
//        const file = e.target.files[0];
//        if (!file) return;
//
//        if (file.type === "application/pdf") {
//            const pdfText = await readPDF(file);
//            document.getElementById("inputText").value = pdfText;
//        } else {
//            const reader = new FileReader();
//            reader.onload = function(event) {
//                document.getElementById("inputText").value = event.target.result;
//            };
//            reader.readAsText(file);
//        }
//    };
//    fileInput.click();
//});
//
//// ---------------- API Helper ----------------
//async function callAPI(endpoint, text) {
//    try {
//        const response = await fetch(`${API_URL}/${endpoint}`, {
//            method: "POST",
//            headers: { "Content-Type": "application/json" },
//            body: JSON.stringify({ text: text })
//        });
//        const data = await response.json();
//        document.getElementById("output").innerText = data.result;
//
//        // Voice readout
//        const utterance = new SpeechSynthesisUtterance(data.result);
//        speechSynthesis.speak(utterance);
//
//    } catch (error) {
//        document.getElementById("output").innerText = "Error: " + error;
//    }
//}
//
//// ---------------- Summarize ----------------
//document.getElementById("summarizeBtn").addEventListener("click", () => {
//    const text = document.getElementById("inputText").value;
//    if (!text) return alert("Please upload a file or enter text!");
//    callAPI("summarize", text);
//});
//
//// ---------------- Math Explain ----------------
//document.getElementById("mathBtn").addEventListener("click", () => {
//    const text = document.getElementById("inputText").value;
//    if (!text) return alert("Please upload a file or enter text!");
//    callAPI("math_explain", text);
//});
//
//// ---------------- Voice Assistant ----------------
//document.getElementById("ttsBtn").addEventListener("click", () => {
//    const text = document.getElementById("inputText").value;
//    if (!text) return alert("Please upload a file or enter text!");
//    const utterance = new SpeechSynthesisUtterance(text);
//    speechSynthesis.speak(utterance);
//});

//Attempt 3
const API_URL = "http://127.0.0.1:8501";

// PDF reader
async function readPDF(file) {
    const typedArray = new Uint8Array(await file.arrayBuffer());
    const pdf = await pdfjsLib.getDocument({ data: typedArray }).promise;
    let text = "";
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        content.items.forEach(item => text += item.str + " ");
        text += "\n";
    }
    return text.trim();
}

// Upload
document.getElementById("uploadBtn").addEventListener("click", () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".txt,.pdf";
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.type === "application/pdf") {
            const text = await readPDF(file);
            document.getElementById("inputText").value = text || "No text found in PDF!";
        } else {
            const reader = new FileReader();
            reader.onload = function(ev) {
                document.getElementById("inputText").value = ev.target.result;
            };
            reader.readAsText(file);
        }
    };
    input.click();
});

// API call helper
async function callAPI(endpoint, text) {
    try {
        const response = await fetch(`${API_URL}/${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        document.getElementById("output").innerText = data.result;

        // Speak result
        const utter = new SpeechSynthesisUtterance(data.result);
        speechSynthesis.speak(utter);
    } catch (err) {
        document.getElementById("output").innerText = "Error: " + err;
    }
}

// Summarize
document.getElementById("summarizeBtn").addEventListener("click", () => {
    const text = document.getElementById("inputText").value.trim();
    if (!text) return alert("Upload PDF or enter text!");
    callAPI("summarize", text);
});

// Math Explain
document.getElementById("mathBtn").addEventListener("click", () => {
    const text = document.getElementById("inputText").value.trim();
    if (!text) return alert("Upload PDF or enter text!");
    callAPI("math_explain", text);
});

// Voice Assistant
document.getElementById("ttsBtn").addEventListener("click", () => {
    const text = document.getElementById("inputText").value.trim();
    if (!text) return alert("Upload PDF or enter text!");
    const utter = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utter);
});
