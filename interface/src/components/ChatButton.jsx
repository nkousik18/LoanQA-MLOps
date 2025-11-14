// src/components/ChatButton.jsx
import React from "react";

const ChatButton = () => (
  <a
    href="http://localhost:8080/chat"
    target="_blank"
    rel="noopener noreferrer"
    className="chatbot-btn"
  >
    💬
  </a>
);

export default ChatButton;   // ✅ this line fixes the error
