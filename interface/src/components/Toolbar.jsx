import React from "react";
import { processText } from "../api/api";

const Toolbar = ({ selection, setSidebar }) => {
  const { text, x, y } = selection;
  const style = {
    position: "absolute",
    top: `${y - 40}px`,
    left: `${x}px`,
    background: "#fff",
    border: "1px solid #ccc",
    borderRadius: "6px",
    display: "flex",
    gap: "10px",
    padding: "6px",
    zIndex: 1000,
    boxShadow: "0 2px 5px rgba(0,0,0,0.15)",
  };

  const handleAction = async (action) => {
    setSidebar({
      open: true,
      title: `${action.toUpperCase()} in progress...`,
      content: "Please wait while the model processes your request...",
    });
    const data = await processText(text, action);
    setSidebar({
      open: true,
      title: `${action.toUpperCase()} Result`,
      content: data.result || "No response from model.",
    });
  };

  return (
    <div style={style}>
      <button onClick={() => handleAction("summary")}>Summarize</button>
      <button onClick={() => handleAction("translate")}>Translate</button>
      <button onClick={() => handleAction("explain")}>Explain</button>
    </div>
  );
};

export default Toolbar;
