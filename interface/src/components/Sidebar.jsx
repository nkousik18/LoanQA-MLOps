import React from "react";

const Sidebar = ({ open, onClose, title, content }) => {
  if (!open) return null;

  return (
    <div
      style={{
        position: "fixed",
        right: 0,
        top: 0,
        height: "100%",
        width: "30%",
        background: "#fff",
        borderLeft: "2px solid #e0e0e0",
        boxShadow: "-4px 0 8px rgba(0,0,0,0.1)",
        padding: "20px",
        overflowY: "auto",
        zIndex: 1000,
      }}
    >
      <h3>{title}</h3>
      <pre style={{ whiteSpace: "pre-wrap", fontFamily: "Inter, sans-serif" }}>
        {content || "No content yet."}
      </pre>
      <button onClick={onClose} style={{ marginTop: "10px" }}>Close</button>
    </div>
  );
};

export default Sidebar;
