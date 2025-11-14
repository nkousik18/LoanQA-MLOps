import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import PdfViewer from "./components/PdfViewer";
import Sidebar from "./components/Sidebar";
import Toolbar from "./components/Toolbar";
import "./index.css";  // ✅ includes PdfViewer styling

/**
 * App.jsx
 * --------------------------------------------
 * Root component for LoanDoc Intelligence Interface
 * Handles routing and layout structure.
 */

const App = () => {
  return (
    <Router>
      <Routes>
        {/* Default route → PDF Viewer */}
        <Route path="/" element={<PdfViewer />} />

        {/* Optional future routes */}
        <Route path="/chat" element={<Sidebar />} />
        <Route path="/toolbar" element={<Toolbar />} />

        {/* Redirect invalid routes */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;
