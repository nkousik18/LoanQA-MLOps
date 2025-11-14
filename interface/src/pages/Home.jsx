import React from "react";
import PdfViewer from "../components/PdfViewer";
import ChatButton from "../components/ChatButton";

const Home = () => (
  <div style={{ position: "relative", height: "100vh", overflow: "hidden" }}>
    <PdfViewer />
    <ChatButton />
  </div>
);

export default Home;
