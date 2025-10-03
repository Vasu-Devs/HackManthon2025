import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DashboardLayout from "./components/DashboardLayout";
import { LandingPage } from "./components/LandingPage";
import ChatInterface from "./components/ChatInterface";
import Analytics from "./components/Analytics";
import DocumentsList from "./components/DocumentsList";
import VerificationInterface from "./components/VerificationInterface";  // ✅ import


export const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-white text-black">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={<DashboardLayout />} />
          <Route path="/chat" element={<ChatInterface />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/policies" element={<DocumentsList />} />
          <Route path="/verification" element={<VerificationInterface />} /> 
        </Routes>
      </div>
    </Router>

    // <div className="min-h-screen bg-white text-black">
    //   <Analytics />
    // </div>
  );
};
