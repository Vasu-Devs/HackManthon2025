import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { LandingPage } from "./components/HomePage/LandingPage";
function App() {
  return (
    <Router>
      <div className="min-h-screen bg-white text-black">
        <Routes>
          <Route path="/" element={<LandingPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
