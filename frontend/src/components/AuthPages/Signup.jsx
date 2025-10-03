import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import StudentBlackPanel from "./StudentModal/StudentBlackPanel";
import StudentWhitePanel from "./StudentModal/StudentWhitePanel";
import AdminBlackPanel from "./AdminModal/AdminBlackPanel";
import AdminWhitePanel from "./AdminModal/AdminWhitePanel";

const Signup = ({ isOpen, onClose }) => {
  const [registrationNumber, setRegistrationNumber] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // states
  const [isStudent, setIsStudent] = useState(true);
  const [isBlackLeft, setIsBlackLeft] = useState(true); // toggle on each switch

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:4000/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          regNo: registrationNumber,
          password: password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store token in localStorage
        localStorage.setItem("authToken", data.token);
        localStorage.setItem("userRole", data.role);
        localStorage.setItem("userRegNo", data.regNo);

        // Route based on role
        if (data.role === "admin") {
          navigate("/dashboard");
        } else if (data.role === "user") {
          navigate("/chat");
        }

        // Close modal
        onClose();
      } else {
        setError(data.error || "Login failed");
      }
    } catch (error) {
      console.error("Login error:", error);
      setError("Network error. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSwitch = () => {
    setIsStudent((prev) => !prev);
    setIsBlackLeft((prev) => !prev); // flip order
  };

  if (!isOpen) return null;

  const BlackPanel = isStudent ? StudentBlackPanel : AdminBlackPanel;
  const WhitePanel = isStudent ? StudentWhitePanel : AdminWhitePanel;

  return (
    <div
      className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="bg-white rounded-lg shadow-2xl overflow-hidden relative flex"
        style={{ width: "793px", height: "484px" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors z-50 cursor-pointer"
        >
          ✕
        </button>

        {/* Panels container */}
        <div className="flex w-full h-full relative overflow-hidden">
          {/* Black panel (slides OVER) */}
          <motion.div
            layout
            transition={{ duration: 0.6, ease: "easeInOut" }}
            className={`w-1/2 relative ${
              isBlackLeft ? "order-1" : "order-2"
            } z-20`}
          >
            <div className="absolute inset-0 bg-gray-900 text-white flex flex-col justify-center">
              <BlackPanel onSwitch={handleSwitch} />
            </div>
          </motion.div>

          {/* White panel (slides BELOW) */}
          <motion.div
            layout
            transition={{ duration: 0.6, ease: "easeInOut" }}
            className={`w-1/2 relative ${
              isBlackLeft ? "order-2" : "order-1"
            } z-10`}
          >
            <div className="absolute inset-0 bg-white flex flex-col justify-center">
              <WhitePanel
                registrationNumber={registrationNumber}
                setRegistrationNumber={setRegistrationNumber}
                password={password}
                setPassword={setPassword}
                showPassword={showPassword}
                handleSubmit={handleSubmit}
                isLoading={isLoading}
                error={error}
              />
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Signup;
