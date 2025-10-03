// src/pages/Analytics.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { User, LogOut, LayoutDashboard, MessageCircle } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";

const Analytics = () => {
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const navigate = useNavigate();

  // Logout function
  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("userRole");
    localStorage.removeItem("userRegNo");
    navigate("/");
  };

  // Navigation functions
  const handleGoToDashboard = () => {
    navigate("/dashboard");
  };

  const handleGoToChat = () => {
    navigate("/chat");
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest(".profile-dropdown")) {
        setShowProfileDropdown(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Dummy Data
  const queriesData = [
    { name: "Success", value: 420 },
    { name: "Failed", value: 80 },
  ];

  const responseTimeData = [
    { day: "Mon", time: 1.2 },
    { day: "Tue", time: 1.6 },
    { day: "Wed", time: 1.1 },
    { day: "Thu", time: 1.4 },
    { day: "Fri", time: 1.8 },
  ];

  const categoryData = [
    { name: "Fees", value: 200 },
    { name: "Scholarships", value: 150 },
    { name: "Timetable", value: 120 },
    { name: "Exams", value: 100 },
  ];

  const languageData = [
    { name: "English", value: 300 },
    { name: "Hindi", value: 180 },
    { name: "Marathi", value: 120 },
    { name: "Other", value: 60 },
  ];

  const activeUsers = [
    ["Time", "Mon", "Tue", "Wed", "Thu", "Fri"],
    ["8 AM", 5, 8, 6, 4, 7],
    ["12 PM", 12, 15, 10, 9, 13],
    ["4 PM", 20, 18, 22, 19, 17],
    ["8 PM", 10, 8, 9, 11, 12],
  ];

  const COLORS = ["#4CAF50", "#F44336", "#2196F3", "#FFC107"];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Analytics Dashboard
            </h1>
            <p className="text-sm text-gray-600 mt-1">
              Monitor system performance and user insights
            </p>
          </div>

          {/* Profile Dropdown */}
          <div className="relative profile-dropdown">
            <button
              onClick={() => setShowProfileDropdown(!showProfileDropdown)}
              className="flex items-center gap-3 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                <User size={18} className="text-white" />
              </div>
              <div className="text-left">
                <div className="text-sm font-medium text-gray-900">
                  {localStorage.getItem("userRegNo") || "Admin"}
                </div>
                <div className="text-xs text-gray-500">Administrator</div>
              </div>
            </button>

            {/* Dropdown Menu */}
            {showProfileDropdown && (
              <div className="absolute right-0 top-full mt-2 bg-white rounded-lg shadow-lg border border-gray-200 py-2 min-w-[200px] z-50">
                <div className="px-4 py-2 text-sm text-gray-600 border-b border-gray-100">
                  {localStorage.getItem("userRegNo") || "Admin"}
                </div>
                <button
                  onClick={handleGoToDashboard}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors"
                >
                  <LayoutDashboard size={16} />
                  Go to Dashboard
                </button>
                <button
                  onClick={handleGoToChat}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors"
                >
                  <MessageCircle size={16} />
                  Go to Chat
                </button>
                <button
                  onClick={handleLogout}
                  className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
                >
                  <LogOut size={16} />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Analytics Content */}
      <main className="p-6">
        {/* CATEGORY 1 */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 text-gray-800 border-b border-gray-200 pb-2">
            System Performance Overview
          </h2>
          <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
            {/* Total Queries Card */}
            <div className="bg-white shadow-lg rounded-xl p-6 flex flex-col items-center hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Total Queries
              </h3>
              <p className="text-3xl font-bold text-blue-600">500</p>
              <p className="text-sm text-gray-500 mt-1">Last 30 days</p>
            </div>

            {/* Donut Chart */}
            <div className="bg-white shadow-lg rounded-xl p-4 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Success vs Failed
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={queriesData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={70}
                    dataKey="value"
                  >
                    {queriesData.map((entry, index) => (
                      <Cell key={index} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Avg Response Time */}
            <div className="bg-white shadow-lg rounded-xl p-4 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Avg Response Time (s)
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={responseTimeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="time"
                    stroke="#2196F3"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Language Usage */}
            <div className="bg-white shadow-lg rounded-xl p-4 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Queries by Language
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={languageData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#4CAF50" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Active User Times Heatmap */}
            <div className="bg-white shadow-lg rounded-xl p-4 col-span-2 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Active User Times
              </h3>
              <div className="grid grid-cols-6 gap-2 text-center text-sm">
                {activeUsers.map((row, i) => (
                  <React.Fragment key={i}>
                    {row.map((cell, j) => (
                      <div
                        key={j}
                        className={`p-2 rounded ${
                          i === 0 || j === 0
                            ? "bg-gray-200 font-semibold"
                            : `bg-blue-${100 + cell * 100}`
                        }`}
                      >
                        {cell}
                      </div>
                    ))}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </section>
        </div>

        {/* CATEGORY 2 */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 text-gray-800 border-b border-gray-200 pb-2">
            Query Insights & Analytics
          </h2>
          <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Queries by Category */}
            <div className="bg-white shadow-lg rounded-xl p-4 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Queries by Category
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={categoryData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label
                  >
                    {categoryData.map((entry, index) => (
                      <Cell key={index} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* FAQ List */}
            <div className="bg-white shadow-lg rounded-xl p-4 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Top FAQs
              </h3>
              <ul className="list-disc pl-6 text-gray-700 space-y-2">
                <li>When is the fee deadline?</li>
                <li>How do I apply for scholarships?</li>
                <li>Where to find the timetable?</li>
                <li>How to check exam results?</li>
              </ul>
            </div>

            {/* Escalations */}
            <div className="bg-white shadow-lg rounded-xl p-6 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Escalations to Staff
              </h3>
              <p className="text-3xl font-bold text-red-500">12</p>
              <p className="text-gray-500 text-sm mt-2">Last 7 days</p>
            </div>

            {/* Recent Queries */}
            <div className="bg-white shadow-lg rounded-xl p-4 hover:shadow-xl transition-shadow border border-gray-100">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">
                Recent Queries
              </h3>
              <table className="w-full text-sm text-left text-gray-600">
                <thead>
                  <tr className="border-b">
                    <th className="p-2">Query</th>
                    <th className="p-2">Status</th>
                    <th className="p-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="p-2">Fee deadline?</td>
                    <td className="p-2 text-green-600">Answered</td>
                    <td className="p-2">2m ago</td>
                  </tr>
                  <tr>
                    <td className="p-2">Exam schedule?</td>
                    <td className="p-2 text-red-600">Failed</td>
                    <td className="p-2">5m ago</td>
                  </tr>
                  <tr>
                    <td className="p-2">Scholarship form?</td>
                    <td className="p-2 text-green-600">Answered</td>
                    <td className="p-2">10m ago</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default Analytics;
