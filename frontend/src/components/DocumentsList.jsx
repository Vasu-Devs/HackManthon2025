// src/components/DocumentsList.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fastAPIService, uploadUtils } from "../services/api.js";
import {
  User,
  LogOut,
  LayoutDashboard,
  MessageCircle,
  FileText,
  Trash2,
  Download,
  Clock,
  CheckCircle,
  AlertCircle,
  Shield,
  Upload,
} from "lucide-react";

const DocumentsList = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const navigate = useNavigate();

  // Fetch documents from backend
  const fetchDocuments = async () => {
    try {
      const token = localStorage.getItem("authToken");
      const response = await fetch("http://127.0.0.1:8000/documents", {
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (response.ok) {
        const data = await response.json();
        const enhanced = data.map((doc, idx) => ({
          id: idx + 1,
          name: doc.name,
          size: doc.size || "—",
          date: doc.date || new Date().toISOString().split("T")[0],
          time: doc.time || new Date().toLocaleTimeString(),
          status: doc.status,
        }));
        setDocuments(enhanced);
      } else {
        console.error("Failed to fetch documents");
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  // Navigation functions
  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("userRole");
    localStorage.removeItem("userRegNo");
    navigate("/");
  };

  const handleGoToDashboard = () => {
    navigate("/dashboard");
  };

  const handleGoToChat = () => {
    navigate("/chat");
  };

  const handleGoToAnalytics = () => {
    navigate("/analytics");
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest(".profile-dropdown")) {
        setShowProfileDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Delete document
  const handleDelete = async (filename) => {
    if (window.confirm(`Are you sure you want to delete ${filename}?`)) {
      try {
        const token = localStorage.getItem("authToken");
        const response = await fetch(
          `http://127.0.0.1:8000/delete_doc/${filename}`,
          {
            method: "DELETE",
            headers: {
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
          }
        );
        if (response.ok) {
          setDocuments((prev) => prev.filter((doc) => doc.name !== filename));
        } else {
          alert("Failed to delete document");
        }
      } catch (error) {
        console.error("Error deleting document:", error);
        alert("Error deleting document");
      }
    }
  };

  // Approve document
  const handleApprove = async (filename) => {
    try {
      const token = localStorage.getItem("authToken");
      const response = await fetch(
        `http://127.0.0.1:8000/approve_doc/${filename}`,
        {
          method: "POST",
          headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        }
      );
      if (response.ok) {
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.name === filename ? { ...doc, status: "verified" } : doc
          )
        );
      } else {
        alert("Failed to approve document");
      }
    } catch (error) {
      console.error("Error approving document:", error);
      alert("Error approving document");
    }
  };

  // Status badges
  const getStatusBadge = (status) => {
    switch (status) {
      case "verified":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <CheckCircle size={12} />
            Verified
          </span>
        );
      case "processing":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
            <Clock size={12} />
            Processing
          </span>
        );
      case "failed":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
            <AlertCircle size={12} />
            Failed
          </span>
        );
      default:
        return status;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Document Management
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Manage all uploaded policy documents and files
          </p>
        </div>

        {/* Upload button */}
        <label className="flex items-center gap-2 cursor-pointer px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
          <Upload size={16} />
          <span>Upload PDF</span>
          <input
            type="file"
            accept="application/pdf"
            hidden
            onChange={async (e) => {
              const file = e.target.files[0];
              if (!file) return;
              try {
                await uploadUtils.uploadWithProgress(file, (s) => {
                  setUploadProgress((prev) => ({
                    ...prev,
                    [file.name]: s.progress,
                  }));
                });
                fetchDocuments();
              } catch (err) {
                console.error(err);
                alert(`❌ Upload failed: ${err.message}`);
              }
            }}
          />
        </label>

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
                onClick={handleGoToAnalytics}
                className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors"
              >
                <FileText size={16} />
                Go to Analytics
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
      </header>

      {/* Main Content */}
      <main className="p-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
            <h2 className="text-lg font-semibold text-gray-900">
              All Uploaded Documents
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Total: {documents.length} documents
            </p>
          </div>

          {/* Documents Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size / Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {documents.map((doc) => (
                  <tr key={doc.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="h-5 w-5 text-gray-400 mr-3" />
                        <div className="text-sm font-medium text-gray-900">
                          {doc.name}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {doc.date}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {doc.time}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {uploadProgress[doc.name] && doc.status === "processing" ? (
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all"
                            style={{ width: `${uploadProgress[doc.name]}%` }}
                          ></div>
                        </div>
                      ) : (
                        doc.size
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {getStatusBadge(doc.status)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => fastAPIService.downloadDocument(doc.name)}
                          className="text-blue-600 hover:text-blue-800 p-1 rounded transition-colors"
                        >
                          <Download size={16} />
                        </button>

                        {doc.status === "processing" && (
                          <button
                            onClick={() => handleApprove(doc.name)}
                            className="text-green-600 hover:text-green-800 p-1 rounded transition-colors"
                            title="Approve document"
                          >
                            <Shield size={16} />
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(doc.name)}
                          className="text-red-600 hover:text-red-800 p-1 rounded transition-colors"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {documents.length === 0 && (
            <div className="text-center py-12">
              <FileText className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                No documents
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Upload some documents to get started.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default DocumentsList;
